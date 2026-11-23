/*
 This kernel is batched inverse kinematics on the GPU meant for RL training.

  * One thread solves one environment (thousands of IK problems in parallel).
  * Target Robot: 7-DOF Franka Panda-style arm, DH parameters.
  * Solver: Levenberg-Marquardt / damped least squares on the full 6D pose
            (position + orientation), adaptive damping, step clamping,
            joint-limit clamping. Warm-startable from the previous q.
  * Memory layout: structure-of-arrays (SoA) so global accesses coalesce:
        q[j * N + env]        joint j of env
        target[k * N + env]   k = 0..2 position xyz, 3..6 quaternion (w,x,y,z)

 I have an RTX 4070ti so my build params:
     nvcc -O3 -arch=sm_89 -std=c++17 ik_batch.cu -o ik_batch
     ./ik_batch [num_envs=65536] [spread_rad=0.5]
When used with py torch it should be compiled with -DIK_NO_MAIN and call ik_init_robot() once, then ik_solve_batch() with device pointers.

*/

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// This was very helpful for debugging
#define CUDA_CHECK(x)                                                              \
    do {                                                                           \
        cudaError_t err__ = (x);                                                   \
        if (err__ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                          \
                    cudaGetErrorString(err__), __FILE__, __LINE__);                \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

constexpr int DOF  = 7;
constexpr int TASK = 6;

// Robot description (constant memory, broadcast to all threads)
__constant__ float c_a[DOF];
__constant__ float c_sa[DOF];   // sin(alpha_{i-1})
__constant__ float c_ca[DOF];   // cos(alpha_{i-1})
__constant__ float c_d[DOF];
__constant__ float c_qmin[DOF];
__constant__ float c_qmax[DOF];
__constant__ float c_flange_d;

struct IKParams {
    int   max_iters = 40;
    float pos_tol   = 1e-4f;   // meters
    float ori_tol   = 1e-3f;   // ~radians (norm of the orientation error vector)
    float lambda0   = 1e-2f;   // initial damping
    float max_step  = 0.5f;    // max joint step norm per iteration (rad)
    float ori_w     = 0.5f;    // weight of orientation rows vs position rows
};

// Small device helpers
struct Pose {
    float p[3];
    float R[9];   // row-major
};

// Forward kinematics + geometric Jacobian (world frame, flange point).
__device__ __forceinline__ void fk_jac(const float (&q)[DOF], Pose& out, float (&J)[TASK][DOF])
{
    float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float p[3] = {0, 0, 0};
    float zs[DOF][3];
    float ps[DOF][3];

/*
I found the unroll effective so I sort of went crazy with it lol 
*/

#pragma unroll
    for (int i = 0; i < DOF; ++i) {
        float st, ct;
        sincosf(q[i], &st, &ct);
        const float sa = c_sa[i], ca = c_ca[i];

        // Local rotation (modified DH: RotX(alpha) TransX(a) RotZ(theta) TransZ(d))
        const float L[9] = {ct,      -st,      0.f,
                            st * ca,  ct * ca, -sa,
                            st * sa,  ct * sa,  ca};
        const float lt[3] = {c_a[i], -sa * c_d[i], ca * c_d[i]};

        // p += R * lt
        p[0] += R[0] * lt[0] + R[1] * lt[1] + R[2] * lt[2];
        p[1] += R[3] * lt[0] + R[4] * lt[1] + R[5] * lt[2];
        p[2] += R[6] * lt[0] + R[7] * lt[1] + R[8] * lt[2];

        // R = R * L
        float Rn[9];
#pragma unroll
        for (int r = 0; r < 3; ++r)
#pragma unroll
            for (int c = 0; c < 3; ++c)
                Rn[r * 3 + c] = R[r * 3 + 0] * L[0 * 3 + c] +
                                R[r * 3 + 1] * L[1 * 3 + c] +
                                R[r * 3 + 2] * L[2 * 3 + c];
#pragma unroll
        for (int k = 0; k < 9; ++k) R[k] = Rn[k];

        // Joint i rotates about z of frame i, through the origin of frame i.
        zs[i][0] = R[2]; zs[i][1] = R[5]; zs[i][2] = R[8];
        ps[i][0] = p[0]; ps[i][1] = p[1]; ps[i][2] = p[2];
    }

    // Fixed flange offset along z
    p[0] += R[2] * c_flange_d;
    p[1] += R[5] * c_flange_d;
    p[2] += R[8] * c_flange_d;

#pragma unroll
    for (int i = 0; i < DOF; ++i) {
        const float rx = p[0] - ps[i][0];
        const float ry = p[1] - ps[i][1];
        const float rz = p[2] - ps[i][2];
        const float zx = zs[i][0], zy = zs[i][1], zz = zs[i][2];
        J[0][i] = zy * rz - zz * ry;
        J[1][i] = zz * rx - zx * rz;
        J[2][i] = zx * ry - zy * rx;
        J[3][i] = zx;
        J[4][i] = zy;
        J[5][i] = zz;
    }

    out.p[0] = p[0]; out.p[1] = p[1]; out.p[2] = p[2];
#pragma unroll
    for (int k = 0; k < 9; ++k) out.R[k] = R[k];
}

// 6D pose error: [target_pos - cur_pos ; 0.5 * sum_i (cur_col_i x tgt_col_i)]
__device__ __forceinline__ void pose_error(const Pose& cur, const float (&tp)[3],
                                           const float (&tR)[9], float (&e)[TASK])
{
    e[0] = tp[0] - cur.p[0];
    e[1] = tp[1] - cur.p[1];
    e[2] = tp[2] - cur.p[2];
    float ox = 0.f, oy = 0.f, oz = 0.f;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float ax = cur.R[0 + c], ay = cur.R[3 + c], az = cur.R[6 + c];
        const float bx = tR[0 + c],    by = tR[3 + c],    bz = tR[6 + c];
        ox += ay * bz - az * by;
        oy += az * bx - ax * bz;
        oz += ax * by - ay * bx;
    }
    e[3] = 0.5f * ox; e[4] = 0.5f * oy; e[5] = 0.5f * oz;
}

__device__ __forceinline__ void quat_to_rot(float w, float x, float y, float z, float (&R)[9])
{
    const float n = rsqrtf(w * w + x * x + y * y + z * z + 1e-20f);
    w *= n; x *= n; y *= n; z *= n;
    R[0] = 1.f - 2.f * (y * y + z * z); R[1] = 2.f * (x * y - z * w);       R[2] = 2.f * (x * z + y * w);
    R[3] = 2.f * (x * y + z * w);       R[4] = 1.f - 2.f * (x * x + z * z); R[5] = 2.f * (y * z - x * w);
    R[6] = 2.f * (x * z - y * w);       R[7] = 2.f * (y * z + x * w);       R[8] = 1.f - 2.f * (x * x + y * y);
}

__device__ __forceinline__ void rot_to_quat(const float (&R)[9], float (&q)[4])
{
    const float tr = R[0] + R[4] + R[8];
    float w, x, y, z, s;
    if (tr > 0.f) {
        s = sqrtf(tr + 1.f) * 2.f;
        w = 0.25f * s; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        s = sqrtf(1.f + R[0] - R[4] - R[8]) * 2.f;
        w = (R[7] - R[5]) / s; x = 0.25f * s; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        s = sqrtf(1.f + R[4] - R[0] - R[8]) * 2.f;
        w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = 0.25f * s; z = (R[5] + R[7]) / s;
    } else {
        s = sqrtf(1.f + R[8] - R[0] - R[4]) * 2.f;
        w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = 0.25f * s;
    }
    q[0] = w; q[1] = x; q[2] = y; q[3] = z;
}

// Solve A x = b for symmetric positive definite 6x6 A (only lower triangle read).
// A is overwritten with its Cholesky factor. Returns false if not SPD.
__device__ __forceinline__ bool solve_spd6(float (&A)[TASK][TASK], const float (&b)[TASK], float (&x)[TASK])
{
#pragma unroll
    for (int j = 0; j < TASK; ++j) {
        float s = A[j][j];
#pragma unroll
        for (int k = 0; k < j; ++k) s -= A[j][k] * A[j][k];
        if (s <= 1e-12f) return false;
        const float d = sqrtf(s);
        A[j][j] = d;
        const float inv = 1.f / d;
#pragma unroll
        for (int i = j + 1; i < TASK; ++i) {
            float t = A[i][j];
#pragma unroll
            for (int k = 0; k < j; ++k) t -= A[i][k] * A[j][k];
            A[i][j] = t * inv;
        }
    }
    float y[TASK];
#pragma unroll
    for (int i = 0; i < TASK; ++i) {
        float t = b[i];
#pragma unroll
        for (int k = 0; k < i; ++k) t -= A[i][k] * y[k];
        y[i] = t / A[i][i];
    }
#pragma unroll
    for (int i = TASK - 1; i >= 0; --i) {
        float t = y[i];
#pragma unroll
        for (int k = i + 1; k < TASK; ++k) t -= A[k][i] * x[k];
        x[i] = t / A[i][i];
    }
    return true;
}

// Stateless counter-based RNG (no per-thread state init cost)
__device__ __forceinline__ uint32_t hash_u32(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
__device__ __forceinline__ float rand01(uint32_t seed, uint32_t idx, uint32_t k)
{
    return (hash_u32(seed ^ hash_u32(idx * 747796405u + k * 2891336453u)) >> 8) * (1.0f / 16777216.0f);
}

// IK kernel: one thread = one environment
__global__ void __launch_bounds__(128)
ik_kernel(int N, const float* __restrict__ target, float* __restrict__ q_io,
          float* __restrict__ err_out, int* __restrict__ iters_out, IKParams P)
{
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= N) return;

    float q[DOF];
#pragma unroll
    for (int j = 0; j < DOF; ++j) q[j] = q_io[j * N + env];

    float tp[3], tR[9];
    tp[0] = target[0 * N + env]; tp[1] = target[1 * N + env]; tp[2] = target[2 * N + env];
    quat_to_rot(target[3 * N + env], target[4 * N + env], target[5 * N + env], target[6 * N + env], tR);

    const float w[TASK] = {1.f, 1.f, 1.f, P.ori_w, P.ori_w, P.ori_w};

    Pose cur;
    float J[TASK][DOF];
    float e[TASK];
    fk_jac(q, cur, J);
    pose_error(cur, tp, tR, e);

    auto cost_of = [&](const float (&ee)[TASK]) {
        float c = 0.f;
#pragma unroll
        for (int r = 0; r < TASK; ++r) { const float v = w[r] * ee[r]; c += v * v; }
        return c;
    };
    auto converged = [&](const float (&ee)[TASK]) {
        const float pe = ee[0] * ee[0] + ee[1] * ee[1] + ee[2] * ee[2];
        const float oe = ee[3] * ee[3] + ee[4] * ee[4] + ee[5] * ee[5];
        return pe < P.pos_tol * P.pos_tol && oe < P.ori_tol * P.ori_tol;
    };

    float cost = cost_of(e);
    float lambda = P.lambda0;
    int it = 0;

    for (; it < P.max_iters; ++it) {
        if (converged(e)) break;

        // Weighted Jacobian / error
        float Jw[TASK][DOF], ew[TASK];
#pragma unroll
        for (int r = 0; r < TASK; ++r) {
            ew[r] = w[r] * e[r];
#pragma unroll
            for (int c = 0; c < DOF; ++c) Jw[r][c] = w[r] * J[r][c];
        }

        // A = Jw Jw^T + lambda^2 I   (lower triangle)
        float A[TASK][TASK];
        const float l2 = lambda * lambda;
#pragma unroll
        for (int i = 0; i < TASK; ++i)
#pragma unroll
            for (int j = 0; j <= i; ++j) {
                float s = 0.f;
#pragma unroll
                for (int k = 0; k < DOF; ++k) s += Jw[i][k] * Jw[j][k];
                A[i][j] = s + (i == j ? l2 : 0.f);
            }

        float y[TASK];
        if (!solve_spd6(A, ew, y)) { lambda = fminf(lambda * 10.f, 1e3f); continue; }

        // dq = Jw^T y, clamp step norm
        float dq[DOF];
        float n2 = 0.f;
#pragma unroll
        for (int c = 0; c < DOF; ++c) {
            float s = 0.f;
#pragma unroll
            for (int r = 0; r < TASK; ++r) s += Jw[r][c] * y[r];
            dq[c] = s;
            n2 += s * s;
        }
        if (n2 > P.max_step * P.max_step) {
            const float sc = P.max_step * rsqrtf(n2);
#pragma unroll
            for (int c = 0; c < DOF; ++c) dq[c] *= sc;
        }

        // Trial point (joint-limit clamped)
        float qn[DOF];
#pragma unroll
        for (int c = 0; c < DOF; ++c) qn[c] = fminf(fmaxf(q[c] + dq[c], c_qmin[c]), c_qmax[c]);

        Pose cur2;
        float J2[TASK][DOF];
        float e2[TASK];
        fk_jac(qn, cur2, J2);
        pose_error(cur2, tp, tR, e2);
        const float cost2 = cost_of(e2);

        if (cost2 < cost) {          // accept, trust the model more
#pragma unroll
            for (int c = 0; c < DOF; ++c) q[c] = qn[c];
#pragma unroll
            for (int r = 0; r < TASK; ++r) {
                e[r] = e2[r];
#pragma unroll
                for (int c = 0; c < DOF; ++c) J[r][c] = J2[r][c];
            }
            cost = cost2;
            lambda = fmaxf(lambda * 0.5f, 1e-6f);
        } else {                     // reject, increase damping
            lambda = fminf(lambda * 4.f, 1e3f);
        }
    }

#pragma unroll
    for (int j = 0; j < DOF; ++j) q_io[j * N + env] = q[j];
    err_out[0 * N + env] = sqrtf(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    err_out[1 * N + env] = sqrtf(e[3] * e[3] + e[4] * e[4] + e[5] * e[5]);
    iters_out[env] = it;
}

// Test-data generation: reachable targets = FK(q_start + random perturbation)
__global__ void gen_problem_kernel(int N, uint32_t seed, float spread,
                                   const float* __restrict__ q_home,
                                   float* __restrict__ q_start, float* __restrict__ target)
{
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= N) return;

    float qs[DOF], qg[DOF];
#pragma unroll
    for (int j = 0; j < DOF; ++j) {
        const float u0 = rand01(seed, env, 2 * j) * 2.f - 1.f;
        const float u1 = rand01(seed, env, 2 * j + 1) * 2.f - 1.f;
        qs[j] = fminf(fmaxf(q_home[j] + 0.1f * u0, c_qmin[j]), c_qmax[j]);   // noisy start
        qg[j] = fminf(fmaxf(qs[j] + spread * u1,   c_qmin[j]), c_qmax[j]);   // goal config
        q_start[j * N + env] = qs[j];
    }
    Pose goal;
    float J[TASK][DOF];
    fk_jac(qg, goal, J);
    float quat[4];
    rot_to_quat(goal.R, quat);
    target[0 * N + env] = goal.p[0];
    target[1 * N + env] = goal.p[1];
    target[2 * N + env] = goal.p[2];
    target[3 * N + env] = quat[0];
    target[4 * N + env] = quat[1];
    target[5 * N + env] = quat[2];
    target[6 * N + env] = quat[3];
}

// Host API
extern "C" void ik_init_robot()
{
    // This is from the Franka Panda Robot arm
    const float PI = 3.14159265358979f;
    const float a[DOF]     = {0.f, 0.f, 0.f, 0.0825f, -0.0825f, 0.f, 0.088f};
    const float d[DOF]     = {0.333f, 0.f, 0.316f, 0.f, 0.384f, 0.f, 0.f};
    const float alpha[DOF] = {0.f, -PI / 2, PI / 2, PI / 2, -PI / 2, PI / 2, PI / 2};
    const float qmin[DOF]  = {-2.8973f, -1.7628f, -2.8973f, -3.0718f, -2.8973f, -0.0175f, -2.8973f};
    const float qmax[DOF]  = { 2.8973f,  1.7628f,  2.8973f, -0.0698f,  2.8973f,  3.7525f,  2.8973f};
    float sa[DOF], ca[DOF];
    for (int i = 0; i < DOF; ++i) { sa[i] = sinf(alpha[i]); ca[i] = cosf(alpha[i]); }
    // Clean up float noise for the 90-degree twists
    for (int i = 0; i < DOF; ++i) {
        if (fabsf(sa[i]) < 1e-6f) sa[i] = 0.f;
        if (fabsf(ca[i]) < 1e-6f) ca[i] = 0.f;
    }
    const float flange = 0.107f;

    CUDA_CHECK(cudaMemcpyToSymbol(c_a, a, sizeof(a)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_sa, sa, sizeof(sa)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ca, ca, sizeof(ca)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_d, d, sizeof(d)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_qmin, qmin, sizeof(qmin)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_qmax, qmax, sizeof(qmax)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_flange_d, &flange, sizeof(flange)));
}

// d_target: [7*N] SoA, d_q: [DOF*N] SoA in/out (initial guess -> solution),
// d_err: [2*N] (pos err, ori err), d_iters: [N]. All device pointers.
extern "C" void ik_solve_batch(int N, const float* d_target, float* d_q, float* d_err,
                               int* d_iters, IKParams params, cudaStream_t stream = 0)
{
    const int block = 128;
    const int grid  = (N + block - 1) / block;
    ik_kernel<<<grid, block, 0, stream>>>(N, d_target, d_q, d_err, d_iters, params);
    CUDA_CHECK(cudaGetLastError());
}

#ifndef IK_NO_MAIN
int main(int argc, char** argv)
{
    const int   N      = (argc > 1) ? atoi(argv[1]) : 65536;
    const float spread = (argc > 2) ? (float)atof(argv[2]) : 0.5f;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d), envs=%d, goal spread=%.2f rad\n",
           prop.name, prop.major, prop.minor, N, spread);

    ik_init_robot();

    float *d_q, *d_target, *d_err, *d_home;
    int* d_iters;
    CUDA_CHECK(cudaMalloc(&d_q,      sizeof(float) * DOF * N));
    CUDA_CHECK(cudaMalloc(&d_target, sizeof(float) * 7 * N));
    CUDA_CHECK(cudaMalloc(&d_err,    sizeof(float) * 2 * N));
    CUDA_CHECK(cudaMalloc(&d_iters,  sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_home,   sizeof(float) * DOF));

    const float home[DOF] = {0.f, -0.785f, 0.f, -2.356f, 0.f, 1.571f, 0.785f};
    CUDA_CHECK(cudaMemcpy(d_home, home, sizeof(home), cudaMemcpyHostToDevice));

    gen_problem_kernel<<<(N + 127) / 128, 128>>>(N, 1234u, spread, d_home, d_q, d_target);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Keep a copy of the initial guesses so we can re-run for timing
    float* d_q0;
    CUDA_CHECK(cudaMalloc(&d_q0, sizeof(float) * DOF * N));
    CUDA_CHECK(cudaMemcpy(d_q0, d_q, sizeof(float) * DOF * N, cudaMemcpyDeviceToDevice));

    IKParams params;  // defaults

    // Warm-up
    ik_solve_batch(N, d_target, d_q, d_err, d_iters, params);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int reps = 20;
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    float total_ms = 0.f;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemcpy(d_q, d_q0, sizeof(float) * DOF * N, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(t0));
        ik_solve_batch(N, d_target, d_q, d_err, d_iters, params);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_ms += ms;
    }
    const float avg_ms = total_ms / reps;

    std::vector<float> h_err(2 * N);
    std::vector<int>   h_iters(N);
    CUDA_CHECK(cudaMemcpy(h_err.data(),   d_err,   sizeof(float) * 2 * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_iters.data(), d_iters, sizeof(int) * N,       cudaMemcpyDeviceToHost));

    int ok = 0;
    double sum_it = 0.0, sum_pos = 0.0;
    for (int i = 0; i < N; ++i) {
        const bool s = h_err[i] < params.pos_tol && h_err[N + i] < params.ori_tol;
        ok += s;
        sum_it += h_iters[i];
        if (s) sum_pos += h_err[i];
    }
    printf("Solve time: %.3f ms per batch  (%.2f M IK solves/s)\n",
           avg_ms, N / (avg_ms * 1e-3) / 1e6);
    printf("Converged: %d / %d (%.2f%%), mean iters: %.2f, mean pos err (converged): %.2e m\n",
           ok, N, 100.0 * ok / N, sum_it / N, ok ? sum_pos / ok : 0.0);

    cudaFree(d_q); cudaFree(d_q0); cudaFree(d_target); cudaFree(d_err);
    cudaFree(d_iters); cudaFree(d_home);
    return 0;
}
#endif
