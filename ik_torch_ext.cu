/*
 PyTorch binding for ik_batch.cu so I can use python first and then try implementing the RL in cuda since this
 Is a project stretch goal to implement it fully in CUDA

 Tensor conventions (all float32, CUDA, contiguous, SoA):
   q       [7, N]   joint angles  (modified in place by solve())
   target  [7, N]   x, y, z, qw, qx, qy, qz
   pose    [7, N]   same layout as target (output of fk())
*/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define IK_NO_MAIN
#include "ik_batch.cu"

__global__ void fk_pose_kernel(int N, const float* __restrict__ q_in, float* __restrict__ pose)
{
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= N) return;

    float q[DOF];
#pragma unroll
    for (int j = 0; j < DOF; ++j) q[j] = q_in[j * N + env];

    Pose p;
    float J[TASK][DOF];
    fk_jac(q, p, J);
    float quat[4];
    rot_to_quat(p.R, quat);

    pose[0 * N + env] = p.p[0];
    pose[1 * N + env] = p.p[1];
    pose[2 * N + env] = p.p[2];
    pose[3 * N + env] = quat[0];
    pose[4 * N + env] = quat[1];
    pose[5 * N + env] = quat[2];
    pose[6 * N + env] = quat[3];
}

static void check_input(const torch::Tensor& t, int64_t rows, const char* name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2 && t.size(0) == rows, name, " must have shape [", rows, ", N]");
}

void init()
{
    ik_init_robot();
}

// Solves IK in place on q. Returns {err [2,N] (pos, ori), iters [N] (int32)}.
std::vector<torch::Tensor> solve(torch::Tensor target, torch::Tensor q,
                                 int max_iters, double pos_tol, double ori_tol,
                                 double lambda0, double max_step, double ori_w)
{
    check_input(target, 7, "target");
    check_input(q, DOF, "q");
    TORCH_CHECK(target.size(1) == q.size(1), "batch size mismatch");
    const at::cuda::CUDAGuard guard(q.device());

    const int N = (int)q.size(1);
    auto err   = torch::empty({2, N}, q.options());
    auto iters = torch::empty({N}, q.options().dtype(torch::kInt32));

    IKParams p;
    p.max_iters = max_iters;
    p.pos_tol   = (float)pos_tol;
    p.ori_tol   = (float)ori_tol;
    p.lambda0   = (float)lambda0;
    p.max_step  = (float)max_step;
    p.ori_w     = (float)ori_w;

    ik_solve_batch(N, target.data_ptr<float>(), q.data_ptr<float>(),
                   err.data_ptr<float>(), iters.data_ptr<int>(), p,
                   at::cuda::getCurrentCUDAStream());
    return {err, iters};
}

torch::Tensor fk(torch::Tensor q)
{
    check_input(q, DOF, "q");
    const at::cuda::CUDAGuard guard(q.device());
    const int N = (int)q.size(1);
    auto pose = torch::empty({7, N}, q.options());
    fk_pose_kernel<<<(N + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        N, q.data_ptr<float>(), pose.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return pose;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init", &init, "Upload robot constants to the GPU (call once)");
    m.def("solve", &solve, "Batched IK (in place on q)",
          py::arg("target"), py::arg("q"),
          py::arg("max_iters") = 20, py::arg("pos_tol") = 1e-4, py::arg("ori_tol") = 1e-3,
          py::arg("lambda0") = 1e-2, py::arg("max_step") = 0.5, py::arg("ori_w") = 0.5);
    m.def("fk", &fk, "Batched forward kinematics -> pose [7, N]");
}