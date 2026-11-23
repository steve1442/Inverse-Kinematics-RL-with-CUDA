"""
ik_rl.py - GPU-vectorized end-effector reaching task + PPO, using the CUDA IK solver.

The policy outputs a Cartesian delta for the end effector (x, y, z). The CUDA IK
solver turns that into joint angles for thousands of environments at once, and the
whole loop (env, IK, policy, PPO) stays on the GPU with no host round trips.

Next goal is to try to get it put into CUDA

Run:
    python ik_rl.py --envs 4096 --iters 300

Requires pytorch, numpy, and ninja

First run compiles the extension (about a minute on my gpu). Requires a PyTorch build with CUDA
and an nvcc version that matches it.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")  # RTX 4070 Ti (Ada)
HERE = os.path.dirname(os.path.abspath(__file__))
ik = load(
    name="ik_ext",
    sources=[os.path.join(HERE, "ik_torch_ext.cu")],
    extra_cuda_cflags=["-O3", "-DIK_NO_MAIN"],
    extra_include_paths=[HERE],
    verbose=True,
)
ik.init()

# Used the DH table from the frank panda robot arm 
DOF = 7
Q_MIN = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
Q_MAX = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


class GpuReachEnv:
    """Batched reaching: move the end effector to a random 3D goal near the home pose.

    State is kept in the solver's SoA layout: q is [DOF, N], pose is [7, N].
    Orientation is held fixed at the home orientation; only position is commanded.
    """

    obs_dim = DOF + 3 + 3 + 3 + 1
    act_dim = 3

    def __init__(self, n_envs, max_steps=64, max_delta=0.03, goal_range=0.12,
                 ik_iters=20, device="cuda"):
        self.N, self.max_steps, self.max_delta = n_envs, max_steps, max_delta
        self.goal_range, self.ik_iters, self.dev = goal_range, ik_iters, device
        f = dict(device=device, dtype=torch.float32)
        self.qmin = torch.tensor(Q_MIN, **f)[:, None]
        self.qmax = torch.tensor(Q_MAX, **f)[:, None]
        self.home = torch.tensor(HOME, **f)[:, None].contiguous()

        home_pose = ik.fk(self.home)  # [7, 1]
        self.home_pos = home_pose[:3, 0].clone()
        self.home_quat = home_pose[3:, 0].clone()

        self.q = torch.zeros(DOF, n_envs, **f).contiguous()
        self.goal = torch.zeros(n_envs, 3, **f)
        self.t = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.pose = None
        self._reset(torch.ones(n_envs, dtype=torch.bool, device=device))

    def _reset(self, mask):
        N, f = self.N, dict(device=self.dev, dtype=torch.float32)
        q_new = self.home + 0.1 * (torch.rand(DOF, N, **f) * 2 - 1)
        q_new = torch.max(torch.min(q_new, self.qmax), self.qmin)
        self.q.copy_(torch.where(mask[None], q_new, self.q))  # in place: keeps layout
        goal_new = self.home_pos[None] + (torch.rand(N, 3, **f) * 2 - 1) * self.goal_range
        self.goal = torch.where(mask[:, None], goal_new, self.goal)
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
        self.pose = ik.fk(self.q)

    def obs(self):
        pos = self.pose[:3].T
        return torch.cat([self.q.T, pos, self.goal, self.goal - pos,
                          (self.t.float() / self.max_steps)[:, None]], dim=1)

    def step(self, action):
        a = action.clamp(-1.0, 1.0)
        target = torch.empty(7, self.N, device=self.dev, dtype=torch.float32)
        target[:3] = self.pose[:3] + a.T * self.max_delta
        target[3:] = self.home_quat[:, None]

        err, _ = ik.solve(target, self.q, max_iters=self.ik_iters)  # q updated in place
        self.pose = ik.fk(self.q)

        dist = (self.pose[:3].T - self.goal).norm(dim=1)
        ik_fail = (err[0] > 1e-3).float()
        reward = -5.0 * dist - 0.01 * (a ** 2).sum(1) - 0.1 * ik_fail + 1.0 * (dist < 0.02).float()

        self.t += 1
        done = self.t >= self.max_steps
        info = {"dist": dist.mean().item(), "ik_fail": ik_fail.mean().item(),
                "success": (dist < 0.02).float().mean().item()}
        if done.any():
            self._reset(done)
        return self.obs(), reward, done.float(), info


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()

        def mlp(out):
            return nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, out))
        self.actor, self.critic = mlp(act_dim), mlp(1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def train(args):
    dev = "cuda"
    env = GpuReachEnv(args.envs, device=dev)
    ac = ActorCritic(env.obs_dim, env.act_dim).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    H, N = args.horizon, args.envs
    gamma, lam, clip = 0.99, 0.95, 0.2

    obs = env.obs()
    for it in range(args.iters):
        t0 = time.time()
        obs_b = torch.zeros(H, N, env.obs_dim, device=dev)
        act_b = torch.zeros(H, N, env.act_dim, device=dev)
        logp_b, val_b, rew_b, done_b = (torch.zeros(H, N, device=dev) for _ in range(4))

        for t in range(H):
            with torch.no_grad():
                d = ac.dist(obs)
                a = d.sample()
                logp_b[t], val_b[t] = d.log_prob(a).sum(-1), ac.value(obs)
            obs_b[t], act_b[t] = obs, a
            obs, rew_b[t], done_b[t], info = env.step(a)

        # GAE
        with torch.no_grad():
            last_v = ac.value(obs)
        adv = torch.zeros_like(rew_b)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(H)):
            next_v = last_v if t == H - 1 else val_b[t + 1]
            nonterm = 1.0 - done_b[t]
            delta = rew_b[t] + gamma * next_v * nonterm - val_b[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + val_b

        # PPO update
        B = H * N
        b_obs, b_act = obs_b.reshape(B, -1), act_b.reshape(B, -1)
        b_logp, b_adv, b_ret = logp_b.reshape(B), adv.reshape(B), ret.reshape(B)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        mb = B // args.minibatches
        for _ in range(args.epochs):
            perm = torch.randperm(B, device=dev)
            for i in range(0, B, mb):
                idx = perm[i:i + mb]
                d = ac.dist(b_obs[idx])
                ratio = (d.log_prob(b_act[idx]).sum(-1) - b_logp[idx]).exp()
                a_ = b_adv[idx]
                pg = -torch.min(ratio * a_, ratio.clamp(1 - clip, 1 + clip) * a_).mean()
                vl = 0.5 * ((ac.value(b_obs[idx]) - b_ret[idx]) ** 2).mean()
                loss = pg + 0.5 * vl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()

        if it % 5 == 0 or it == args.iters - 1:
            torch.cuda.synchronize()
            sps = B / (time.time() - t0)
            print(f"it {it:4d} | rew/step {rew_b.mean().item():7.3f} | dist {info['dist']:.3f} m | "
                  f"success {info['success']*100:5.1f}% | ik_fail {info['ik_fail']*100:4.1f}% | "
                  f"{sps/1e3:7.1f}k env-steps/s")

    torch.save(ac.state_dict(), os.path.join(HERE, "reach_policy.pt"))
    print("saved reach_policy.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    train(p.parse_args())