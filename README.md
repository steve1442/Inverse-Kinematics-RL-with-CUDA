GPU Inverse Kinematics for Reinforcement Learning
Overview

This project solves inverse kinematics (IK) for thousands of robot arms in parallel on the GPU, and uses that to train a simple reinforcement learning (RL) agent.

The robot is a 7-joint arm (Franka Panda-style).
Each environment is solved by one GPU thread using damped least squares (Levenberg-Marquardt).
An RL agent (PPO) learns to move the arm's end effector to random goal positions. It picks a small (x, y, z) move each step, and the CUDA solver converts that into joint angles.

Built for an RTX 4070 Ti (compute capability 8.9, sm_89).

Files
File	What it does
ik_batch.cu	The CUDA IK solver, plus a standalone benchmark program
ik_torch_ext.cu	Wraps the solver so PyTorch can call it on GPU tensors
ik_rl.py	Reaching environment and PPO training script

Requirements
NVIDIA GPU and driver
CUDA Toolkit (nvcc), version matching your PyTorch build
Python 3 with PyTorch, numpy, ninja
Option 1: Run the standalone IK benchmark

Compile and run the solver on its own to check it works and see how fast it is:

nvcc -O3 -arch=sm_89 -std=c++17 ik_batch.cu -o ik_batch
./ik_batch 65536 0.5

The two arguments are optional:

65536 is the number of environments solved in parallel.
0.5 is the goal spread in radians. It controls how far each test target is from the starting pose. Small values are easy (like an RL step), large values (2-3) are hard.

The output shows solves per second, the percentage of problems that converged, and the mean iteration count.

Option 2: Train the RL agent
python ik_rl.py --envs 4096 --iters 300

The first run compiles the CUDA extension, which takes about a minute. After that it starts training and prints progress every few iterations:

reward per step
average distance to the goal (meters)
success rate (within 2 cm of the goal)
IK failure rate
environment steps per second

The trained policy is saved to reach_policy.pt.

Optional arguments: --envs, --iters, --horizon, --epochs, --minibatches, --lr.

Using the solver
