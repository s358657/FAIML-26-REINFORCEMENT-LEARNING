# Robust Reinforcement Learning for Sim-to-Sim Transfer in Robotic Pushing Tasks

This repository contains the final project implementation for the **Fundamentals of Artificial Intelligence, Machine and Deep Learning** course. 
The project addresses the **Sim-to-Sim (and Sim-to-Real) transfer gap** in robotic manipulation by training a robotic arm (`PandaGym PushTask`) to reliably and efficiently push a cube under extreme variations of environmental physical conditions.

---

## 📂 Repository Structure

The project is divided into two main sections:

* **`part1/`**: Fundamental study of Policy Gradient algorithms implemented from scratch and evaluated on the `Hopper-v4` locomotion environment.
* **`part2/`**: Advanced study of Off-Policy algorithms and environmental robustness techniques on `PandaGym`.
    * `train_sb3.py`: Main script for training and evaluation using Stable-Baselines3 (supports PPO/SAC strategies and TensorBoard logging).
    * `rand_wrapper.py`: Custom gym environment wrapper for injecting Domain Randomization (UDR and ADR).

---

## 🤖 Algorithms & Methodology

### Part 1: Hopper-v4 (Custom Implementation)
A comparative analysis evaluating the impact of variance reduction techniques on policy gradients:
1.  **Vanilla REINFORCE**: Pure Monte Carlo algorithm (highly unstable due to high gradient variance).
2.  **REINFORCE with Constant Baseline**: Introduction of a baseline factor ($b=20$) to stabilize updates (yields the best performance in the Hopper environment).
3.  **Actor-Critic**: Bootstrapped value function (critic) implementation to estimate the advantage function.

### Part 2: PandaGym PushTask (SB3 & Domain Randomization)
Transferring learned policies from a *Source Domain* to an unseen *Target Domain*:
* **PPO vs SAC Comparison**: Evaluation of sample efficiency.
* **Architectural Optimization**: Analysis of the trade-off between memorization and generalization by comparing a standard network layout (Model 1) with an over-parameterized configuration (Model 2).
* **Uniform Domain Randomization (UDR)**: Randomization of the cube's **mass** sampled uniformly during training while keeping friction static.
* **Automatic Domain Randomization (ADR)**: An advanced multi-variable dynamic curriculum that expands environment boundaries by simultaneously scaling both **mass** and **surface friction** .
