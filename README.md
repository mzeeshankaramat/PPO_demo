# Proximal Policy Optimization (PPO)

This repository contains an implementation of **Proximal Policy Optimization (PPO)** using PyTorch and Gymnasium (OpenAI Gym fork). The agent is built for environments with **discrete action spaces**, like `CartPole-v1`, `MountainCar-v0`, or `LunarLander-v2`.

---

## 🧠 Overview

**PPO** is a powerful policy gradient method that improves training stability by:
- Using **clipped policy updates**
- Leveraging a **value function baseline**
- Estimating advantages using **Generalized Advantage Estimation (GAE)**

---

## 📦 Features

- ✅ Discrete action space support (using `Categorical` distribution)
- ✅ GAE-based advantage estimation
- ✅ Clipped surrogate loss for stable updates
- ✅ Separate policy and value networks
- ✅ Batch updates with multiple training epochs
- ✅ Reward and state visualization

---

## 🛠️ Installation

Make sure you have Python 3.7+ and the following dependencies:

```bash
pip install torch gymnasium matplotlib numpy
