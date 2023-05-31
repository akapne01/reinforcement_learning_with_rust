# Reinforcement Learning (RL) in Rust

Currently hold implementations of following RL problems:

- k-armed bandit or multi armed bandit problem. Where we are faced with k slot machines that each provide reward that is either 0 or 1. If reward is 0 then it signifies loss, if reward is 1, it signifies win. Each slot machine or bandit has a specific probability set that is not known in advance. It is agents task to learn which machines have what proabability of winning and explit that knowledge using epsilon-greedy action selection policy.

# Notes

- Please note that currently the parameters epsilon, alpha, number of bandits, number of games, number of tries in a game, etc. are defined as constants. You can update them in src/constants.rs

# Referneces:

- Python version of stohastic implementation of k-armed bandit problem: https://www.dominodatalab.com/blog/k-armed-bandit-problem
