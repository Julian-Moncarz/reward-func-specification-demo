# Reward Lab

live site: https://julian-moncarz.github.io/reward-func-specification-demo/

A tiny, in-browser demo of reward misspecification/outer misalignment and goal misgeneralization/inner Misalignment.
Each one has a toy environment that you write reward functions for which demonstrates it.

1. You describe a reward function in English.
2. An LLM (via your own OpenRouter key) turns it into code.
3. An agent is then trained on that reward with tabular
Q-learning in a few seconds
4. You watch what the reward you wrote makes the agent do.
