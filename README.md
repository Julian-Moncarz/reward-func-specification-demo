# Reward Lab

A tiny, in-browser demo of **reward misspecification** and **goal misgeneralization** —
the gap between the reward you *write* and the behavior you *get*.

Describe a reward in plain English. An LLM (via your own OpenRouter key) turns it into a
JavaScript reward function. An agent is then trained from scratch on that reward with tabular
Q-learning — **entirely in your browser**, in a few seconds — and you watch what the reward you
wrote *actually* makes the agent do.

## Scenarios

- **🤖 Cleanup Robot** — clean the dirt, don't break the potted plants, then dock. Reward only the
  dirt and the robot learns to *smash the plants*: each one spills soil it can re-"clean" for more
  reward. The robot's camera only sees nearby dirt, so an under-specified reward goes badly.
  *(outer misalignment / reward hacking)*
- **🪙 Coin Quest** — collect the coin. During training the coin always sits right next to the exit,
  so the agent can learn "head to the exit" and still look perfect. Flip on the **test world** and
  the coin spawns somewhere random: the agent beelines for the exit and ignores the coin. It never
  learned the goal you meant. *(inner misalignment / goal misgeneralization)*

## Running it

It's a fully static site — no server, no build step.

```sh
# any static server works; or just open index.html in a browser
python3 -m http.server 8000
# then visit http://localhost:8000
```

You'll need an [OpenRouter API key](https://openrouter.ai/keys) for the natural-language → reward
step. Paste it into the box at the top. The key is used only in your browser and is sent only to
`openrouter.ai` — it never touches a server of ours (there isn't one). By default it lives only in
the current tab; tick **remember on this device** to keep it in `localStorage`.

The default model is `anthropic/claude-haiku-4.5`; override it in the model box if you like.

## How it works

| File | Role |
|------|------|
| `index.html` | UI, canvas rendering, and the train→play wiring |
| `envs.js` | The two gridworlds (state, dynamics, observations) |
| `trainer.js` | Tabular Q-learning + greedy rollout recording |
| `rewardgen.js` | Browser → OpenRouter call that writes the reward function |

Training is ε-greedy Q-learning over a lookup table `Q(state, action)`, updated with
`Q ← Q + α·(r + γ·max Q(next) − Q)` after every step. The animation is the final greedy policy.
Nothing else is optimized — whatever you see, the reward you wrote caused it.

`envs.js` and `trainer.js` also export under CommonJS, so the engine can be exercised headlessly
in Node for testing.

## Deploying

Any static host works. This repo is set up for **GitHub Pages** served from the repository root,
so pushing to `main` publishes it. The `.nojekyll` file tells Pages to serve the files as-is.
Because each visitor brings their own OpenRouter key, hosting it costs you nothing in API usage.
