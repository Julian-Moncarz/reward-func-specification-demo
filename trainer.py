"""Tabular Q-learning against a user-supplied reward function, plus greedy
rollout recording for the frontend animation."""

import math
import random

from envs import ENVS, TRAIN_PARAMS


class RewardError(Exception):
    pass


def compile_reward(code):
    """Compile user/Claude-written reward code into a callable.

    The code must define `def reward(prev, action, state): ...` where prev and
    state are the env's state dicts and action is 0-3.
    """
    ns = {"math": math, "abs": abs, "min": min, "max": max, "__builtins__": {
        "abs": abs, "min": min, "max": max, "round": round, "int": int,
        "float": float, "bool": bool, "len": len, "sum": sum, "True": True,
        "False": False, "None": None,
    }}
    try:
        exec(code, ns)
    except Exception as e:
        raise RewardError(f"reward code failed to compile: {e}")
    fn = ns.get("reward")
    if not callable(fn):
        raise RewardError("code must define a function `reward(prev, action, state)`")
    return fn


def validate_reward(env_id, fn):
    env = ENVS[env_id]()
    env.reset()
    prev = env.state_dict()
    env.step(3)
    try:
        r = fn(prev, 3, env.state_dict())
        float(r)
    except Exception as e:
        raise RewardError(f"reward function crashed on a sample step: {e}")


def train(env_id, reward_fn, progress=None, seed=0):
    """Q-learning. Returns the Q-table. Calls progress(ep, total) periodically."""
    p = TRAIN_PARAMS[env_id]
    episodes, alpha, gamma = p["episodes"], p["alpha"], p["gamma"]
    rng = random.Random(seed)
    env = ENVS[env_id]()
    Q = {}

    def q(s):
        if s not in Q:
            Q[s] = [0.0, 0.0, 0.0, 0.0]
        return Q[s]

    decay_until = int(episodes * 0.7)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / decay_until)
        a_lr = max(0.05, alpha * (1.0 - 0.8 * ep / episodes))
        env.reset()
        s = env.obs()
        prev_dict = env.state_dict()
        while not env.done:
            qs = q(s)
            if rng.random() < eps:
                a = rng.randrange(4)
            else:
                m = max(qs)
                a = qs.index(m)
            env.step(a)
            cur_dict = env.state_dict()
            try:
                r = float(reward_fn(prev_dict, a, cur_dict))
            except Exception as e:
                raise RewardError(f"reward function crashed mid-training: {e}")
            ns = env.obs()
            # bootstrap through time-limit truncation; only true terminals stop
            target = r if env.terminal else r + gamma * max(q(ns))
            qs[a] += a_lr * (target - qs[a])
            s, prev_dict = ns, cur_dict
        if progress and ep % 500 == 0:
            progress(ep, episodes)
    if progress:
        progress(episodes, episodes)
    return Q


def rollout(env_id, Q, reward_fn, shift=False):
    """Greedy rollout; returns a list of frames for the frontend."""
    env = ENVS[env_id]()
    env.reset(shift=shift)
    prev_dict = env.state_dict()
    total = 0.0
    frames = [{"step": 0, **env.frame(), "reward": 0.0, "total": 0.0, "events": [],
               "state": prev_dict}]
    while not env.done:
        s = env.obs()
        qs = Q.get(s)
        if qs is None or max(qs) == min(qs) == 0.0:
            # unseen / indifferent state: act greedily anyway (ties -> action 0
            # would look frozen; nudge with a fixed preference order R,D,U,L)
            qs = qs or [0.0] * 4
            order = [3, 1, 0, 2]
            a = max(order, key=lambda i: (qs[i], -order.index(i)))
        else:
            a = qs.index(max(qs))
        events = env.step(a) or []
        cur_dict = env.state_dict()
        try:
            r = float(reward_fn(prev_dict, a, cur_dict))
        except Exception:
            r = 0.0
        total += r
        frames.append({"step": env.steps, **env.frame(), "reward": round(r, 2),
                       "total": round(total, 2), "events": events,
                       "state": cur_dict})
        prev_dict = cur_dict
    return frames
