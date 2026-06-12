"""Turn a natural-language reward description into Python reward code by
shelling out to the Claude Code CLI (`claude -p`), so no API key is needed."""

import json
import os
import re
import subprocess

from envs import ENVS

MODEL = os.environ.get("REWARD_LAB_MODEL", "claude-haiku-4-5-20251001")

PROMPT_TEMPLATE = """You translate a natural-language reward description into a Python reward \
function for a tabular-RL gridworld. Translate LITERALLY: implement exactly what is asked, \
nothing more. Do NOT add penalties, bonuses, shaping, or safeguards the user did not ask for, \
even if you think the description is incomplete or would lead to unintended behavior. \
(This is for an AI-safety demo about misspecified rewards — faithfulness to the spec is the \
whole point.)

The environment is "{env_name}". {tagline}

The function signature is:

    def reward(prev, action, state):

`prev` and `state` are dicts (the state before and after the action). `action` is an int 0-3 \
(up/down/left/right). Available keys in both dicts:

{state_docs}

Counters like the cumulative ones only ever increase; to reward an event once, use deltas, \
e.g. `state['dirt_cleaned'] - prev['dirt_cleaned']`.

The user's reward description:
\"\"\"{spec}\"\"\"

STRICT REQUIREMENT: every reward trigger in the description must come with an explicit number \
("1 point per dirt cleaned", "-5 per vase"). If any trigger has no number ("some points", "a big \
penalty", "reward it for X" with no amount), do NOT write code. Instead reply with ONLY:
{{"error": "<one short sentence saying which trigger is missing a number and showing how to phrase it>"}}

Otherwise reply with ONLY a JSON object (no markdown fences, no prose before or after) of the form:
{{"code": "<the python source of the reward function>", "summary": [{{"icon": "<single emoji>", "text": "<short human-readable line, e.g. '+10 per coin collected'>"}}]}}

The summary must list every term of the reward function, one entry per term, using the exact \
numeric values from the code. Keep the code simple and readable; only use keys listed above."""


class GenError(Exception):
    pass


def build_prompt(env_id, spec):
    env = ENVS[env_id]
    docs = "\n".join(f"  - {k}: {v}" for k, v in env.STATE_DOCS.items())
    return PROMPT_TEMPLATE.format(env_name=env.name, tagline=env.tagline,
                                  state_docs=docs, spec=spec)


def extract_json(text):
    """Pull the first balanced {...} block out of possibly-wrapped output."""
    start = text.find("{")
    if start == -1:
        raise GenError("no JSON in model output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
        elif ch == '"' and not esc:
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise GenError("unbalanced JSON in model output")


def generate_reward(env_id, spec, timeout=90):
    prompt = build_prompt(env_id, spec)
    try:
        out = subprocess.run(
            ["claude", "-p", prompt, "--model", MODEL, "--output-format", "json"],
            capture_output=True, text=True, timeout=timeout, cwd="/tmp",
        )
    except subprocess.TimeoutExpired:
        raise GenError("Claude took too long to respond")
    except FileNotFoundError:
        raise GenError("`claude` CLI not found on PATH")
    if out.returncode != 0:
        raise GenError(f"claude CLI failed: {out.stderr[:300]}")
    try:
        wrapper = json.loads(out.stdout)
        result_text = wrapper.get("result", "")
    except json.JSONDecodeError:
        result_text = out.stdout
    data = extract_json(result_text)
    if data.get("error"):
        raise GenError(data["error"])
    code = data.get("code", "")
    summary = data.get("summary", [])
    if "def reward" not in code:
        raise GenError("model did not return a reward function")
    # normalize escaped newlines if the model double-escaped
    if "\\n" in code and "\n" not in code:
        code = code.replace("\\n", "\n")
    return {"code": code, "summary": summary}
