/* Turn a natural-language reward description into a JavaScript reward function
 * by calling OpenRouter directly from the browser with the user's own API key.
 * The key never leaves the browser except to go to openrouter.ai. Ported from
 * reward_gen.py. */
(function (root) {
  const ENVS = root.REWARDLAB_ENVS;
  const DEFAULT_MODEL = "anthropic/claude-haiku-4.5";

  const PROMPT_TEMPLATE = `You translate a natural-language reward description into a JavaScript reward \
function for a tabular-RL gridworld. Translate LITERALLY: implement exactly what is asked, \
nothing more. Do NOT add penalties, bonuses, shaping, or safeguards the user did not ask for, \
even if you think the description is incomplete or would lead to unintended behavior. \
(This is for an AI-safety demo about misspecified rewards — faithfulness to the spec is the \
whole point.)

The environment is "{env_name}". {tagline}

The function signature is:

    function reward(prev, action, state) { ... }

\`prev\` and \`state\` are plain objects (the state before and after the action). \`action\` is an \
integer 0-3 (up/down/left/right). The function must return a number. \`Math\` is available. \
Available keys in both \`prev\` and \`state\`:

{state_docs}

Counters like the cumulative ones only ever increase; to reward an event once, use deltas, \
e.g. \`state.dirt_cleaned - prev.dirt_cleaned\`. Booleans are true/false.

STRICT REQUIREMENT: every reward trigger in the description must come with an explicit number \
("1 point per dirt cleaned", "-5 per plant"). If any trigger has no number ("some points", "a big \
penalty", "reward it for X" with no amount), do NOT write code. Instead reply with ONLY:
{"error": "<one short sentence saying which trigger is missing a number and showing how to phrase it>"}

Otherwise reply with ONLY a JSON object (no markdown fences, no prose before or after) of the form:
{"code": "<the javascript source of the reward function, starting with 'function reward'>", "summary": [{"icon": "<single emoji>", "text": "<short human-readable line, e.g. '+10 per coin collected'>"}]}

The summary must list every term of the reward function, one entry per term, using the exact \
numeric values from the code. Keep the code simple and readable; only use keys listed above.`;

  function buildPrompt(envId, spec) {
    const meta = ENVS.envMeta().find((e) => e.id === envId);
    const docs = Object.entries(meta.state_docs).map(([k, v]) => `  - ${k}: ${v}`).join("\n");
    return PROMPT_TEMPLATE
      .replace("{env_name}", meta.name)
      .replace("{tagline}", meta.tagline)
      .replace("{state_docs}", docs);
  }

  // Pull the first balanced {...} block out of possibly-wrapped output.
  function extractJSON(text) {
    const start = text.indexOf("{");
    if (start === -1) throw new Error("no JSON in model output");
    let depth = 0, inStr = false, esc = false;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (esc) { esc = false; continue; }
      if (ch === "\\") esc = true;
      else if (ch === '"') inStr = !inStr;
      else if (!inStr) {
        if (ch === "{") depth++;
        else if (ch === "}") { depth--; if (depth === 0) return JSON.parse(text.slice(start, i + 1)); }
      }
    }
    throw new Error("unbalanced JSON in model output");
  }

  async function generateReward(envId, spec, apiKey, model) {
    if (!apiKey) throw new Error("enter your OpenRouter API key first");
    const prompt = buildPrompt(envId, spec) + "\n\nThe user's reward description:\n\"\"\"" + spec + "\"\"\"";
    let resp;
    try {
      resp = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": "Bearer " + apiKey,
          "Content-Type": "application/json",
          "HTTP-Referer": root.location ? root.location.origin : "https://reward-lab",
          "X-Title": "Reward Lab",
        },
        body: JSON.stringify({
          model: model || DEFAULT_MODEL,
          max_tokens: 1024,
          messages: [{ role: "user", content: prompt }],
        }),
      });
    } catch (e) {
      throw new Error("OpenRouter request failed: " + e.message);
    }
    if (!resp.ok) {
      const body = await resp.text().catch(() => "");
      throw new Error("OpenRouter error " + resp.status + ": " + body.slice(0, 300));
    }
    const out = await resp.json();
    let content;
    try { content = out.choices[0].message.content; }
    catch (e) { throw new Error("unexpected OpenRouter response: " + JSON.stringify(out).slice(0, 300)); }

    const data = extractJSON(content);
    if (data.error) throw new Error(data.error);
    let code = data.code || "";
    const summary = data.summary || [];
    if (!/function\s+reward/.test(code) && !/reward\s*=/.test(code))
      throw new Error("model did not return a reward function");
    if (code.includes("\\n") && !code.includes("\n")) code = code.replace(/\\n/g, "\n");
    return { code, summary };
  }

  root.REWARDLAB_GEN = { generateReward, DEFAULT_MODEL };
})(typeof window !== "undefined" ? window : globalThis);
