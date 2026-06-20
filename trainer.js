/* Tabular Q-learning against a user-supplied reward function, plus greedy
 * rollout recording for the frontend animation. Ported from trainer.py and
 * run entirely in the browser (or in Node, for tests).
 */
(function (root) {
  const ENVS = (typeof require !== "undefined")
    ? require("./envs.js")
    : root.REWARDLAB_ENVS;
  const { CLASSES, TRAIN_PARAMS } = ENVS;

  class RewardError extends Error {}

  // Deterministic PRNG so a given spec trains the same way every time.
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function argmax(qs) {
    let bi = 0, bv = qs[0];
    for (let i = 1; i < qs.length; i++) if (qs[i] > bv) { bv = qs[i]; bi = i; }
    return bi;
  }

  /* Compile an LLM/user-written reward function into a callable. The code must
   * define `function reward(prev, action, state) { ... }` returning a number.
   * This runs in the user's own browser tab with their own key — there is no
   * server to attack — but it is still arbitrary JS, so only paste specs you
   * trust to be turned into reward code. */
  function compileReward(code) {
    let fn;
    try {
      fn = new Function(code + "\n;return typeof reward === 'function' ? reward : undefined;")();
    } catch (e) {
      throw new RewardError("reward code failed to compile: " + e.message);
    }
    if (typeof fn !== "function")
      throw new RewardError("code must define a function `reward(prev, action, state)`");
    return fn;
  }

  function validateReward(envId, fn) {
    const env = new CLASSES[envId]();
    env.reset();
    const prev = env.stateDict();
    env.step(3);
    let r;
    try { r = Number(fn(prev, 3, env.stateDict())); }
    catch (e) { throw new RewardError("reward function crashed on a sample step: " + e.message); }
    if (!Number.isFinite(r))
      throw new RewardError("reward function must return a finite number, got " + r);
  }

  // Yield a macrotask so the page can repaint. MessageChannel is NOT throttled
  // in backgrounded/occluded tabs the way setTimeout(0) is (clamped to ~1s),
  // so training stays fast even when the tab isn't focused. Falls back to
  // setTimeout where MessageChannel is unavailable (e.g. older Node).
  const sleep = (typeof MessageChannel !== "undefined")
    ? () => new Promise((res) => {
        const ch = new MessageChannel();
        ch.port1.onmessage = () => { ch.port1.close(); res(); };
        ch.port2.postMessage(0);
      })
    : () => new Promise((res) => setTimeout(res, 0));

  /* Q-learning. Returns the Q-table (Map<obsKey, [4]>). Awaits a yield to the
   * event loop every `yieldEvery` episodes so the page stays responsive and
   * `progress(ep, total)` can repaint. */
  async function train(envId, rewardFn, opts = {}) {
    const { progress = null, seed = 0, yieldEvery = 500 } = opts;
    const p = TRAIN_PARAMS[envId];
    const { episodes, alpha, gamma } = p;
    const rng = mulberry32(seed);
    const env = new CLASSES[envId]();
    const Q = new Map();
    const q = (s) => { let v = Q.get(s); if (!v) { v = [0, 0, 0, 0]; Q.set(s, v); } return v; };

    const decayUntil = Math.floor(episodes * 0.7);
    for (let ep = 0; ep < episodes; ep++) {
      const eps = Math.max(0.05, 1.0 - ep / decayUntil);
      const aLr = Math.max(0.05, alpha * (1.0 - 0.8 * ep / episodes));
      env.reset();
      let s = env.obsKey();
      let prevDict = env.stateDict();
      while (!env.done) {
        const qs = q(s);
        const a = rng() < eps ? Math.floor(rng() * 4) : argmax(qs);
        env.step(a);
        const curDict = env.stateDict();
        let r;
        try { r = Number(rewardFn(prevDict, a, curDict)); }
        catch (e) { throw new RewardError("reward function crashed mid-training: " + e.message); }
        if (!Number.isFinite(r)) r = 0;
        const ns = env.obsKey();
        // bootstrap through time-limit truncation; only true terminals stop
        const target = env.terminal ? r : r + gamma * Math.max.apply(null, q(ns));
        qs[a] += aLr * (target - qs[a]);
        s = ns; prevDict = curDict;
      }
      if (progress && ep % 500 === 0) progress(ep, episodes);
      if (ep % yieldEvery === 0) await sleep();
    }
    if (progress) progress(episodes, episodes);
    return Q;
  }

  const round2 = (x) => Math.round(x * 100) / 100;

  /* Greedy rollout; returns a list of frames for the frontend. */
  function rollout(envId, Q, rewardFn, opts = {}) {
    const { shift = false, seed = 0 } = opts;
    const rng = mulberry32(seed + (shift ? 1 : 0));
    const env = new CLASSES[envId]();
    env.reset(shift, rng);
    let prevDict = env.stateDict();
    let total = 0;
    const frames = [Object.assign(
      { step: 0, reward: 0, total: 0, events: [], state: prevDict }, env.frame())];
    const order = [3, 1, 0, 2]; // preference when a state is unseen/indifferent
    while (!env.done) {
      const s = env.obsKey();
      let qs = Q.get(s);
      let a;
      if (!qs || (Math.max.apply(null, qs) === 0 && Math.min.apply(null, qs) === 0)) {
        const q4 = qs || [0, 0, 0, 0];
        a = order.reduce((best, i) =>
          (q4[i] > q4[best] || (q4[i] === q4[best] && order.indexOf(i) < order.indexOf(best))) ? i : best, order[0]);
      } else {
        a = argmax(qs);
      }
      const events = env.step(a) || [];
      const curDict = env.stateDict();
      let r;
      try { r = Number(rewardFn(prevDict, a, curDict)); if (!Number.isFinite(r)) r = 0; }
      catch (e) { r = 0; }
      total += r;
      frames.push(Object.assign(
        { step: env.steps, reward: round2(r), total: round2(total), events, state: curDict },
        env.frame()));
      prevDict = curDict;
    }
    return frames;
  }

  const api = { RewardError, compileReward, validateReward, train, rollout, mulberry32 };
  root.REWARDLAB_TRAINER = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
