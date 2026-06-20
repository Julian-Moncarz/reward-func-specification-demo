/* Three small gridworlds ported from envs.py. Designed so that misspecified
 * rewards produce visibly misaligned policies, and (for Coin Quest) a correct
 * reward still misgeneralizes under a test-time shift.
 *
 * All envs share:
 *   - 8x8 grid, actions 0=up 1=down 2=left 3=right (moving into a wall = stay)
 *   - reset(shift) / step(action)
 *   - stateDict()  -> object handed to the user's reward function
 *   - obsKey()     -> string key for the tabular Q-table
 *   - frame()      -> JSON-serializable render state for the frontend
 *
 * Loads as a classic script (attaches REWARDLAB_ENVS to window) and also
 * exports for Node so the logic can be unit-tested headlessly.
 */
(function (root) {
  const MOVES = { 0: [0, -1], 1: [0, 1], 2: [-1, 0], 3: [1, 0] };
  const W = 8, H = 8;
  const key = (x, y) => x + "," + y;
  const manhattan = (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);

  class CleanupEnv {
    static id = "cleanup";
    static name = "Cleanup Robot";
    static icon = "🤖";
    static tagline =
      "A robot vacuum: dirt to clean, fragile potted plants, a charging dock.";
    static max_steps = 80;
    static intended = "Clean all the dirt without breaking anything, then go dock.";

    static DIRT = [[2, 1], [5, 2], [1, 4], [4, 5]];
    static PLANTS = [[3, 3], [6, 1]];
    // each knocked-over plant spills its soil onto these fixed cells
    static SPILLS = { 0: [[2, 3], [4, 3]], 1: [[6, 2], [7, 1]] };
    static DOCK = [7, 7];
    static START = [0, 0];
    static FOV = 2; // camera range (manhattan)

    static STATE_DOCS = {
      x: "agent column, 0-7",
      y: "agent row, 0-7",
      steps: "steps taken so far this episode",
      dirt_cleaned: "total dirt cells cleaned so far (cumulative)",
      dirt_remaining: "dirt cells currently on the floor",
      plants_broken: "potted plants knocked over so far (each spills soil: MORE dirt)",
      plants_intact: "potted plants still standing",
      docked: "true the moment the robot reaches the charging dock (ends episode)",
      dist_to_dock: "manhattan distance to the dock",
      dirt_in_view: "dirt cells the robot's camera currently sees (within 2 cells of the robot)",
      done: "true if the episode just ended",
    };

    constructor() { this.reset(); }

    reset() {
      this.pos = CleanupEnv.START.slice();
      this.steps = 0;
      this.dirt = new Set(CleanupEnv.DIRT.map((c) => key(c[0], c[1])));
      this.plants = new Set([0, 1]);
      this.dirt_cleaned = 0;
      this.plants_broken = 0;
      this.docked = false;
      this.done = false;
      this.terminal = false;
      return this;
    }

    step(action) {
      if (this.done) return [];
      const [dx, dy] = MOVES[action];
      const nx = this.pos[0] + dx, ny = this.pos[1] + dy;
      if (nx >= 0 && nx < W && ny >= 0 && ny < H) this.pos = [nx, ny];
      this.steps += 1;
      const events = [];
      const here = key(this.pos[0], this.pos[1]);
      for (const vi of Array.from(this.plants)) {
        if (this.pos[0] === CleanupEnv.PLANTS[vi][0] && this.pos[1] === CleanupEnv.PLANTS[vi][1]) {
          this.plants.delete(vi);
          this.plants_broken += 1;
          for (const cell of CleanupEnv.SPILLS[vi]) this.dirt.add(key(cell[0], cell[1]));
          events.push("plant_broken");
        }
      }
      if (this.dirt.has(here)) {
        this.dirt.delete(here);
        this.dirt_cleaned += 1;
        events.push("dirt_cleaned");
      }
      if (this.pos[0] === CleanupEnv.DOCK[0] && this.pos[1] === CleanupEnv.DOCK[1]) {
        this.docked = true; this.done = true; this.terminal = true;
        events.push("docked");
      }
      if (this.steps >= CleanupEnv.max_steps) this.done = true;
      return events;
    }

    stateDict() {
      let inView = 0;
      for (const c of this.dirt) {
        const [cx, cy] = c.split(",").map(Number);
        if (manhattan(this.pos, [cx, cy]) <= 2) inView += 1;
      }
      return {
        x: this.pos[0], y: this.pos[1], steps: this.steps,
        dirt_cleaned: this.dirt_cleaned,
        dirt_remaining: this.dirt.size,
        plants_broken: this.plants_broken,
        plants_intact: this.plants.size,
        docked: this.docked,
        dist_to_dock: manhattan(this.pos, CleanupEnv.DOCK),
        dirt_in_view: inView,
        done: this.done,
      };
    }

    obsKey() {
      // The policy sees: where it is, dirt within camera range, the (tall,
      // visible-from-anywhere) plants, and its cleaned-count as memory. It has
      // NO access to the location of dirt outside the camera FOV.
      const dirtCells = CleanupEnv.DIRT.concat(CleanupEnv.SPILLS[0], CleanupEnv.SPILLS[1]);
      let dmask = 0;
      for (let i = 0; i < dirtCells.length; i++) {
        const c = dirtCells[i];
        if (this.dirt.has(key(c[0], c[1])) && manhattan(this.pos, c) <= CleanupEnv.FOV)
          dmask |= 1 << i;
      }
      let pmask = 0;
      for (const v of this.plants) pmask |= 1 << v;
      return this.pos[0] + "," + this.pos[1] + "|" + dmask + "|" + pmask + "|" + this.dirt_cleaned;
    }

    frame() {
      const dirt = Array.from(this.dirt).map((c) => c.split(",").map(Number)).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const plants = Array.from(this.plants).sort().map((v) => CleanupEnv.PLANTS[v].slice());
      const broken = [0, 1].filter((v) => !this.plants.has(v)).map((v) => CleanupEnv.PLANTS[v].slice());
      return { agent: this.pos.slice(), dirt, plants, broken, dock: CleanupEnv.DOCK.slice() };
    }

    static layout() {
      return {
        dock: CleanupEnv.DOCK.slice(), start: CleanupEnv.START.slice(),
        dirt: CleanupEnv.DIRT.map((c) => c.slice()),
        plants: CleanupEnv.PLANTS.map((c) => c.slice()),
      };
    }
  }

  class CoinGridEnv {
    static id = "coins";
    static name = "Coin Quest";
    static icon = "🪙";
    static tagline =
      "One coin, one exit. Train it... then drop the coin somewhere random and see what it REALLY learned.";
    static max_steps = 40;
    static intended = "Grab the coin. (During training the coin always sits right beside the exit...)";

    static START = [0, 3];
    static EXIT = [7, 3];
    static COIN_TRAIN = [6, 3]; // during training: always right beside the exit

    static STATE_DOCS = {
      x: "agent column, 0-7",
      y: "agent row, 0-7",
      steps: "steps taken so far this episode",
      coin_collected: "true once the coin has been picked up",
      at_exit: "true when standing on the exit (ends episode)",
      dist_to_coin: "manhattan distance to the coin",
      dist_to_exit: "manhattan distance to the exit",
      done: "true if the episode just ended",
    };

    constructor() { this.reset(); }

    reset(shift = false, rng = Math.random) {
      this.pos = CoinGridEnv.START.slice();
      if (shift) {
        const cells = [];
        for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) {
          const same = (a) => a[0] === x && a[1] === y;
          if (!same(CoinGridEnv.START) && !same(CoinGridEnv.EXIT) && !same(CoinGridEnv.COIN_TRAIN))
            cells.push([x, y]);
        }
        this.coin = cells[Math.floor(rng() * cells.length)];
      } else {
        this.coin = CoinGridEnv.COIN_TRAIN.slice();
      }
      this.steps = 0;
      this.coin_collected = false;
      this.at_exit = false;
      this.done = false;
      this.terminal = false;
      return this;
    }

    step(action) {
      if (this.done) return [];
      const [dx, dy] = MOVES[action];
      const nx = this.pos[0] + dx, ny = this.pos[1] + dy;
      if (nx >= 0 && nx < W && ny >= 0 && ny < H) this.pos = [nx, ny];
      this.steps += 1;
      const events = [];
      if (!this.coin_collected && this.pos[0] === this.coin[0] && this.pos[1] === this.coin[1]) {
        this.coin_collected = true;
        events.push("coin");
      }
      if (this.pos[0] === CoinGridEnv.EXIT[0] && this.pos[1] === CoinGridEnv.EXIT[1]) {
        this.at_exit = true; this.done = true; this.terminal = true;
        events.push("exit");
      }
      if (this.steps >= CoinGridEnv.max_steps) this.done = true;
      return events;
    }

    stateDict() {
      return {
        x: this.pos[0], y: this.pos[1], steps: this.steps,
        coin_collected: this.coin_collected,
        at_exit: this.at_exit,
        dist_to_coin: this.coin_collected ? 0 : manhattan(this.pos, this.coin),
        dist_to_exit: manhattan(this.pos, CoinGridEnv.EXIT),
        done: this.done,
      };
    }

    obsKey() {
      // The coin's position IS in the observation — but during training the
      // coin never moves, so the policy gets no chance to learn what it means.
      // At test time, every relocated-coin state is one the Q-table never saw.
      const coin = this.coin_collected ? "_" : this.coin[0] + "," + this.coin[1];
      return this.pos[0] + "," + this.pos[1] + "|" + coin;
    }

    frame() {
      return {
        agent: this.pos.slice(),
        coin: this.coin_collected ? null : this.coin.slice(),
        exit: CoinGridEnv.EXIT.slice(),
      };
    }

    static layout() {
      return {
        start: CoinGridEnv.START.slice(), exit: CoinGridEnv.EXIT.slice(),
        coin: CoinGridEnv.COIN_TRAIN.slice(),
      };
    }
  }

  const CLASSES = { cleanup: CleanupEnv, coins: CoinGridEnv };

  const TRAIN_PARAMS = {
    cleanup: { episodes: 60000, alpha: 0.25, gamma: 0.97 },
    coins: { episodes: 6000, alpha: 0.25, gamma: 0.97 },
  };

  // metadata the UI/prompt need, mirroring the old /api/envs payload
  function envMeta() {
    return Object.values(CLASSES).map((E) => ({
      id: E.id, name: E.name, icon: E.icon, tagline: E.tagline,
      intended: E.intended, max_steps: E.max_steps,
      state_docs: E.STATE_DOCS, layout: E.layout(),
      has_shift: E.id === "coins",
    }));
  }

  const api = { CLASSES, TRAIN_PARAMS, envMeta, manhattan };
  root.REWARDLAB_ENVS = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
