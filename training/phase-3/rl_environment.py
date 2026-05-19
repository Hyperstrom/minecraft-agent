"""
rl_environment.py
=================
Reinforcement Learning environment that wraps a Mineflayer bot.
This runs LOCALLY on your machine alongside a Minecraft server.

Architecture:
  Python RL Trainer  <-->  HTTP API  <-->  Node.js Mineflayer Bot  <-->  Minecraft Server

The Mineflayer bot runs a tiny Express server on port 3001.
This Python class talks to it to get state and send actions.

Install requirements:
    pip install gymnasium requests numpy torch transformers

Start the bot server first:
    node bot/rl_server.js

Then run RL training:
    python rl_training/train_rl.py
"""

import json
import time
import math
import requests
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional

# ── Constants ──────────────────────────────────────────────────────────────────
BOT_API_URL  = "http://localhost:3001"
ACTIONS      = ["MOVE_FORWARD", "MOVE_BACK", "MOVE_LEFT", "MOVE_RIGHT",
                 "JUMP", "ATTACK", "MINE", "CRAFT", "EAT", "PLACE", "WAIT"]
ACTION_IDX   = {a: i for i, a in enumerate(ACTIONS)}
MAX_NEARBY   = 8     # max blocks/mobs tracked in state
INV_SLOTS    = 36    # standard Minecraft inventory size

# Reward shaping — these make the RL signal dense so the agent learns faster
REWARDS = {
    # Positive rewards
    "collect_wood":       +2.0,
    "collect_stone":      +1.5,
    "collect_iron":       +3.0,
    "collect_diamond":    +10.0,
    "craft_item":         +5.0,
    "smelt_item":         +4.0,
    "kill_hostile":       +8.0,
    "eat_food":           +1.0,
    "survive_night":      +5.0,
    "reach_goal":         +20.0,
    "explore_new_chunk":  +0.5,
    # Negative rewards
    "take_damage":        -2.0,
    "die":                -20.0,
    "idle_too_long":      -0.5,
    "invalid_action":     -0.1,
    "fall_damage":        -3.0,
}


class MinecraftRLEnv(gym.Env):
    """
    OpenAI Gymnasium-compatible Minecraft environment.
    
    Observation space: flat vector of game state
    Action space:      discrete — 11 possible actions
    
    The LLM policy outputs JSON, which gets mapped to discrete actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, bot_url: str = BOT_API_URL, max_steps: int = 1000,
                 goal: str = "craft_iron_pickaxe", render_mode=None):
        super().__init__()
        self.bot_url   = bot_url
        self.max_steps = max_steps
        self.goal      = goal
        self.step_count = 0
        self.last_state = None
        self.visited_chunks = set()

        # Observation: [hp, food, pos_x, pos_y, pos_z, time_norm,
        #               inv_counts(36), nearby_block_ids(8), mob_dists(4)]
        obs_size = 6 + INV_SLOTS + MAX_NEARBY + 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

    # ── Core Gymnasium API ─────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.visited_chunks = set()
        # Tell the bot to respawn/reset
        try:
            requests.post(f"{self.bot_url}/reset", json={"goal": self.goal}, timeout=10)
            time.sleep(2.0)  # wait for server to load
        except requests.RequestException as e:
            print(f"⚠️  Bot reset failed: {e}")
        state = self._get_state()
        self.last_state = state
        return self._encode_state(state), {}

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        action_name = ACTIONS[action_idx]

        # Send action to bot
        reward = 0.0
        try:
            resp = requests.post(
                f"{self.bot_url}/action",
                json={"action": action_name, "goal": self.goal},
                timeout=5,
            )
            result = resp.json()
            reward += self._compute_reward(result, action_name)
        except requests.RequestException:
            reward += REWARDS["invalid_action"]
            result = {}

        # Get new state
        new_state = self._get_state()
        obs = self._encode_state(new_state)

        # Check termination
        terminated = new_state.get("hp", 20) <= 0  # bot died
        truncated  = self.step_count >= self.max_steps
        done       = terminated or truncated

        # Goal reward
        if result.get("goal_achieved"):
            reward += REWARDS["reach_goal"]
            terminated = True

        self.last_state = new_state
        return obs, reward, terminated, truncated, {"state": new_state, "action": action_name}

    def render(self):
        if self.last_state:
            s = self.last_state
            print(f"Step {self.step_count:>4} | HP:{s.get('hp',0):>2} "
                  f"Food:{s.get('food',0):>2} | Goal: {self.goal}")

    # ── State Encoding ─────────────────────────────────────────────────────────

    def _get_state(self) -> Dict:
        try:
            resp = requests.get(f"{self.bot_url}/state", timeout=5)
            return resp.json()
        except requests.RequestException:
            return {"hp": 20, "food": 20, "pos": {"x": 0, "y": 64, "z": 0},
                    "inv": {}, "nearby": [], "mobs": [], "time": 0}

    def _encode_state(self, state: Dict) -> np.ndarray:
        """Convert JSON game state → normalized float vector for RL."""
        KNOWN_ITEMS = [
            "oak_log", "wooden_planks", "stick", "cobblestone", "stone",
            "coal", "iron_ore", "iron_ingot", "gold_ore", "diamond",
            "wooden_pickaxe", "stone_pickaxe", "iron_pickaxe",
            "wooden_sword", "stone_sword", "iron_sword",
            "crafting_table", "furnace", "chest", "torch",
            "bread", "cooked_beef", "apple", "wheat",
            "bow", "arrow", "shield", "bucket",
            "dirt", "sand", "gravel", "glass",
            "wool", "leather", "string", "bone",
        ]
        KNOWN_BLOCKS = ["oak_log", "cobblestone", "stone", "iron_ore",
                        "gold_ore", "diamond_ore", "crafting_table",
                        "furnace", "water", "lava"]

        vec = []

        # [0-1] Vitals
        vec.append(state.get("hp",   20) / 20.0)
        vec.append(state.get("food", 20) / 20.0)

        # [2-4] Position (normalized to ±500 block range)
        pos = state.get("pos", {"x": 0, "y": 64, "z": 0})
        vec.append(np.clip(pos.get("x", 0) / 500.0, -1, 1))
        vec.append(np.clip((pos.get("y", 64) - 64) / 64.0, -1, 1))
        vec.append(np.clip(pos.get("z", 0) / 500.0, -1, 1))

        # [5] Time of day (0=dawn, 1=midnight)
        vec.append(state.get("time", 0) / 24000.0)

        # [6-41] Inventory counts (normalized to 0-1, cap at 64)
        inv = state.get("inv", {})
        for item in KNOWN_ITEMS:
            vec.append(min(inv.get(item, 0), 64) / 64.0)

        # [42-49] Nearby blocks (distance-encoded)
        nearby = state.get("nearby", [])
        block_vec = [0.0] * MAX_NEARBY
        for i, b in enumerate(nearby[:MAX_NEARBY]):
            if b.get("name") in KNOWN_BLOCKS:
                block_vec[i] = 1.0 - min(b.get("dist", 8.0), 8.0) / 8.0
        vec.extend(block_vec)

        # [50-53] Hostile mob distances (4 closest)
        mobs = [m for m in state.get("mobs", []) if m.get("hostile")]
        mob_vec = [0.0] * 4
        for i, m in enumerate(mobs[:4]):
            mob_vec[i] = 1.0 - min(m.get("dist", 16.0), 16.0) / 16.0
        vec.extend(mob_vec)

        arr = np.array(vec, dtype=np.float32)

        # Track chunk exploration for reward
        chunk = (int(pos.get("x", 0)) // 16, int(pos.get("z", 0)) // 16)
        self.visited_chunks.add(chunk)

        return arr

    # ── Reward Computation ─────────────────────────────────────────────────────

    def _compute_reward(self, result: Dict, action: str) -> float:
        reward = 0.0

        # Item collection rewards
        collected = result.get("collected", {})
        if "oak_log" in collected or "birch_log" in collected:
            reward += REWARDS["collect_wood"] * collected.get("oak_log", 0)
        if "cobblestone" in collected or "stone" in collected:
            reward += REWARDS["collect_stone"]
        if "iron_ore" in collected:
            reward += REWARDS["collect_iron"]
        if "diamond" in collected:
            reward += REWARDS["collect_diamond"]

        # Crafting / smelting
        if result.get("crafted"):
            reward += REWARDS["craft_item"]
        if result.get("smelted"):
            reward += REWARDS["smelt_item"]

        # Combat
        if result.get("killed_mob"):
            reward += REWARDS["kill_hostile"]

        # Health events
        damage = result.get("damage_taken", 0)
        if damage > 0:
            reward += REWARDS["take_damage"] * damage
        if result.get("died"):
            reward += REWARDS["die"]

        # Eating
        if result.get("ate"):
            reward += REWARDS["eat_food"]

        # Idling penalty
        if action == "WAIT":
            reward += REWARDS["idle_too_long"]

        # Invalid action
        if result.get("invalid"):
            reward += REWARDS["invalid_action"]

        return reward

    def state_to_json(self) -> str:
        """Returns current state as the JSON string the LLM expects."""
        if self.last_state:
            s = self.last_state.copy()
            s["goal"] = self.goal
            return json.dumps(s, separators=(',', ':'))
        return "{}"


# ── Utility: Convert LLM JSON output → discrete action index ──────────────────

def llm_action_to_index(llm_output: str) -> int:
    """
    Parse the LLM's JSON output and map it to a discrete action index.
    Falls back to WAIT on invalid JSON.
    """
    try:
        parsed = json.loads(llm_output.strip())
        action = parsed.get("action", "WAIT").upper()

        # Map LLM action names to env action names
        mapping = {
            "MOVE":     lambda p: f"MOVE_{p.get('direction','FORWARD').upper()}",
            "ATTACK":   lambda p: "ATTACK",
            "MINE":     lambda p: "MINE",
            "CRAFT":    lambda p: "CRAFT",
            "EAT":      lambda p: "EAT",
            "PLACE":    lambda p: "PLACE",
            "WAIT":     lambda p: "WAIT",
            "SHOOT":    lambda p: "ATTACK",
            "SMELT":    lambda p: "CRAFT",
            "DROP":     lambda p: "WAIT",
            "JUMP":     lambda p: "JUMP",
        }

        if action in mapping:
            resolved = mapping[action](parsed)
            return ACTION_IDX.get(resolved, ACTION_IDX["WAIT"])
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass
    return ACTION_IDX["WAIT"]


if __name__ == "__main__":
    # Quick sanity test (without a real bot)
    print("Testing environment creation...")
    env = MinecraftRLEnv()
    print(f"Obs space : {env.observation_space.shape}")
    print(f"Act space : {env.action_space.n} actions: {ACTIONS}")

    dummy_state = {
        "hp": 18, "food": 16, "pos": {"x": 10, "y": 64, "z": -5},
        "inv": {"oak_log": 5, "wooden_planks": 12, "stick": 4},
        "nearby": [{"name": "oak_log", "dist": 2.1}],
        "mobs": [], "time": 6000, "raining": False, "goal": "craft_iron_pickaxe"
    }
    env.last_state = dummy_state
    obs = env._encode_state(dummy_state)
    print(f"Encoded obs shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]")
    print("✅ Environment OK")
