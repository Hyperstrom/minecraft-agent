"""
train_rl.py
===========
Phase 4: Reinforcement Learning training loop.
Runs the Phase 3 LLM as a policy in the Minecraft environment using PPO.

How it works:
  1. LLM receives JSON game state as input text
  2. LLM generates JSON action as output
  3. Action is sent to Mineflayer bot via HTTP
  4. Reward signal comes back from the environment
  5. PPO updates the LLM's LoRA weights based on reward

Architecture:
  LLM Policy (Qwen-1.5B + LoRA)  →  JSON action  →  MinecraftRLEnv  →  reward
       ↑_____________________________ PPO gradient __________________________↑

Requirements:
    pip install trl>=0.8 torch transformers peft gymnasium requests tqdm

Run:
    # First start the Mineflayer RL server:
    node bot/rl_server.js

    # Then train:
    python rl_training/train_rl.py --model Tron101101/mineagent-phase3-lora --steps 50000
"""

import os
import json
import time
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig
from torch.optim import AdamW

from rl_environment import MinecraftRLEnv, llm_action_to_index, ACTIONS


# ── PPO Hyperparameters ────────────────────────────────────────────────────────
PPO_CONFIG = {
    "clip_epsilon":     0.2,    # PPO clip range
    "value_coef":       0.5,    # value loss weight
    "entropy_coef":     0.01,   # entropy bonus (encourages exploration)
    "gamma":            0.99,   # discount factor
    "gae_lambda":       0.95,   # GAE lambda
    "lr":               1e-5,   # very low LR for RL fine-tuning
    "batch_size":       8,      # rollout batch size
    "ppo_epochs":       4,      # PPO update epochs per rollout
    "max_grad_norm":    0.5,    # gradient clipping
}


class LLMPolicy:
    """
    Wraps the trained LLM for use as a PPO policy.
    
    The LLM acts as both actor (action distribution) and 
    the reward model gives external signal from the environment.
    
    For RL, we treat the LLM's log-probability of the generated
    action token as our policy log-prob for PPO.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = device
        print(f"Loading policy model: {model_id}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Try loading as a LoRA model first, fall back to base
        try:
            from unsloth import FastLanguageModel
            self.model, self.tok = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
            )
            # Add a small NEW LoRA adapter for RL fine-tuning
            # (separate from the supervised LoRA weights)
            self.model = FastLanguageModel.get_peft_model(
                self.model, r=16, lora_alpha=32,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                 'gate_proj', 'up_proj', 'down_proj'],
                lora_dropout=0.0, bias='none',
            )
        except Exception as e:
            print(f"Unsloth load failed ({e}), using standard HF...")
            self.tok = AutoTokenizer.from_pretrained(model_id)
            base = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, device_map="auto")
            lora_config = LoraConfig(r=16, lora_alpha=32,
                                      target_modules=['q_proj', 'v_proj'],
                                      lora_dropout=0.0)
            self.model = get_peft_model(base, lora_config)

        self.tok.pad_token = self.tok.eos_token
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=PPO_CONFIG["lr"]
        )
        print(f"✅ Policy loaded. Trainable params: "
              f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def get_action(self, state_json: str, goal: str, temperature: float = 0.8) -> Tuple[str, float]:
        """
        Run inference: state JSON → JSON action string + log probability.
        Returns: (action_json_str, log_prob)
        """
        state = json.loads(state_json)
        state["goal"] = goal
        prompt = json.dumps(state, separators=(',', ':'))

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs,
                max_new_tokens=48,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tok.eos_token_id,
            )

        # Extract generated tokens and their log probs
        gen_ids = output.sequences[0][inputs.shape[1]:]
        action_text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()

        # Compute log probability of the generated sequence
        scores = output.scores  # list of (vocab_size,) tensors
        log_prob = 0.0
        for i, score in enumerate(scores):
            if i < len(gen_ids):
                probs = torch.softmax(score, dim=-1)
                token_prob = probs[0, gen_ids[i]].item()
                log_prob += np.log(max(token_prob, 1e-10))

        return action_text, log_prob

    def compute_loss(self, state_jsons: List[str], actions: List[str],
                     old_log_probs: List[float], advantages: List[float],
                     returns: List[float]) -> torch.Tensor:
        """PPO loss computation."""
        total_loss = torch.tensor(0.0, requires_grad=True, device=self.device)

        for state_json, action, old_lp, adv, ret in zip(
                state_jsons, actions, old_log_probs, advantages, returns):

            # Build the full conversation (state + action)
            messages = [
                {"role": "user",      "content": state_json},
                {"role": "assistant", "content": action},
            ]
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            inputs = self.tok(text, return_tensors="pt",
                              truncation=True, max_length=512).to(self.device)

            # Forward pass to get log probs
            with torch.enable_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                new_log_prob = -outputs.loss.item() * inputs["input_ids"].shape[1]

            # PPO ratio
            ratio = torch.exp(torch.tensor(new_log_prob - old_lp))
            adv_t = torch.tensor(adv, dtype=torch.float32)

            # Clipped PPO objective
            policy_loss = -torch.min(
                ratio * adv_t,
                torch.clamp(ratio, 1 - PPO_CONFIG["clip_epsilon"],
                            1 + PPO_CONFIG["clip_epsilon"]) * adv_t
            )
            total_loss = total_loss + policy_loss + PPO_CONFIG["value_coef"] * outputs.loss

        return total_loss / len(state_jsons)


# ── Rollout Buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.states:     List[str]   = []
        self.actions:    List[str]   = []
        self.rewards:    List[float] = []
        self.log_probs:  List[float] = []
        self.dones:      List[bool]  = []

    def add(self, state_json, action_json, reward, log_prob, done):
        self.states.append(state_json)
        self.actions.append(action_json)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_advantages(self) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages and discounted returns."""
        gamma      = PPO_CONFIG["gamma"]
        gae_lambda = PPO_CONFIG["gae_lambda"]
        n = len(self.rewards)
        advantages = [0.0] * n
        returns    = [0.0] * n

        gae = 0.0
        for t in reversed(range(n)):
            next_val  = 0.0 if self.dones[t] else self.rewards[t]
            delta     = self.rewards[t] + gamma * next_val - 0.0
            gae       = delta + gamma * gae_lambda * (0 if self.dones[t] else gae)
            advantages[t] = gae
            returns[t]    = gae

        # Normalize advantages
        adv_arr = np.array(advantages)
        if adv_arr.std() > 1e-8:
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
        return adv_arr.tolist(), returns

    def clear(self):
        self.__init__()


# ── Main Training Loop ─────────────────────────────────────────────────────────

def train_rl(model_id: str, total_steps: int = 50000, goal: str = "craft_iron_pickaxe",
             save_every: int = 5000, output_dir: str = "./checkpoints"):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Phase 4: Reinforcement Learning")
    print(f"  Model  : {model_id}")
    print(f"  Goal   : {goal}")
    print(f"  Steps  : {total_steps:,}")
    print(f"{'='*60}\n")

    # Init environment and policy
    env    = MinecraftRLEnv(goal=goal, max_steps=200)
    policy = LLMPolicy(model_id=model_id)
    buffer = RolloutBuffer()

    # Metrics
    episode_rewards = []
    episode_lengths = []
    episode_reward  = 0.0
    episode_length  = 0
    total_episodes  = 0
    best_reward     = float('-inf')

    pbar = tqdm(total=total_steps, desc="🎮 RL Training", dynamic_ncols=True)

    obs, _ = env.reset()
    step   = 0

    while step < total_steps:
        # ── Collect rollout ───────────────────────────────────────
        state_json = env.state_to_json()
        action_json, log_prob = policy.get_action(state_json, goal)

        # Map LLM output to discrete action
        action_idx = llm_action_to_index(action_json)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated

        buffer.add(state_json, action_json, reward, log_prob, done)

        episode_reward += reward
        episode_length += 1
        step += 1

        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            total_episodes += 1
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

        # ── PPO Update every batch_size steps ─────────────────────
        if len(buffer.states) >= PPO_CONFIG["batch_size"]:
            advantages, returns = buffer.compute_advantages()

            policy.model.train()
            update_losses = []

            for _ in range(PPO_CONFIG["ppo_epochs"]):
                loss = policy.compute_loss(
                    buffer.states, buffer.actions,
                    buffer.log_probs, advantages, returns
                )
                policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.model.parameters(), PPO_CONFIG["max_grad_norm"])
                policy.optimizer.step()
                update_losses.append(loss.item())

            policy.model.eval()
            buffer.clear()

            # Update progress bar
            mean_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            pbar.set_postfix({
                "ep_reward": f"{mean_reward:.2f}",
                "episodes":  total_episodes,
                "loss":      f"{np.mean(update_losses):.4f}",
            })
            tqdm.write(
                f"  Step {step:>6}/{total_steps} | "
                f"Episodes: {total_episodes:>4} | "
                f"Mean Reward (10ep): {mean_reward:>7.2f} | "
                f"PPO Loss: {np.mean(update_losses):.4f}"
            )

        pbar.update(1)

        # ── Save checkpoint ────────────────────────────────────────
        if step % save_every == 0:
            ckpt_path = f"{output_dir}/step_{step}"
            policy.model.save_pretrained(ckpt_path)
            policy.tok.save_pretrained(ckpt_path)

            mean_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_path   = f"{output_dir}/best_model"
                policy.model.save_pretrained(best_path)
                policy.tok.save_pretrained(best_path)
                tqdm.write(f"  ⭐ New best! Mean reward: {best_reward:.2f} → {best_path}")
            else:
                tqdm.write(f"  💾 Checkpoint saved → {ckpt_path}")

    pbar.close()
    env.close()

    # Final save
    final_path = f"{output_dir}/final_rl_model"
    policy.model.save_pretrained(final_path)
    policy.tok.save_pretrained(final_path)

    print(f"\n{'='*60}")
    print(f"  ✅ RL Training Complete!")
    print(f"  Total steps    : {step:,}")
    print(f"  Total episodes : {total_episodes:,}")
    print(f"  Best reward    : {best_reward:.2f}")
    print(f"  Final model    : {final_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Tron101101/mineagent-phase3-lora")
    parser.add_argument("--steps",      type=int, default=50000)
    parser.add_argument("--goal",       default="craft_iron_pickaxe")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--output",     default="./checkpoints")
    args = parser.parse_args()

    train_rl(
        model_id=args.model,
        total_steps=args.steps,
        goal=args.goal,
        save_every=args.save_every,
        output_dir=args.output,
    )
