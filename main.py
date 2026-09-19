"""
Stardew Valley Fishing Minigame — Dueling Double DQN (8-D)
=========================================================
Architecture locked by HANDOFF.md:
  • 8-D observation (no hand-engineered physics features)
  • Dueling Double DQN with 3-step returns
  • Uniform replay, epsilon-greedy, hard target updates
  • Checkpoint artefacts: action traces + eval_score (greedy fixed-suite)
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from collections import deque, namedtuple
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment import OBS_DIM, FishingMinigameEnv
from eval_metrics import evaluate_checkpoint, format_eval_summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))


# ═══════════════════════════════════════════════════════════════════════════
#  VECTORIZED ENV
# ═══════════════════════════════════════════════════════════════════════════
class VectorizedEnv:
    def __init__(self, num_envs=4, **env_kwargs):
        self.num_envs = num_envs
        env_kwargs["render_mode"] = None
        self.envs = [FishingMinigameEnv(**env_kwargs) for _ in range(num_envs)]
        self.dones = [False] * num_envs

    def reset(self):
        states = [env.reset() for env in self.envs]
        self.dones = [False] * self.num_envs
        return np.array(states, dtype=np.float32)

    def step(self, actions):
        results = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.dones[i]:
                state = env.reset()
                self.dones[i] = False
                results.append((state, 0.0, False, {}))
            else:
                results.append(env.step(action))
                self.dones[i] = results[-1][2]
        states, rewards, dones, infos = zip(*results)
        return (
            np.array(states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos),
        )

    def close(self):
        for env in self.envs:
            env.close()


# ═══════════════════════════════════════════════════════════════════════════
#  MILESTONE TRACKER
# ═══════════════════════════════════════════════════════════════════════════
class MilestoneTracker:
    """Track and log major training milestones for YouTube video storytelling."""

    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.milestones = {
            "first_success": False,
            "first_win_streak_3": False,
            "first_win_streak_5": False,
            "first_win_streak_10": False,
            "first_win_streak_25": False,
            "first_win_streak_50": False,
            "first_win_streak_100": False,
            "easy_mastery_75": False,
            "easy_mastery_90": False,
            "easy_mastery_95": False,
            "medium_unlocked": False,
            "medium_mastery_75": False,
            "medium_mastery_90": False,
            "hard_unlocked": False,
            "hard_mastery_75": False,
            "hard_mastery_90": False,
            "overall_50_percent": False,
            "overall_75_percent": False,
            "overall_80_percent": False,
            "overall_90_percent": False,
            "overall_95_percent": False,
            "overall_99_percent": False,
            "first_sinker_catch": False,
            "first_dart_catch": False,
            "first_smooth_catch": False,
            "first_mixed_catch": False,
            "first_floater_catch": False,
            "sinker_mastery_80": False,
            "dart_mastery_80": False,
            "smooth_mastery_80": False,
            "mixed_mastery_80": False,
            "floater_mastery_80": False,
            "epsilon_below_0_5": False,
            "epsilon_below_0_25": False,
            "epsilon_below_0_1": False,
            "epsilon_below_0_01": False,
            "episode_100": False,
            "episode_500": False,
            "episode_1000": False,
            "episode_2500": False,
            "episode_5000": False,
            "first_perfect_eval": False,
            "first_flawless_20": False,
            "speed_demon_50": False,
            "comeback_after_5_losses": False,
            "all_behaviors_caught": False,
            "eval_score_50": False,
            "eval_score_80": False,
        }
        self.current_win_streak = 0
        self.max_win_streak = 0
        self.consecutive_losses = 0
        self.prev_epsilon = 1.0
        self.behavior_stats = {
            b: {"attempts": 0, "successes": 0}
            for b in ["sinker", "dart", "smooth", "mixed", "floater"]
        }
        self.behaviors_caught = set()
        self.shortest_catch = float("inf")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"training_metrics_{timestamp}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "Episode",
                "Score",
                "Success",
                "Fish",
                "Difficulty",
                "Behavior",
                "Episode_Length",
                "Epsilon",
                "Win_Streak",
                "Avg_Score_100",
                "Win_Rate_100",
                "Easy_Success_Rate",
                "Medium_Success_Rate",
                "Hard_Success_Rate",
                "Shortest_Catch",
                "Behaviors_Discovered",
                "Sinker_Rate",
                "Dart_Rate",
                "Smooth_Rate",
                "Mixed_Rate",
                "Floater_Rate",
                "Tap_Frequency",
                "Mean_Centering_Error",
                "Eval_Score",
                "Eval_Catch_Rate",
                "Eval_Score_Hard",
            ]
        )
        self.last_eval: dict | None = None
        self.milestone_log_path = os.path.join(log_dir, f"milestones_{timestamp}.txt")
        with open(self.milestone_log_path, "w") as f:
            f.write(f"Training Milestones Log - Started {timestamp}\n{'=' * 60}\n\n")

    def check_milestone(self, name, condition, episode, message):
        if not self.milestones.get(name, False) and condition:
            self.milestones[name] = True
            log_msg = f"MILESTONE at Episode {episode}: {message}"
            print(f"\n{'=' * 60}\n{log_msg}\n{'=' * 60}\n")
            with open(self.milestone_log_path, "a", encoding="utf-8") as f:
                f.write(f"Episode {episode}: {message}\n")
            return True
        return False

    def record_eval(self, episode: int, result: dict) -> None:
        """Append a checkpoint eval row (Eval_* columns populated)."""
        self.last_eval = result
        self.csv_writer.writerow(
            [
                episode,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                f"{result['eval_score']:.4f}",
                f"{result['eval_catch_rate']:.4f}",
                f"{result['eval_score_hard']:.4f}",
            ]
        )
        self.csv_file.flush()
        self.check_milestone(
            f"eval_score_50",
            result["eval_score"] >= 0.5,
            episode,
            f"eval_score reached 50% ({result['eval_score']:.1%})",
        )
        self.check_milestone(
            f"eval_score_80",
            result["eval_score"] >= 0.8,
            episode,
            f"eval_score reached 80% ({result['eval_score']:.1%})",
        )

    def update(self, episode, score, success, fish_info, epsilon, stats, tap_hz=0.0, centering_err=0.0):
        behavior = fish_info["behavior"]
        if behavior in self.behavior_stats:
            self.behavior_stats[behavior]["attempts"] += 1
            if success:
                self.behavior_stats[behavior]["successes"] += 1
                self.behaviors_caught.add(behavior)
        if success and fish_info["episode_length"] < self.shortest_catch:
            self.shortest_catch = fish_info["episode_length"]
        if success:
            self.current_win_streak += 1
            self.max_win_streak = max(self.max_win_streak, self.current_win_streak)
            self.consecutive_losses = 0
        else:
            self.current_win_streak = 0
            self.consecutive_losses += 1

        behavior_rates = {}
        for btype, data in self.behavior_stats.items():
            behavior_rates[btype] = (
                data["successes"] / data["attempts"] * 100 if data["attempts"] > 0 else 0.0
            )

        self.csv_writer.writerow(
            [
                episode,
                f"{score:.2f}",
                1 if success else 0,
                fish_info["name"],
                fish_info["difficulty"],
                fish_info["behavior"],
                fish_info["episode_length"],
                f"{epsilon:.4f}",
                self.current_win_streak,
                f"{stats['avg_score']:.2f}",
                f"{stats['win_rate']:.1f}",
                f"{stats['easy_success_rate']:.1f}",
                f"{stats['medium_success_rate']:.1f}",
                f"{stats['hard_success_rate']:.1f}",
                self.shortest_catch if self.shortest_catch != float("inf") else 0,
                len(self.behaviors_caught),
                f"{behavior_rates.get('sinker', 0):.1f}",
                f"{behavior_rates.get('dart', 0):.1f}",
                f"{behavior_rates.get('smooth', 0):.1f}",
                f"{behavior_rates.get('mixed', 0):.1f}",
                f"{behavior_rates.get('floater', 0):.1f}",
                f"{tap_hz:.3f}",
                f"{centering_err:.4f}",
                "",  # Eval_Score — filled only via record_eval at checkpoints
                "",
                "",
            ]
        )
        self.csv_file.flush()

        for ep, key in [
            (100, "episode_100"),
            (500, "episode_500"),
            (1000, "episode_1000"),
            (2500, "episode_2500"),
            (5000, "episode_5000"),
        ]:
            if episode == ep:
                self.check_milestone(key, True, episode, f"Reached episode {ep}!")

        if success:
            self.check_milestone(
                "first_success", True, episode, f"First successful catch! ({fish_info['name']})"
            )

        for streak, key in [
            (3, "first_win_streak_3"),
            (5, "first_win_streak_5"),
            (10, "first_win_streak_10"),
            (25, "first_win_streak_25"),
            (50, "first_win_streak_50"),
            (100, "first_win_streak_100"),
        ]:
            if self.current_win_streak == streak:
                self.check_milestone(key, True, episode, f"First {streak}-episode win streak!")
                break

        if self.current_win_streak >= 20:
            self.check_milestone(
                "first_flawless_20",
                True,
                episode,
                f"Flawless! {self.current_win_streak}-episode win streak!",
            )

        if success:
            bmap = {
                "sinker": "first_sinker_catch",
                "dart": "first_dart_catch",
                "smooth": "first_smooth_catch",
                "mixed": "first_mixed_catch",
                "floater": "first_floater_catch",
            }
            if behavior in bmap:
                self.check_milestone(
                    bmap[behavior],
                    True,
                    episode,
                    f"First {behavior} fish caught! ({fish_info['name']})",
                )

        if len(self.behaviors_caught) >= 5:
            self.check_milestone(
                "all_behaviors_caught", True, episode, "Caught all 5 fish behavior types!"
            )

        for btype, rate in behavior_rates.items():
            mk = f"{btype}_mastery_80"
            if self.behavior_stats[btype]["attempts"] >= 20 and rate >= 80:
                self.check_milestone(
                    mk, True, episode, f"{btype.capitalize()} fish mastered! ({rate:.1f}% success)"
                )

        if success and fish_info["episode_length"] < 50:
            self.check_milestone(
                "speed_demon_50",
                True,
                episode,
                f"Speed demon! Caught in {fish_info['episode_length']} steps!",
            )
        if success and self.consecutive_losses >= 5:
            self.check_milestone(
                "comeback_after_5_losses",
                True,
                episode,
                f"Comeback! Won after {self.consecutive_losses} losses!",
            )

        for thr, key in [
            (0.5, "epsilon_below_0_5"),
            (0.25, "epsilon_below_0_25"),
            (0.1, "epsilon_below_0_1"),
            (0.01, "epsilon_below_0_01"),
        ]:
            if self.prev_epsilon >= thr and epsilon < thr:
                self.check_milestone(key, True, episode, f"Epsilon below {thr} ({epsilon:.4f})")
        self.prev_epsilon = epsilon

        def _check_difficulty_level(level_name, rate_field, enabled_field):
            lvl = stats.get(rate_field, 0)
            if stats.get(enabled_field, False):
                self.check_milestone(
                    f"{level_name}_unlocked", True, episode, f"{level_name.capitalize()} unlocked!"
                )
                if lvl >= 75:
                    self.check_milestone(
                        f"{level_name}_mastery_75",
                        True,
                        episode,
                        f"{level_name.capitalize()} mastery! ({lvl:.1f}%)",
                    )
                if lvl >= 90:
                    self.check_milestone(
                        f"{level_name}_mastery_90",
                        True,
                        episode,
                        f"{level_name.capitalize()} excellence! ({lvl:.1f}%)",
                    )

        _check_difficulty_level("easy", "easy_success_rate", "easy_enabled")
        _check_difficulty_level("medium", "medium_success_rate", "medium_enabled")
        _check_difficulty_level("hard", "hard_success_rate", "hard_enabled")

        overall = stats["win_rate"]
        for thr, key in [
            (50, "overall_50_percent"),
            (75, "overall_75_percent"),
            (80, "overall_80_percent"),
            (90, "overall_90_percent"),
            (95, "overall_95_percent"),
            (99, "overall_99_percent"),
        ]:
            if overall >= thr:
                self.check_milestone(key, True, episode, f"{thr}% win rate achieved! ({overall:.1f}%)")

    def close(self):
        self.csv_file.close()
        print(f"\nTraining metrics saved to: {self.csv_path}")
        print(f"Milestone log saved to: {self.milestone_log_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  REPLAY + N-STEP
# ═══════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_dim = batch[0].state.shape[0]
        s = np.zeros((batch_size, state_dim), dtype=np.float32)
        a = np.zeros((batch_size, 1), dtype=np.int64)
        r = np.zeros((batch_size, 1), dtype=np.float32)
        n = np.zeros((batch_size, state_dim), dtype=np.float32)
        d = np.zeros((batch_size, 1), dtype=np.float32)
        for i, e in enumerate(batch):
            s[i] = e.state
            a[i] = e.action
            r[i] = e.reward
            n[i] = e.next_state
            d[i] = e.done
        return (
            torch.from_numpy(s).to(device),
            torch.from_numpy(a).to(device),
            torch.from_numpy(r).to(device),
            torch.from_numpy(n).to(device),
            torch.from_numpy(d).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class NStepBuffer:
    """Accumulates n-step transitions before pushing to the replay buffer."""

    def __init__(self, n=3, gamma=0.99):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) < self.n and not done:
            return None

        # Build n-step (or shorter if episode ended early)
        R = 0.0
        for i, (_, _, r, _, _) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
        s0, a0, _, _, _ = self.buffer[0]
        _, _, _, sn, dn = self.buffer[-1]
        self.buffer.popleft()
        if done:
            # Flush remaining truncated n-step returns
            flushed = [(s0, a0, R, sn, float(dn))]
            while self.buffer:
                R = 0.0
                for i, (_, _, r, _, _) in enumerate(self.buffer):
                    R += (self.gamma ** i) * r
                s0, a0, _, _, _ = self.buffer[0]
                _, _, _, sn, dn = self.buffer[-1]
                flushed.append((s0, a0, R, sn, float(dn)))
                self.buffer.popleft()
            return flushed
        return [(s0, a0, R, sn, float(dn))]

    def reset(self):
        self.buffer.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  NETWORK — Dueling DQN (HANDOFF §1.3)
# ═══════════════════════════════════════════════════════════════════════════
class DuelingDQN(nn.Module):
    """
    Linear(8→128)→ReLU → Linear(128→64)→ReLU → Value / Advantage streams.
    Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))
    """

    def __init__(self, state_dim=OBS_DIM, action_dim=2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        h = self.feature(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))


# ═══════════════════════════════════════════════════════════════════════════
#  AGENT — Double DQN + n-step + epsilon-greedy
# ═══════════════════════════════════════════════════════════════════════════
class DQNAgent:
    def __init__(
        self,
        state_dim=OBS_DIM,
        action_dim=2,
        lr=2e-4,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=128,
        update_every=4,
        n_step=3,
        target_update_freq=500,
        grad_clip=1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        self.n_step = n_step
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip

        self.q_network = DuelingDQN(state_dim, action_dim).to(device)
        self.target_network = DuelingDQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.n_step_buffers = {}  # env_id -> NStepBuffer

        self.t_step = 0
        self.learn_step = 0

    def _n_buf(self, env_id=0):
        if env_id not in self.n_step_buffers:
            self.n_step_buffers[env_id] = NStepBuffer(self.n_step, self.gamma)
        return self.n_step_buffers[env_id]

    def step(self, state, action, reward, next_state, done, env_id=0):
        transitions = self._n_buf(env_id).push(state, action, reward, next_state, done)
        if transitions:
            for s, a, r, ns, d in transitions:
                self.memory.add(s, a, r, ns, d)
        if done:
            self._n_buf(env_id).reset()

        self.t_step += 1
        if self.t_step % self.update_every == 0 and len(self.memory) >= self.batch_size:
            self.learn()

    def act(self, state, eps=0.0):
        if random.random() < eps:
            return random.randrange(self.action_dim)
        state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            q = self.q_network(state_t)
        self.q_network.train()
        return int(q.argmax(dim=1).item())

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Double DQN: online net selects, target net evaluates
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            # n-step return already baked into rewards; discount remaining with gamma^n
            targets = rewards + (self.gamma ** self.n_step) * next_q * (1.0 - dones)

        current_q = self.q_network(states).gather(1, actions)
        loss = F.smooth_l1_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "q_state_dict": self.q_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "architecture": "DuelingDoubleDQN",
                "obs_dim": OBS_DIM,
            },
            path,
        )
        print(f"Saved checkpoint: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.q_network.load_state_dict(ckpt["q_state_dict"])
        self.target_network.load_state_dict(ckpt.get("target_state_dict", ckpt["q_state_dict"]))
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded checkpoint: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CHECKPOINT ARTEFACTS (HANDOFF §2.1)
# ═══════════════════════════════════════════════════════════════════════════
TRACE_FISH = "Carp"
TRACE_SEED = 42


def record_action_trace(agent, fish_name=TRACE_FISH, seed=TRACE_SEED, max_t=2000):
    """Record (t, action, bobber_pos, bar_center, in_bar) for website replay."""
    env = FishingMinigameEnv(
        render_mode=None, seed=seed, fish_name=fish_name, augment_fish=False
    )
    env.fish_name = fish_name
    state = env.reset(seed=seed)

    trace = []
    done = False
    t = 0
    while not done and t < max_t:
        action = agent.act(state, eps=0.0)
        bar_center = env.bobberBarPos + env.bobberBarHeight * 0.5
        trace.append(
            {
                "t": t,
                "action": int(action),
                "bobber_pos": float(env.bobberPosition),
                "bar_center": float(bar_center),
                "in_bar": bool(env.bobberInBar),
            }
        )
        state, _, done, _ = env.step(action)
        t += 1

    success = bool(env.distanceFromCatching >= 1.0)
    env.close()
    return {
        "fish": fish_name,
        "seed": seed,
        "success": success,
        "length": len(trace),
        "frames": trace,
    }


def save_checkpoint_artefacts(agent, episode, tracker=None, out_dir="training_logs/evolution"):
    os.makedirs(out_dir, exist_ok=True)
    trace = record_action_trace(agent)

    print(f"  Running checkpoint eval (greedy, fixed fish list)…")
    eval_result = evaluate_checkpoint(agent)
    if tracker is not None:
        tracker.record_eval(episode, eval_result)

    # Tap frequency + mean centering error from the trace
    actions = [f["action"] for f in trace["frames"]]
    taps = sum(1 for i in range(1, len(actions)) if actions[i] == 1 and actions[i - 1] == 0)
    duration_s = max(len(actions), 1) / 60.0
    tap_hz = taps / duration_s
    centering = []
    for f in trace["frames"]:
        centering.append(abs(f["bar_center"] - f["bobber_pos"]) / 568.0)
    mean_centering = float(np.mean(centering)) if centering else 0.0

    # Keep per_behaviour shape compatible with older evolution JSON consumers
    per_behaviour = {
        "rates": eval_result["eval_by_behaviour"],
        "raw": {
            b: {
                "attempts": sum(
                    1
                    for f in eval_result["per_fish"]
                    if f["behaviour"] == b
                )
                * eval_result["seeds_per_fish"],
                "successes": int(
                    round(
                        eval_result["eval_by_behaviour"][b]
                        * sum(1 for f in eval_result["per_fish"] if f["behaviour"] == b)
                        * eval_result["seeds_per_fish"]
                    )
                ),
            }
            for b in eval_result["eval_by_behaviour"]
        },
    }

    payload = {
        "episode": episode,
        "trace": trace,
        "per_behaviour": per_behaviour,
        "tap_frequency_hz": tap_hz,
        "mean_centering_error": mean_centering,
        "eval_score": eval_result["eval_score"],
        "eval_catch_rate": eval_result["eval_catch_rate"],
        "eval_score_hard": eval_result["eval_score_hard"],
        "eval": eval_result,
    }
    path = os.path.join(out_dir, f"episode_{episode}.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"  Evolution artefact: {path}")
    print(f"  {format_eval_summary(eval_result)}")
    print(f"  (tap={tap_hz:.2f}Hz, center_err={mean_centering:.4f})")
    return payload


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def _episode_stats(scores_window, successes_window, difficulty_buckets):
    avg_score = float(np.mean(scores_window)) if scores_window else 0.0
    win_rate = (
        100.0 * sum(successes_window) / len(successes_window) if successes_window else 0.0
    )

    def rate(bucket):
        if not bucket:
            return 0.0, False
        return 100.0 * sum(bucket) / len(bucket), True

    easy_r, easy_on = rate(difficulty_buckets["easy"])
    med_r, med_on = rate(difficulty_buckets["medium"])
    hard_r, hard_on = rate(difficulty_buckets["hard"])
    return {
        "avg_score": avg_score,
        "win_rate": win_rate,
        "easy_success_rate": easy_r,
        "medium_success_rate": med_r,
        "hard_success_rate": hard_r,
        "easy_enabled": easy_on,
        "medium_enabled": med_on,
        "hard_enabled": hard_on,
    }


def train_dqn(
    env,
    agent,
    n_episodes=10_000,
    max_t=2000,
    eps_start=1.0,
    eps_end=0.02,
    eps_decay=0.9995,
    save_every=500,
    render_every=0,
):
    scores = []
    scores_window = deque(maxlen=100)
    successes_window = deque(maxlen=100)
    difficulty_buckets = {
        "easy": deque(maxlen=100),
        "medium": deque(maxlen=100),
        "hard": deque(maxlen=100),
    }
    tracker = MilestoneTracker()
    eps = eps_start
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("training_logs/graphs", exist_ok=True)

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0.0
        actions_ep = []
        centering_errs = []

        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done, env_id=0)

            bar_center = env.bobberBarPos + env.bobberBarHeight * 0.5
            centering_errs.append(abs(bar_center - env.bobberPosition) / env.track_height)
            actions_ep.append(action)

            state = next_state
            score += reward
            if done:
                break

        success = env.distanceFromCatching >= 1.0
        scores.append(score)
        scores_window.append(score)
        successes_window.append(1 if success else 0)

        diff = info.get("fish_difficulty", env.difficulty)
        if diff < 40:
            difficulty_buckets["easy"].append(1 if success else 0)
        elif diff < 70:
            difficulty_buckets["medium"].append(1 if success else 0)
        else:
            difficulty_buckets["hard"].append(1 if success else 0)

        taps = sum(
            1 for i in range(1, len(actions_ep)) if actions_ep[i] == 1 and actions_ep[i - 1] == 0
        )
        tap_hz = taps / (max(len(actions_ep), 1) / 60.0)
        mean_centering = float(np.mean(centering_errs)) if centering_errs else 0.0

        stats = _episode_stats(scores_window, successes_window, difficulty_buckets)
        fish_info = {
            "name": info.get("fish_name", "?"),
            "difficulty": diff,
            "behavior": info.get("fish_behaviour", "?").lower(),
            "episode_length": info.get("episode_length", t + 1),
        }
        tracker.update(
            i_episode, score, success, fish_info, eps, stats, tap_hz, mean_centering
        )

        eps = max(eps_end, eps * eps_decay)

        if i_episode % 20 == 0:
            print(
                f"Ep {i_episode:5d} | score={score:7.1f} | "
                f"avg100={stats['avg_score']:7.1f} | win={stats['win_rate']:5.1f}% | "
                f"eps={eps:.3f} | tap={tap_hz:.1f}Hz"
            )

        if i_episode % save_every == 0:
            ckpt = f"models/checkpoints/episode_{i_episode}.pth"
            agent.save(ckpt)
            save_checkpoint_artefacts(agent, i_episode, tracker=tracker)
            _plot_training(scores, i_episode)

        if render_every and i_episode % render_every == 0:
            env.set_render_mode("human")
            _demo_episode(env, agent)
            env.set_render_mode(None)

    tracker.close()
    return scores


def train_dqn_vectorized(
    env_vec,
    agent,
    n_episodes=10_000,
    max_t=2000,
    eps_start=1.0,
    eps_end=0.02,
    eps_decay=0.9995,
    save_every=500,
    render_every=0,
):
    """Vectorized training — counts completed episodes across parallel envs."""
    scores = []
    scores_window = deque(maxlen=100)
    successes_window = deque(maxlen=100)
    difficulty_buckets = {
        "easy": deque(maxlen=100),
        "medium": deque(maxlen=100),
        "hard": deque(maxlen=100),
    }
    tracker = MilestoneTracker()
    eps = eps_start
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("training_logs/graphs", exist_ok=True)

    states = env_vec.reset()
    ep_scores = [0.0] * env_vec.num_envs
    ep_actions = [[] for _ in range(env_vec.num_envs)]
    ep_centering = [[] for _ in range(env_vec.num_envs)]
    completed = 0
    next_save = save_every

    while completed < n_episodes:
        actions = [agent.act(states[i], eps) for i in range(env_vec.num_envs)]
        next_states, rewards, dones, infos = env_vec.step(actions)

        for i in range(env_vec.num_envs):
            agent.step(
                states[i], actions[i], float(rewards[i]), next_states[i], bool(dones[i]), env_id=i
            )
            ep_scores[i] += float(rewards[i])
            ep_actions[i].append(actions[i])
            env = env_vec.envs[i]
            bar_center = env.bobberBarPos + env.bobberBarHeight * 0.5
            ep_centering[i].append(abs(bar_center - env.bobberPosition) / env.track_height)

            if dones[i]:
                completed += 1
                success = env.distanceFromCatching >= 1.0
                score = ep_scores[i]
                info = infos[i] if infos[i] else {}
                scores.append(score)
                scores_window.append(score)
                successes_window.append(1 if success else 0)

                diff = info.get("fish_difficulty", env.difficulty)
                if diff < 40:
                    difficulty_buckets["easy"].append(1 if success else 0)
                elif diff < 70:
                    difficulty_buckets["medium"].append(1 if success else 0)
                else:
                    difficulty_buckets["hard"].append(1 if success else 0)

                taps = sum(
                    1
                    for j in range(1, len(ep_actions[i]))
                    if ep_actions[i][j] == 1 and ep_actions[i][j - 1] == 0
                )
                tap_hz = taps / (max(len(ep_actions[i]), 1) / 60.0)
                mean_centering = float(np.mean(ep_centering[i])) if ep_centering[i] else 0.0

                stats = _episode_stats(scores_window, successes_window, difficulty_buckets)
                fish_info = {
                    "name": info.get("fish_name", "?"),
                    "difficulty": diff,
                    "behavior": str(info.get("fish_behaviour", "?")).lower(),
                    "episode_length": info.get("episode_length", len(ep_actions[i])),
                }
                tracker.update(
                    completed, score, success, fish_info, eps, stats, tap_hz, mean_centering
                )
                eps = max(eps_end, eps * eps_decay)

                ep_scores[i] = 0.0
                ep_actions[i] = []
                ep_centering[i] = []

                if completed % 20 == 0:
                    print(
                        f"Ep {completed:5d} | score={score:7.1f} | "
                        f"avg100={stats['avg_score']:7.1f} | win={stats['win_rate']:5.1f}% | "
                        f"eps={eps:.3f} | tap={tap_hz:.1f}Hz"
                    )

                if completed >= next_save:
                    ckpt = f"models/checkpoints/episode_{completed}.pth"
                    agent.save(ckpt)
                    save_checkpoint_artefacts(agent, completed, tracker=tracker)
                    _plot_training(scores, completed)
                    next_save += save_every

        states = next_states

    tracker.close()
    return scores


def _plot_training(scores, episode):
    if len(scores) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(scores, alpha=0.3, label="episode")
    if len(scores) >= 50:
        kernel = np.ones(50) / 50
        smooth = np.convolve(scores, kernel, mode="valid")
        ax.plot(range(49, len(scores)), smooth, label="ma50")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title(f"Training scores @ episode {episode}")
    ax.legend()
    path = f"training_logs/graphs/episode_{episode}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _demo_episode(env, agent, max_t=2000):
    state = env.reset()
    for _ in range(max_t):
        action = agent.act(state, eps=0.0)
        state, _, done, _ = env.step(action)
        if done:
            break


def evaluate_agent(env, agent, n_episodes=20, render=False):
    if render:
        env.set_render_mode("human")
    success_count = 0
    scores = []
    behavior_results = {
        b: {"attempts": 0, "success": 0} for b in ["sinker", "dart", "smooth", "mixed", "floater"]
    }
    for i in range(n_episodes):
        state = env.reset()
        score = 0.0
        behavior = env.current_fish["behaviour"].lower()
        behavior_results[behavior]["attempts"] += 1
        while True:
            action = agent.act(state, eps=0.0)
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                if env.distanceFromCatching >= 1.0:
                    success_count += 1
                    behavior_results[behavior]["success"] += 1
                break
        scores.append(score)
        print(f"Eval {i + 1}/{n_episodes}: {score:.1f} ({'win' if env.distanceFromCatching >= 1.0 else 'lose'})")

    print(f"\nOverall: {success_count}/{n_episodes} ({100 * success_count / n_episodes:.1f}%)")
    for b, r in behavior_results.items():
        if r["attempts"]:
            print(f"  {b}: {r['success']}/{r['attempts']} ({100 * r['success'] / r['attempts']:.1f}%)")
    if render:
        env.set_render_mode(None)
    return scores, behavior_results


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    env_vec = VectorizedEnv(num_envs=4, render_mode=None)
    env = env_vec.envs[0]

    agent = DQNAgent(
        state_dim=OBS_DIM,
        action_dim=2,
        lr=2e-4,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=128,
        update_every=4,
        n_step=3,
        target_update_freq=500,
        grad_clip=1.0,
    )

    n_params = sum(p.numel() for p in agent.q_network.parameters())
    print(f"Dueling Double DQN — {OBS_DIM}-D input, {n_params:,} parameters")

    train_new_model = True
    skip_evaluation = True
    # Override via CLI: python main.py --episodes 5000 --save-every 500 --checkpoint <path> --eps-start 0.2
    n_episodes = 10_000
    if "--episodes" in sys.argv:
        idx = sys.argv.index("--episodes")
        n_episodes = int(sys.argv[idx + 1])

    checkpoint_path = None
    if "--checkpoint" in sys.argv:
        idx = sys.argv.index("--checkpoint")
        checkpoint_path = sys.argv[idx + 1]
        agent.load(checkpoint_path)

    eps_start = 0.2 if checkpoint_path else 1.0
    if "--eps-start" in sys.argv:
        idx = sys.argv.index("--eps-start")
        eps_start = float(sys.argv[idx + 1])

    save_every = 500
    if "--save-every" in sys.argv:
        idx = sys.argv.index("--save-every")
        save_every = int(sys.argv[idx + 1])
    elif n_episodes < save_every:
        save_every = max(10, n_episodes // 2)

    if train_new_model:
        scores = train_dqn_vectorized(
            env_vec=env_vec,
            agent=agent,
            n_episodes=n_episodes,
            max_t=2000,
            eps_start=eps_start,
            eps_end=0.02,
            eps_decay=0.9995,
            save_every=save_every,
            render_every=0,
        )
        # Final artefact if last episode wasn't already a save_every boundary
        if n_episodes % save_every != 0:
            final_path = f"models/checkpoints/episode_{n_episodes}.pth"
            agent.save(final_path)
            save_checkpoint_artefacts(agent, n_episodes)

    if not skip_evaluation:
        evaluate_agent(env, agent, n_episodes=20, render=False)

    env_vec.close()
