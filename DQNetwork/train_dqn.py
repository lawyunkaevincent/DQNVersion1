from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import trange

from dqn_env import DQNStepEnvironment
from dispatcher import setup_logger
from feature_extractor import flatten_decision_features
from q_network import ParametricQNetwork
from replay_buffer import ReplayBuffer, Transition


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EpisodeStats:
    episode: int
    total_reward: float
    normalised_reward: float   # reward / num_decisions — comparable across episodes
    steps: int
    mean_loss: float
    completed_requests: int
    picked_up_requests: int
    avg_wait_until_pickup: float
    avg_excess_ride_time: float
    epsilon: float
    lr: float


class DQNAgent:
    def __init__(
        self,
        feature_columns: list[str],
        scaler,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        device: torch.device,
        gamma: float,
        lr: float,
        lr_min: float,
        tau: float,
        forbid_defer_when_action_exists: bool = True,
    ):
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.forbid_defer_when_action_exists = forbid_defer_when_action_exists

        self.online_net = ParametricQNetwork(
            input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout
        ).to(device)
        self.target_net = ParametricQNetwork(
            input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout
        ).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None
        self.lr_min = lr_min

    def init_scheduler(self, total_train_steps: int) -> None:
        """Call once after warm-start, before the first gradient update."""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_train_steps), eta_min=self.lr_min
        )

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def load_warm_start(self, state_dict: dict[str, Any]) -> None:
        missing, unexpected = self.online_net.load_state_dict(state_dict, strict=False)
        self.target_net.load_state_dict(self.online_net.state_dict())
        print(f"Warm start loaded. missing={missing} unexpected={unexpected}")

    def decision_to_matrix(self, decision_point, taxi_plans) -> np.ndarray:
        rows = []
        for cand in decision_point.candidate_actions:
            feat = flatten_decision_features(
                decision_point.state_summary,
                decision_point.request,
                cand,
                taxi_plans,
                decision_point.sim_time,
            )
            missing = [c for c in self.feature_columns if c not in feat]
            if missing:
                raise KeyError(f"Missing feature columns at DQN time: {missing[:10]}")
            rows.append([float(feat[c]) for c in self.feature_columns])
        x = np.asarray(rows, dtype=np.float32)
        x = self.scaler.transform(x).astype(np.float32)
        mask = np.ones((x.shape[0], 1), dtype=np.float32)
        return np.concatenate([x, mask], axis=1)

    def _fair_random_action(
        self,
        valid_indices: list[int],
        candidate_actions: list,
    ) -> int:
        """
        Taxi-fair random exploration.

        Problem with naive random.choice(valid_indices):
          A taxi with N existing stops generates O(N²) insertion candidates,
          while an idle taxi generates exactly 1. Uniform sampling over all
          candidates therefore picks the busiest taxi with probability
          proportional to N² — a vicious circle where long stop lists
          attract more random assignments, making them even longer.

        Fix — two-level sampling:
          1. Pick a taxi uniformly at random from all taxis that have at
             least one valid candidate.
          2. Pick uniformly from that taxi's candidates.

        This gives every eligible taxi exactly the same probability of
        being chosen during exploration, regardless of how many insertion
        positions it currently has.
        """
        # Group valid candidate indices by taxi_id
        taxi_to_indices: dict[str, list[int]] = {}
        for idx in valid_indices:
            cand = candidate_actions[idx]
            key = cand.taxi_id if not cand.is_defer else "__defer__"
            taxi_to_indices.setdefault(key, []).append(idx)

        # Pick a taxi uniformly, then a candidate within it uniformly
        chosen_taxi = random.choice(list(taxi_to_indices.keys()))
        return random.choice(taxi_to_indices[chosen_taxi])

    def select_action(
        self, state_matrix: np.ndarray, decision_point, epsilon: float
    ) -> int:
        with torch.no_grad():
            inp = torch.from_numpy(state_matrix[None, :, :]).to(self.device)
            q_vals, _ = self.online_net(inp)
            scores = q_vals[0].detach().cpu().numpy().astype(float)

        valid_indices = list(range(len(scores)))
        if self.forbid_defer_when_action_exists and any(
            not c.is_defer for c in decision_point.candidate_actions
        ):
            valid_indices = [
                i for i, c in enumerate(decision_point.candidate_actions)
                if not c.is_defer
            ]
            for i, c in enumerate(decision_point.candidate_actions):
                if c.is_defer:
                    scores[i] = -1e9

        if random.random() < epsilon:
            return self._fair_random_action(valid_indices, decision_point.candidate_actions)
        return int(np.argmax(scores))

    def train_step(self, replay: ReplayBuffer, batch_size: int) -> float:
        batch = replay.sample(batch_size, self.device)
        q_values, _ = self.online_net(batch.states)
        q_sa = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online net selects action, target net evaluates it
            next_q_online, _ = self.online_net(batch.next_states)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target, _ = self.target_net(batch.next_states)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            next_q = next_q * batch.next_state_exists
            targets = batch.rewards + self.gamma * (1.0 - batch.dones) * next_q

        loss = F.smooth_l1_loss(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 5.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.soft_update()
        return float(loss.item())

    def soft_update(self) -> None:
        for target_p, online_p in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_p.data.copy_(
                target_p.data * (1.0 - self.tau) + online_p.data * self.tau
            )


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def compute_shaped_reward(
    accumulator,
    elapsed_time: float,
    chosen_is_defer: bool,
) -> float:
    """
    Shaped, normalised reward that produces a stable learning signal.

    Key changes vs original IntervalAccumulator.compute_reward():

    1. Divide cost terms by elapsed_time so the reward does not depend on
       how long the inter-decision interval happened to be.  With the original
       formulation, the same assignment can receive wildly different rewards
       (-10 vs -500) depending purely on whether the gap to the next decision
       was 5 seconds or 50 seconds — the network cannot learn from that signal.

    2. Scale the completion bonus to the same order of magnitude as the
       normalised costs (+1.0 per dropoff) so neither term dominates.

    3. Small defer penalty so the agent is discouraged from choosing DEFER
       unless it genuinely improves long-term value.
    """
    t = max(elapsed_time, 1.0)   # guard against zero-length intervals

    wait_penalty  = 0.5  * (accumulator.wait_cost      / t)
    # In compute_shaped_reward, cap wait cost per decision
    wait_penalty = 0.5 * min(accumulator.wait_cost / t, 50.0)  # cap at 50 per decision
    ride_penalty  = 0.3  * (accumulator.ride_cost       / t)
    empty_penalty = 0.001 * (accumulator.empty_dist_cost / t)

    completion_bonus = 1.0 * accumulator.completed_dropoffs
    defer_penalty    = 0.2 if chosen_is_defer else 0.0

    return completion_bonus - wait_penalty - ride_penalty - empty_penalty - defer_penalty


# ---------------------------------------------------------------------------
# Evaluation / summary helpers
# ---------------------------------------------------------------------------

def summarize_env(env: DQNStepEnvironment) -> dict[str, float]:
    requests = list(env.requests.values())
    completed = [r for r in requests if r.status.name == "COMPLETED"]
    picked_up = [r for r in requests if r.pickup_time is not None]
    dropped = [
        r for r in completed
        if r.dropoff_time is not None and r.pickup_time is not None
    ]
    avg_wait = (
        sum((r.pickup_time - r.request_time) for r in picked_up) / len(picked_up)
        if picked_up else 0.0
    )
    avg_excess = (
        sum((r.excess_ride_time or 0.0) for r in dropped) / len(dropped)
        if dropped else 0.0
    )
    return {
        "completed_requests": float(len(completed)),
        "picked_up_requests": float(len(picked_up)),
        "avg_wait_until_pickup": float(avg_wait),
        "avg_excess_ride_time": float(avg_excess),
    }


def evaluate_policy(
    cfg: str, step_length: float, use_gui: bool, agent: DQNAgent
) -> dict[str, float]:
    env = DQNStepEnvironment(
        cfg_path=cfg, step_length=step_length, use_gui=use_gui,
        policy=None, dataset_logger=None, verbose=False,
    )
    total_reward = 0.0
    steps = 0
    try:
        decision = env.reset_episode()
        while decision is not None:
            state = agent.decision_to_matrix(decision, env.taxi_plans)
            action = agent.select_action(state, decision, epsilon=0.0)
            result = env.step_decision(action)
            shaped = compute_shaped_reward(
                env.accumulator,
                env.accumulator.elapsed_time,
                bool(result.info.get("chosen_is_defer", False)),
            )
            total_reward += shaped
            steps += 1
            decision = None if result.done else result.next_decision
        summary = summarize_env(env)
        summary.update({"eval_total_reward": total_reward, "eval_steps": steps})
        return summary
    finally:
        env.close_episode()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DQN training with shaped reward and LR schedule."
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--imitation-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument(
        "--warmup-episodes", type=int, default=2,
        help="Run this many episodes with random actions before any training."
             " Fills the replay buffer with diverse experience first."
             " Keep low (2-3) because random actions make episodes run much"
             " longer than normal — passengers wait longer to be served."
    )
    parser.add_argument(
        "--train-every", type=int, default=4,
        help="Perform one gradient update every N decisions."
             " Reduces overfitting to the most recent experience."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95,
        help="Discount factor. Use ~0.95 rather than 0.99 because consecutive"
             " decisions are ~100 simulation seconds apart — future rewards"
             " need to stay relevant at that timescale."
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-min", type=float, default=1e-5,
        help="Minimum LR for cosine annealing schedule."
    )
    parser.add_argument(
        "--tau", type=float, default=0.005,
        help="Soft target update rate. Lower = more stable target Q-values."
    )
    parser.add_argument("--epsilon-start", type=float, default=0.15)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-length", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fix: setup_logger so _log() calls inside dispatcher/env actually appear
    # on the console. Without this, the "drt_dispatcher" logger has no handlers
    # and all output is silently dropped.
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(str(output_dir / f"dqn_train_{ts}.log"))

    imitation_dir = Path(args.imitation_model_dir)
    metadata = json.loads(
        (imitation_dir / "model_metadata.json").read_text(encoding="utf-8")
    )
    scaler = joblib.load(imitation_dir / "feature_scaler.joblib")
    feature_columns = list(metadata["feature_columns"])
    hidden_dims = list(metadata.get("hidden_dims", [256, 128]))
    dropout = float(metadata.get("dropout", 0.1))
    input_dim = int(metadata.get("input_dim", len(feature_columns)))

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    agent = DQNAgent(
        feature_columns=feature_columns,
        scaler=scaler,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        device=device,
        gamma=args.gamma,
        lr=args.lr,
        lr_min=args.lr_min,
        tau=args.tau,
    )
    warm_state = torch.load(imitation_dir / "imitation_model.pt", map_location=device)
    agent.load_warm_start(warm_state)

    # Estimate total gradient steps to configure LR scheduler.
    # ~200 decisions per episode is a reasonable estimate for this scenario.
    est_decisions_per_ep = 200
    training_episodes = max(1, args.episodes - args.warmup_episodes)
    total_train_steps = (training_episodes * est_decisions_per_ep) // args.train_every
    agent.init_scheduler(total_train_steps)

    print(f"Device: {device}")
    print(f"Warmup: {args.warmup_episodes} episodes (random actions, no training)")
    print(f"Training: {training_episodes} episodes")
    print(f"Gradient update: every {args.train_every} decisions")
    print(f"LR schedule: {args.lr} -> {args.lr_min} over {total_train_steps} steps")
    print(f"Gamma: {args.gamma}  Tau: {args.tau}")

    replay = ReplayBuffer(capacity=args.replay_size)
    history: list[dict] = []
    best_eval_reward = -float("inf")
    global_decision_count = 0

    for episode in trange(1, args.episodes + 1, desc="DQN episodes"):
        is_warmup = episode <= args.warmup_episodes

        # Epsilon only decays during training episodes
        training_progress = max(
            0.0,
            (episode - args.warmup_episodes) / max(1, training_episodes - 1),
        )
        epsilon = args.epsilon_start + (
            args.epsilon_end - args.epsilon_start
        ) * training_progress
        epsilon = float(np.clip(epsilon, args.epsilon_end, args.epsilon_start))

        env = DQNStepEnvironment(
            cfg_path=args.cfg,
            step_length=args.step_length,
            use_gui=args.gui,
            policy=None,
            dataset_logger=None,
            verbose=False,
        )
        total_reward = 0.0
        losses: list[float] = []
        steps = 0

        try:
            if is_warmup:
                print(f"  [WARMUP {episode}/{args.warmup_episodes}] Starting episode...", flush=True)
            decision = env.reset_episode()
            while decision is not None:
                state = agent.decision_to_matrix(decision, env.taxi_plans)

                # During warmup use high epsilon (not pure 1.0) so the agent
                # still picks reasonable actions occasionally — pure random
                # makes passengers wait much longer and slows the simulation.
                act_epsilon = 0.8 if is_warmup else epsilon
                action_idx = agent.select_action(state, decision, epsilon=act_epsilon)

                result = env.step_decision(action_idx)
                global_decision_count += 1

                if is_warmup and steps > 0 and steps % 50 == 0:
                    print(
                        f"  [WARMUP {episode}/{args.warmup_episodes}]"
                        f"  decisions={steps}  buffer={len(replay)}",
                        flush=True,
                    )

                shaped_r = compute_shaped_reward(
                    env.accumulator,
                    env.accumulator.elapsed_time,
                    bool(result.info.get("chosen_is_defer", False)),
                )

                next_state = (
                    None
                    if result.done or result.next_decision is None
                    else agent.decision_to_matrix(result.next_decision, env.taxi_plans)
                )

                replay.add(
                    Transition(
                        state=state,
                        action_index=action_idx,
                        reward=shaped_r,
                        next_state=next_state,
                        done=bool(result.done),
                    )
                )

                total_reward += shaped_r
                steps += 1

                # Gradient update: only after warmup, only every N decisions,
                # only when buffer has enough samples
                if (
                    not is_warmup
                    and len(replay) >= args.batch_size
                    and global_decision_count % args.train_every == 0
                ):
                    losses.append(agent.train_step(replay, args.batch_size))

                decision = None if result.done else result.next_decision

            summary = summarize_env(env)
            normalised_r = total_reward / max(1, steps)

            stats = EpisodeStats(
                episode=episode,
                total_reward=float(total_reward),
                normalised_reward=float(normalised_r),
                steps=steps,
                mean_loss=float(np.mean(losses)) if losses else float("nan"),
                completed_requests=int(summary["completed_requests"]),
                picked_up_requests=int(summary["picked_up_requests"]),
                avg_wait_until_pickup=float(summary["avg_wait_until_pickup"]),
                avg_excess_ride_time=float(summary["avg_excess_ride_time"]),
                epsilon=float(epsilon),
                lr=agent.current_lr(),
            )
            history.append(stats.__dict__)

            if is_warmup:
                print(
                    f"  [WARMUP {episode}/{args.warmup_episodes}] DONE"
                    f"  buffer={len(replay)}"
                    f"  decisions={steps}"
                    f"  avg_wait={summary['avg_wait_until_pickup']:.0f}s"
                    f"  completed={summary['completed_requests']:.0f}",
                    flush=True,
                )

        finally:
            env.close_episode()

        # Evaluation
        if not is_warmup and (
            episode % args.eval_every == 0 or episode == args.episodes
        ):
            eval_summary = evaluate_policy(args.cfg, args.step_length, False, agent)
            row = history[-1]
            for k, v in eval_summary.items():
                row[f"eval_{k}"] = v
            eval_r = eval_summary["eval_total_reward"]
            print(
                f"\n  [EVAL ep={episode}]  reward={eval_r:.3f}"
                f"  wait={eval_summary.get('avg_wait_until_pickup', 0):.1f}s"
                f"  completed={eval_summary.get('completed_requests', 0):.0f}"
                f"  lr={agent.current_lr():.2e}"
            )
            if eval_r > best_eval_reward:
                best_eval_reward = eval_r
                torch.save(
                    agent.online_net.state_dict(), output_dir / "dqn_model.pt"
                )
                print(f"  → New best model saved (eval reward={best_eval_reward:.3f})")

        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    # Fallback save if no eval checkpoint was ever written
    if not (output_dir / "dqn_model.pt").exists():
        torch.save(agent.online_net.state_dict(), output_dir / "dqn_model.pt")

    joblib.dump(scaler, output_dir / "feature_scaler.joblib")
    dqn_metadata = {
        "feature_columns": feature_columns,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "input_dim": input_dim,
        "warm_start_from": str(imitation_dir),
        "episodes": args.episodes,
        "warmup_episodes": args.warmup_episodes,
        "train_every": args.train_every,
        "gamma": args.gamma,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "tau": args.tau,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
    }
    (output_dir / "dqn_metadata.json").write_text(
        json.dumps(dqn_metadata, indent=2), encoding="utf-8"
    )
    print(f"\nSaved DQN artifacts to {output_dir}")
    print(f"Best eval reward: {best_eval_reward:.3f}")


if __name__ == "__main__":
    main()