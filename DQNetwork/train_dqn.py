from __future__ import annotations

import argparse
import copy
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
    steps: int
    mean_loss: float
    completed_requests: int
    picked_up_requests: int
    avg_wait_until_pickup: float
    avg_excess_ride_time: float
    epsilon: float


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
        tau: float,
        forbid_defer_when_action_exists: bool = True,
    ):
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.forbid_defer_when_action_exists = forbid_defer_when_action_exists

        self.online_net = ParametricQNetwork(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
        self.target_net = ParametricQNetwork(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

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

    def select_action(self, state_matrix: np.ndarray, decision_point, epsilon: float) -> int:
        with torch.no_grad():
            inp = torch.from_numpy(state_matrix[None, :, :]).to(self.device)
            q_vals, _ = self.online_net(inp)
            scores = q_vals[0].detach().cpu().numpy().astype(float)

        valid_indices = list(range(len(scores)))
        if self.forbid_defer_when_action_exists and any(not c.is_defer for c in decision_point.candidate_actions):
            valid_indices = [i for i, c in enumerate(decision_point.candidate_actions) if not c.is_defer]
            for i, c in enumerate(decision_point.candidate_actions):
                if c.is_defer:
                    scores[i] = -1e9

        if random.random() < epsilon:
            return random.choice(valid_indices)
        return int(np.argmax(scores))

    def train_step(self, replay: ReplayBuffer, batch_size: int) -> float:
        batch = replay.sample(batch_size, self.device)
        q_values, _ = self.online_net(batch.states)
        q_sa = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
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
        self.soft_update()
        return float(loss.item())

    def soft_update(self) -> None:
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + online_param.data * self.tau)


def summarize_env(env: DQNStepEnvironment) -> dict[str, float]:
    requests = list(env.requests.values())
    completed = [r for r in requests if r.status.name == "COMPLETED"]
    picked_up = [r for r in requests if r.pickup_time is not None]
    dropped = [r for r in completed if r.dropoff_time is not None and r.pickup_time is not None]
    avg_wait = sum((r.pickup_time - r.request_time) for r in picked_up) / len(picked_up) if picked_up else 0.0
    avg_excess = sum((r.excess_ride_time or 0.0) for r in dropped) / len(dropped) if dropped else 0.0
    return {
        "completed_requests": float(len(completed)),
        "picked_up_requests": float(len(picked_up)),
        "avg_wait_until_pickup": float(avg_wait),
        "avg_excess_ride_time": float(avg_excess),
    }


def evaluate_policy(cfg: str, step_length: float, use_gui: bool, agent: DQNAgent) -> dict[str, float]:
    env = DQNStepEnvironment(cfg_path=cfg, step_length=step_length, use_gui=use_gui, policy=None, dataset_logger=None, verbose=False)
    total_reward = 0.0
    steps = 0
    try:
        decision = env.reset_episode()
        while decision is not None:
            state = agent.decision_to_matrix(decision, env.taxi_plans)
            action = agent.select_action(state, decision, epsilon=0.0)
            result = env.step_decision(action)
            total_reward += result.reward
            steps += 1
            decision = None if result.done else result.next_decision
        summary = summarize_env(env)
        summary.update({"eval_total_reward": total_reward, "eval_steps": steps})
        return summary
    finally:
        env.close_episode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start DQN training for the single-decision DRT environment.")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--imitation-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--warmup-transitions", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-length", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imitation_dir = Path(args.imitation_model_dir)
    metadata = json.loads((imitation_dir / "model_metadata.json").read_text(encoding="utf-8"))
    scaler = joblib.load(imitation_dir / "feature_scaler.joblib")
    feature_columns = list(metadata["feature_columns"])
    hidden_dims = list(metadata.get("hidden_dims", [256, 128]))
    dropout = float(metadata.get("dropout", 0.1))
    input_dim = int(metadata.get("input_dim", len(feature_columns)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    agent = DQNAgent(
        feature_columns=feature_columns,
        scaler=scaler,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        device=device,
        gamma=args.gamma,
        lr=args.lr,
        tau=args.tau,
    )
    warm_state = torch.load(imitation_dir / "imitation_model.pt", map_location=device)
    agent.load_warm_start(warm_state)

    replay = ReplayBuffer(capacity=args.replay_size)
    history: list[dict[str, float]] = []
    best_eval_reward = -float("inf")

    for episode in trange(1, args.episodes + 1, desc="DQN episodes"):
        epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * max(0.0, (args.episodes - episode) / max(1, args.episodes - 1))
        env = DQNStepEnvironment(cfg_path=args.cfg, step_length=args.step_length, use_gui=args.gui, policy=None, dataset_logger=None, verbose=False)
        total_reward = 0.0
        losses: list[float] = []
        steps = 0
        try:
            decision = env.reset_episode()
            while decision is not None:
                state = agent.decision_to_matrix(decision, env.taxi_plans)
                action_idx = agent.select_action(state, decision, epsilon=epsilon)
                result = env.step_decision(action_idx)
                next_state = None if result.done or result.next_decision is None else agent.decision_to_matrix(result.next_decision, env.taxi_plans)
                replay.add(Transition(state=state, action_index=action_idx, reward=float(result.reward), next_state=next_state, done=bool(result.done)))
                total_reward += result.reward
                steps += 1
                if len(replay) >= max(args.batch_size, args.warmup_transitions):
                    losses.append(agent.train_step(replay, args.batch_size))
                decision = None if result.done else result.next_decision

            summary = summarize_env(env)
            stats = EpisodeStats(
                episode=episode,
                total_reward=float(total_reward),
                steps=steps,
                mean_loss=float(np.mean(losses)) if losses else 0.0,
                completed_requests=int(summary["completed_requests"]),
                picked_up_requests=int(summary["picked_up_requests"]),
                avg_wait_until_pickup=float(summary["avg_wait_until_pickup"]),
                avg_excess_ride_time=float(summary["avg_excess_ride_time"]),
                epsilon=float(epsilon),
            )
            history.append(stats.__dict__)
        finally:
            env.close_episode()

        if episode % args.eval_every == 0 or episode == args.episodes:
            eval_summary = evaluate_policy(args.cfg, args.step_length, False, agent)
            row = history[-1]
            for k, v in eval_summary.items():
                row[f"eval_{k}"] = v
            if eval_summary["eval_total_reward"] > best_eval_reward:
                best_eval_reward = eval_summary["eval_total_reward"]
                torch.save(agent.online_net.state_dict(), output_dir / "dqn_model.pt")
                print(f"Saved new best model at episode {episode} (eval reward={best_eval_reward:.3f})")

        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

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
        "gamma": args.gamma,
        "lr": args.lr,
        "tau": args.tau,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
    }
    (output_dir / "dqn_metadata.json").write_text(json.dumps(dqn_metadata, indent=2), encoding="utf-8")
    print(f"Saved DQN artifacts to {output_dir}")


if __name__ == "__main__":
    main()
