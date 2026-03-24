from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


METADATA_COLUMNS = {
    "decision_id",
    "policy_name",
    "request_id",
    "person_id",
    "candidate_taxi_id",
    "chosen",
    "rank",
    "heuristic_score",
    # legacy duplicated numeric columns that should never be model features
    "candidate_pickup_index",
    "candidate_dropoff_index",
    "candidate_is_defer",
}


@dataclass
class GroupedBatch:
    x: torch.Tensor
    chosen_index: torch.Tensor
    teacher_scores: torch.Tensor
    valid_mask: torch.Tensor


class DecisionGroupedDataset(Dataset):
    def __init__(self, grouped: list[tuple[np.ndarray, int, np.ndarray]]):
        self.grouped = grouped

    def __len__(self) -> int:
        return len(self.grouped)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, np.ndarray]:
        return self.grouped[idx]


@dataclass
class PreparedData:
    train_groups: list[tuple[np.ndarray, int, np.ndarray]]
    val_groups: list[tuple[np.ndarray, int, np.ndarray]]
    test_groups: list[tuple[np.ndarray, int, np.ndarray]]
    scaler: StandardScaler
    feature_columns: list[str]
    raw_dataframe: pd.DataFrame


@dataclass
class EvalResult:
    loss: float
    choice_accuracy: float
    top1_match_rate: float
    mean_teacher_rank_of_pred: float


class CandidateScorerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f = x.shape
        logits = self.net(x.reshape(b * c, f)).reshape(b, c)
        return logits


class ImitationRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.1):
        super().__init__()
        self.scorer = CandidateScorerMLP(input_dim, hidden_dims, dropout)

    def forward(self, x_with_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_with_mask[..., :-1]
        mask = x_with_mask[..., -1]
        logits = self.scorer(x)
        logits = logits.masked_fill(mask <= 0.0, -1e9)
        return logits, mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"decision_id", "chosen", "heuristic_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in METADATA_COLUMNS]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after excluding metadata columns.")
    return feature_cols


def validate_one_positive_per_decision(df: pd.DataFrame) -> None:
    chosen_count = df.groupby("decision_id")["chosen"].sum()
    bad = chosen_count[chosen_count != 1]
    if not bad.empty:
        preview = bad.head(10).to_dict()
        raise ValueError(
            "Each decision_id must have exactly one chosen=1 row. "
            f"Bad examples: {preview}"
        )


def split_decision_ids(decision_ids: list[str], val_ratio: float, test_ratio: float, seed: int):
    train_ids, temp_ids = train_test_split(decision_ids, test_size=val_ratio + test_ratio, random_state=seed)
    if len(temp_ids) == 0:
        return train_ids, [], []
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(temp_ids, test_size=relative_test, random_state=seed)
    return train_ids, val_ids, test_ids


def fit_scaler(train_df: pd.DataFrame, feature_columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns].to_numpy(dtype=np.float32))
    return scaler


def sanitize_teacher_scores(raw_scores: np.ndarray) -> np.ndarray:
    scores = raw_scores.astype(np.float32).copy()
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return np.zeros_like(scores, dtype=np.float32)

    finite_vals = scores[finite_mask]
    huge_negative = scores < -1e8
    if huge_negative.any():
        min_finite = float(np.min(finite_vals))
        max_finite = float(np.max(finite_vals))
        gap = max(20.0, (max_finite - min_finite) + 5.0)
        scores[huge_negative] = min_finite - gap

    scores = np.clip(scores, -1e6, 1e6)
    return scores.astype(np.float32)


def build_grouped_examples(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
) -> list[tuple[np.ndarray, int, np.ndarray]]:
    groups: list[tuple[np.ndarray, int, np.ndarray]] = []
    for decision_id, grp in df.groupby("decision_id", sort=False):
        grp = grp.reset_index(drop=True)
        feats = scaler.transform(grp[feature_columns].to_numpy(dtype=np.float32)).astype(np.float32)
        chosen_idxs = np.where(grp["chosen"].to_numpy(dtype=np.int64) == 1)[0]
        if len(chosen_idxs) != 1:
            raise ValueError(f"Decision {decision_id} does not have exactly one chosen candidate.")
        teacher_scores = sanitize_teacher_scores(grp["heuristic_score"].to_numpy(dtype=np.float32))
        groups.append((feats, int(chosen_idxs[0]), teacher_scores))
    return groups


def prepare_data(csv_path: str | Path, val_ratio: float, test_ratio: float, seed: int) -> PreparedData:
    df = load_dataset(csv_path)
    validate_one_positive_per_decision(df)
    feature_columns = infer_feature_columns(df)
    decision_ids = df["decision_id"].astype(str).drop_duplicates().tolist()
    train_ids, val_ids, test_ids = split_decision_ids(decision_ids, val_ratio, test_ratio, seed)

    train_df = df[df["decision_id"].astype(str).isin(train_ids)].copy()
    val_df = df[df["decision_id"].astype(str).isin(val_ids)].copy()
    test_df = df[df["decision_id"].astype(str).isin(test_ids)].copy()

    scaler = fit_scaler(train_df, feature_columns)

    train_groups = build_grouped_examples(train_df, feature_columns, scaler)
    val_groups = build_grouped_examples(val_df, feature_columns, scaler) if len(val_df) else []
    test_groups = build_grouped_examples(test_df, feature_columns, scaler) if len(test_df) else []

    return PreparedData(
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        scaler=scaler,
        feature_columns=feature_columns,
        raw_dataframe=df,
    )


def collate_grouped_batch(batch: Sequence[tuple[np.ndarray, int, np.ndarray]]) -> GroupedBatch:
    max_cands = max(features.shape[0] for features, _, _ in batch)
    feat_dim = batch[0][0].shape[1]
    x = np.zeros((len(batch), max_cands, feat_dim), dtype=np.float32)
    mask = np.zeros((len(batch), max_cands), dtype=np.float32)
    chosen = np.zeros((len(batch),), dtype=np.int64)
    teacher_scores = np.full((len(batch), max_cands), -1e9, dtype=np.float32)

    for i, (features, chosen_idx, scores) in enumerate(batch):
        n = features.shape[0]
        x[i, :n, :] = features
        mask[i, :n] = 1.0
        chosen[i] = chosen_idx
        teacher_scores[i, :n] = scores

    return GroupedBatch(
        x=torch.from_numpy(np.concatenate([x, mask[..., None]], axis=2)),
        chosen_index=torch.from_numpy(chosen),
        teacher_scores=torch.from_numpy(teacher_scores),
        valid_mask=torch.from_numpy(mask),
    )


def make_loader(groups: list[tuple[np.ndarray, int, np.ndarray]], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        DecisionGroupedDataset(groups),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_grouped_batch,
    )


def masked_teacher_distribution(teacher_scores: torch.Tensor, valid_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    masked_scores = teacher_scores.masked_fill(valid_mask <= 0.0, -1e9)
    return torch.softmax(masked_scores / temperature, dim=1)


def compute_losses(
    logits: torch.Tensor,
    chosen_index: torch.Tensor,
    teacher_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    teacher_temperature: float,
    lambda_ce: float,
    lambda_kl: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce_loss = F.cross_entropy(logits, chosen_index)
    teacher_prob = masked_teacher_distribution(teacher_scores, valid_mask, teacher_temperature)
    student_log_prob = F.log_softmax(logits, dim=1)
    kl_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
    total_loss = (lambda_ce * ce_loss) + (lambda_kl * kl_loss)
    return total_loss, {
        "ce_loss": float(ce_loss.detach().cpu().item()),
        "kl_loss": float(kl_loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
    }


@torch.no_grad()
def evaluate(
    model: ImitationRanker,
    loader: DataLoader,
    device: torch.device,
    teacher_temperature: float,
    lambda_ce: float,
    lambda_kl: float,
) -> EvalResult:
    model.eval()
    losses: list[float] = []
    all_pred: list[int] = []
    all_true: list[int] = []
    teacher_ranks: list[float] = []

    for batch in loader:
        x = batch.x.to(device)
        y = batch.chosen_index.to(device)
        teacher_scores = batch.teacher_scores.to(device)
        valid_mask = batch.valid_mask.to(device)

        logits, _mask = model(x)
        loss, _parts = compute_losses(
            logits=logits,
            chosen_index=y,
            teacher_scores=teacher_scores,
            valid_mask=valid_mask,
            teacher_temperature=teacher_temperature,
            lambda_ce=lambda_ce,
            lambda_kl=lambda_kl,
        )
        losses.append(float(loss.item()))

        pred = logits.argmax(dim=1)
        all_pred.extend(pred.cpu().numpy().tolist())
        all_true.extend(y.cpu().numpy().tolist())

        teacher_np = teacher_scores.cpu().numpy()
        pred_np = pred.cpu().numpy()
        mask_np = valid_mask.cpu().numpy()
        for i in range(teacher_np.shape[0]):
            valid_idx = np.where(mask_np[i] > 0.0)[0]
            valid_teacher = teacher_np[i, valid_idx]
            order = np.argsort(-valid_teacher)
            teacher_rank_map = {int(valid_idx[pos]): int(rank + 1) for rank, pos in enumerate(order)}
            teacher_ranks.append(float(teacher_rank_map[int(pred_np[i])]))

    acc = accuracy_score(all_true, all_pred) if all_true else float("nan")
    top1 = acc
    mean_teacher_rank = float(np.mean(teacher_ranks)) if teacher_ranks else float("nan")
    return EvalResult(
        loss=float(np.mean(losses)) if losses else float("nan"),
        choice_accuracy=float(acc),
        top1_match_rate=float(top1),
        mean_teacher_rank_of_pred=mean_teacher_rank,
    )


def train_model(
    prepared: PreparedData,
    output_dir: str | Path,
    hidden_dims: Sequence[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    device: torch.device,
    teacher_temperature: float,
    lambda_ce: float,
    lambda_kl: float,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(prepared.train_groups, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(prepared.val_groups, batch_size=batch_size, shuffle=False) if prepared.val_groups else None
    test_loader = make_loader(prepared.test_groups, batch_size=batch_size, shuffle=False) if prepared.test_groups else None

    model = ImitationRanker(
        input_dim=len(prepared.feature_columns),
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    best_state = None
    best_metric = -math.inf
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_total_losses: list[float] = []
        train_ce_losses: list[float] = []
        train_kl_losses: list[float] = []
        train_accs: list[float] = []

        for batch in train_loader:
            x = batch.x.to(device)
            y = batch.chosen_index.to(device)
            teacher_scores = batch.teacher_scores.to(device)
            valid_mask = batch.valid_mask.to(device)

            optimizer.zero_grad()
            logits, _mask = model(x)
            loss, parts = compute_losses(
                logits=logits,
                chosen_index=y,
                teacher_scores=teacher_scores,
                valid_mask=valid_mask,
                teacher_temperature=teacher_temperature,
                lambda_ce=lambda_ce,
                lambda_kl=lambda_kl,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_total_losses.append(parts["total_loss"])
            train_ce_losses.append(parts["ce_loss"])
            train_kl_losses.append(parts["kl_loss"])
            pred = logits.argmax(dim=1)
            train_accs.append(float((pred == y).float().mean().item()))

        train_loss = float(np.mean(train_total_losses)) if train_total_losses else float("nan")
        train_ce = float(np.mean(train_ce_losses)) if train_ce_losses else float("nan")
        train_kl = float(np.mean(train_kl_losses)) if train_kl_losses else float("nan")
        train_acc = float(np.mean(train_accs)) if train_accs else float("nan")

        if val_loader is not None:
            val_result = evaluate(model, val_loader, device, teacher_temperature, lambda_ce, lambda_kl)
            monitor_metric = val_result.choice_accuracy - 0.02 * val_result.mean_teacher_rank_of_pred
        else:
            val_result = EvalResult(loss=float("nan"), choice_accuracy=float("nan"), top1_match_rate=float("nan"), mean_teacher_rank_of_pred=float("nan"))
            monitor_metric = train_acc

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce_loss": train_ce,
            "train_kl_loss": train_kl,
            "train_choice_accuracy": train_acc,
            "val_loss": val_result.loss,
            "val_choice_accuracy": val_result.choice_accuracy,
            "val_mean_teacher_rank_of_pred": val_result.mean_teacher_rank_of_pred,
        })
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} ce={train_ce:.4f} kl={train_kl:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_result.loss:.4f} val_acc={val_result.choice_accuracy:.4f} val_rank={val_result.mean_teacher_rank_of_pred:.3f}"
        )

        if monitor_metric > best_metric:
            best_metric = monitor_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    train_eval = evaluate(model, train_loader, device, teacher_temperature, lambda_ce, lambda_kl)
    val_eval = evaluate(model, val_loader, device, teacher_temperature, lambda_ce, lambda_kl) if val_loader is not None else EvalResult(float("nan"), float("nan"), float("nan"), float("nan"))
    test_eval = evaluate(model, test_loader, device, teacher_temperature, lambda_ce, lambda_kl) if test_loader is not None else EvalResult(float("nan"), float("nan"), float("nan"), float("nan"))

    torch.save(model.state_dict(), output_dir / "imitation_model.pt")
    joblib.dump(prepared.scaler, output_dir / "feature_scaler.joblib")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    metadata = {
        "feature_columns": prepared.feature_columns,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "teacher_temperature": teacher_temperature,
        "lambda_ce": lambda_ce,
        "lambda_kl": lambda_kl,
        "train_decisions": len(prepared.train_groups),
        "val_decisions": len(prepared.val_groups),
        "test_decisions": len(prepared.test_groups),
        "train_result": train_eval.__dict__,
        "val_result": val_eval.__dict__,
        "test_result": test_eval.__dict__,
        "model_class": "ImitationRanker",
        "input_dim": len(prepared.feature_columns),
    }
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nFinal results")
    print(json.dumps(metadata, indent=2))
    return metadata


def parse_hidden_dims(s: str) -> list[int]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("hidden_dims must be a comma-separated list, e.g. 128,64")
    return [int(v) for v in vals]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an imitation-learning ranking model from candidate-level dispatcher data.")
    parser.add_argument("--dataset", required=True, help="Path to imitation_dataset.csv")
    parser.add_argument("--output-dir", default="artifacts/imitation_model", help="Directory to save model artifacts")
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=[256, 128], help="Comma-separated hidden sizes, e.g. 256,128")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-temperature", type=float, default=5.0, help="Softmax temperature used to turn heuristic scores into teacher ranking targets.")
    parser.add_argument("--lambda-ce", type=float, default=1.0, help="Weight for chosen-candidate cross entropy.")
    parser.add_argument("--lambda-kl", type=float, default=0.5, help="Weight for teacher-ranking KL loss.")
    args = parser.parse_args()

    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    set_seed(args.seed)
    prepared = prepare_data(
        csv_path=args.dataset,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_model(
        prepared=prepared,
        output_dir=args.output_dir,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=torch.device(args.device),
        teacher_temperature=args.teacher_temperature,
        lambda_ce=args.lambda_ce,
        lambda_kl=args.lambda_kl,
    )


if __name__ == "__main__":
    main()
