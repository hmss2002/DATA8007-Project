from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data8007_rl.config import method_label
from data8007_rl.utils import ensure_dir, save_json


def _read_csvs(paths: List[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_results(log_dir: str, output_dir: str) -> Dict:
    log_path = Path(log_dir)
    figure_dir = ensure_dir(output_dir)
    result_dir = ensure_dir(str(Path(output_dir).parent))
    train_df = _read_csvs(sorted(log_path.glob("train_*.csv")))
    eval_df = _read_csvs(sorted(log_path.glob("eval_*.csv")))
    if train_df.empty:
        raise FileNotFoundError(f"No training logs found in {log_dir}")

    plt.figure(figsize=(8, 5))
    for method, method_df in train_df.groupby("method"):
        grouped = method_df.groupby("episode")["moving_avg_50"].agg(["mean", "std"]).reset_index()
        label = method_label(str(method))
        plt.plot(grouped["episode"], grouped["mean"], label=label)
        lower = grouped["mean"] - grouped["std"].fillna(0.0)
        upper = grouped["mean"] + grouped["std"].fillna(0.0)
        plt.fill_between(grouped["episode"], lower, upper, alpha=0.18)
    plt.axhline(475, color="black", linestyle="--", linewidth=1, label="Solved threshold")
    plt.xlabel("Episode")
    plt.ylabel("50-episode moving average return")
    plt.title("CartPole-v1 learning curves")
    plt.legend()
    plt.tight_layout()
    learning_curve_path = figure_dir / "learning_curves.png"
    plt.savefig(learning_curve_path, dpi=200)
    plt.close()

    if not eval_df.empty:
        plt.figure(figsize=(8, 5))
        for method, method_df in eval_df.groupby("method"):
            grouped = method_df.groupby("episode")["mean_return"].agg(["mean", "std"]).reset_index()
            label = method_label(str(method))
            plt.plot(grouped["episode"], grouped["mean"], marker="o", label=label)
            lower = grouped["mean"] - grouped["std"].fillna(0.0)
            upper = grouped["mean"] + grouped["std"].fillna(0.0)
            plt.fill_between(grouped["episode"], lower, upper, alpha=0.18)
        plt.xlabel("Training episode")
        plt.ylabel("Greedy evaluation return")
        plt.title("Evaluation during training")
        plt.legend()
        plt.tight_layout()
        eval_curve_path = figure_dir / "evaluation_curves.png"
        plt.savefig(eval_curve_path, dpi=200)
        plt.close()
    else:
        eval_curve_path = None

    final_rows = train_df.sort_values("episode").groupby(["method", "seed"]).tail(1)
    summary = final_rows.groupby("method").agg(
        seeds=("seed", "count"),
        final_return_mean=("return", "mean"),
        final_return_std=("return", "std"),
        final_moving_avg_50_mean=("moving_avg_50", "mean"),
        final_moving_avg_50_std=("moving_avg_50", "std"),
    ).reset_index()
    if not eval_df.empty:
        best_eval = eval_df.groupby(["method", "seed"])["mean_return"].max().reset_index()
        best_summary = best_eval.groupby("method")["mean_return"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "best_eval_mean", "std": "best_eval_std"})
        summary = summary.merge(best_summary, on="method", how="left")
    summary_path = result_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    payload = {
        "learning_curve": str(learning_curve_path),
        "evaluation_curve": str(eval_curve_path) if eval_curve_path is not None else None,
        "summary_csv": str(summary_path),
        "methods": sorted(str(method) for method in train_df["method"].unique()),
    }
    save_json(str(result_dir / "plot_manifest.json"), payload)
    return payload
