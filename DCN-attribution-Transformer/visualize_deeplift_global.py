"""Visualize study-area DeepLIFT factor attributions."""


import argparse
import json
import os
import sys

import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import ScalarMappable
from tqdm import tqdm


from attribution_guided_model import create_attribution_guided_model


def beeswarm_y(x_vals, row_height=0.50, n_bins=120):
    """Return deterministic vertical offsets for a compact beeswarm plot."""
    count = len(x_vals)
    offsets = np.zeros(count)
    if count == 0 or x_vals.max() == x_vals.min():
        return offsets
    edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    bins = np.clip(np.digitize(x_vals, edges, right=True), 0, n_bins - 1)
    generator = np.random.default_rng(seed=0)
    for bin_index in np.unique(bins):
        indices = np.where(bins == bin_index)[0]
        if len(indices) == 1:
            continue
        values = np.linspace(-row_height / 2, row_height / 2, len(indices))
        generator.shuffle(values)
        offsets[indices] = values
    return offsets


MODEL_PATH = None
FULL_AREA_DATA = None
TRAIN_DATA = None
TRAIN_LABELS = None
TRAIN_SCALER = None
PREDICTION_MAP = None
OUTPUT_DIR = None

WINDOW_SIZE = 9
WINDOW_HALF = WINDOW_SIZE // 2
ATTRIBUTION_BATCH = 8192
DISPLAY_SAMPLE_MAX = 10000
TOP_GEOCHEMICAL_FACTORS = 10
N_GEOCHEMICAL_FACTORS = 39

CHANNEL_NAMES = [
    "Ag",
    "Al₂O₃",
    "As",
    "Au",
    "B",
    "Ba",
    "Be",
    "Bi",
    "CaO",
    "Cd",
    "Co",
    "Cr",
    "Cu",
    "F",
    "Fe₂O₃",
    "Hg",
    "K₂O",
    "La",
    "Li",
    "MgO",
    "Mn",
    "Mo",
    "Na₂O",
    "Nb",
    "Ni",
    "P",
    "Pb",
    "Sb",
    "SiO₂",
    "Sn",
    "Sr",
    "Th",
    "Ti",
    "U",
    "V",
    "W",
    "Y",
    "Zn",
    "Zr",
    "Strata",
    "Faults",
    "Granite",
]
DISPLAY_NAMES = {
    "Al₂O₃": "Al$_2$O$_3$",
    "Fe₂O₃": "Fe$_2$O$_3$",
    "K₂O": "K$_2$O",
    "Na₂O": "Na$_2$O",
    "SiO₂": "SiO$_2$",
}

plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def window_channel_means(data, window_size):

    if data.ndim != 3:
        raise ValueError(f"全区数据应为 (H,W,C)，实际为 {data.shape}")
    integral = np.pad(
        np.cumsum(
            np.cumsum(data, axis=0, dtype=np.float64),
            axis=1,
            dtype=np.float64,
        ),
        ((1, 0), (1, 0), (0, 0)),
        mode="constant",
    )
    sums = (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    )
    return (sums / float(window_size * window_size)).astype(np.float32)


def load_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    config = checkpoint.get("model_config", {})
    regularization = checkpoint.get("regularization", {})
    if config.get("factor_branch_type") != "linear":
        raise RuntimeError("研究区精确因子 DeepLIFT 要求 linear 因子分支")
    if not regularization.get("factor_end_to_end", False):
        raise RuntimeError("检查点未标记为端到端因子归因模型")
    if regularization.get("factor_prefit_freeze", True):
        raise RuntimeError("检查点的因子分支仍被标记为冻结")

    model = create_attribution_guided_model(
        num_classes=int(config.get("num_classes", 2)),
        feature_dim=int(config.get("feature_dim", 32)),
        transformer_depth=int(config.get("transformer_depth", 1)),
        transformer_heads=int(config.get("transformer_heads", 2)),
        factor_pooling=config.get("factor_pooling", "mean"),
        factor_branch_type="linear",
        dropout=float(config.get("dropout", 0.30)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def set_training_reference(model, scaler, device):

    train_raw = np.load(TRAIN_DATA).astype(np.float32)
    train_labels = np.asarray(np.load(TRAIN_LABELS)).reshape(-1)
    if train_raw.ndim != 4 or train_raw.shape[-1] != len(CHANNEL_NAMES):
        raise ValueError(f"训练数据形状错误: {train_raw.shape}")
    negative = train_labels == 0
    if not np.any(negative):
        raise RuntimeError("训练数据中没有非矿化样本，无法构建参考基线")

    n_channels = train_raw.shape[-1]
    train_norm = scaler.transform(train_raw.reshape(-1, n_channels)).reshape(
        train_raw.shape
    )
    reference_factor = train_norm[negative].mean(axis=(0, 1, 2)).astype(np.float32)
    model.attribution_network.factor_branch.set_reference(
        torch.from_numpy(reference_factor).to(device)
    )
    return reference_factor, int(negative.sum())


def extract_study_area_factor_inputs(scaler):

    data = np.load(FULL_AREA_DATA).astype(np.float32)
    if data.ndim != 3 or data.shape[-1] != len(CHANNEL_NAMES):
        raise ValueError(f"全区数据形状错误: {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("全区数据包含 NaN 或无穷值，无法复用训练标准化器")

    height, width, n_channels = data.shape
    valid_center = ~np.all(data == 0, axis=2)
    valid_window_mask = valid_center[
        WINDOW_HALF : height - WINDOW_HALF,
        WINDOW_HALF : width - WINDOW_HALF,
    ]

    raw_mean_grid = window_channel_means(data, WINDOW_SIZE)
    if raw_mean_grid.shape[:2] != valid_window_mask.shape:
        raise RuntimeError("滑窗均值网格与有效中心掩膜形状不一致")
    top_rows, top_cols = np.nonzero(valid_window_mask)
    factor_raw = raw_mean_grid[valid_window_mask]
    factor_norm = scaler.transform(factor_raw.reshape(-1, n_channels)).astype(
        np.float32, copy=False
    )
    center_rows = top_rows.astype(np.int32) + WINDOW_HALF
    center_cols = top_cols.astype(np.int32) + WINDOW_HALF
    return factor_norm, factor_raw, center_rows, center_cols, data.shape


def compute_contributions(model, factor_norm, device):

    n_samples = len(factor_norm)
    contributions = np.empty((n_samples, len(CHANNEL_NAMES)), dtype=np.float32)
    max_completeness_error = 0.0
    branch = model.attribution_network.factor_branch
    with torch.no_grad():
        for start in tqdm(
            range(0, n_samples, ATTRIBUTION_BATCH),
            desc="全研究区 DeepLIFT",
            ncols=78,
        ):
            end = min(start + ATTRIBUTION_BATCH, n_samples)
            batch = torch.from_numpy(factor_norm[start:end]).to(device)
            _, batch_contributions, completeness_delta = branch.explain(batch)
            contributions[start:end] = batch_contributions.cpu().numpy()
            max_completeness_error = max(
                max_completeness_error,
                float(completeness_delta.abs().max().item()),
            )
    return contributions, max_completeness_error


def aggregate_rankings(contributions):
    mean_absolute = np.abs(contributions).mean(axis=0)
    mean_positive = np.clip(contributions, 0.0, None).mean(axis=0)
    positive_total = float(mean_positive.sum())
    absolute_total = float(mean_absolute.sum())
    if positive_total <= 0 or absolute_total <= 0:
        raise RuntimeError("全区贡献为0，无法生成排名")
    positive_pct = 100.0 * mean_positive / positive_total
    absolute_pct = 100.0 * mean_absolute / absolute_total

    geochemical = np.arange(N_GEOCHEMICAL_FACTORS)
    controls = np.arange(N_GEOCHEMICAL_FACTORS, len(CHANNEL_NAMES))
    top_geochemical = geochemical[
        np.argsort(-mean_positive[geochemical])[:TOP_GEOCHEMICAL_FACTORS]
    ]
    selected = np.concatenate((top_geochemical, controls))
    selected_desc = selected[np.argsort(-mean_positive[selected])]
    return (
        mean_absolute,
        mean_positive,
        absolute_pct,
        positive_pct,
        top_geochemical,
        controls,
        selected_desc,
    )


def plot_beeswarm(
    contributions,
    factor_raw,
    display_indices,
    selected_desc,
    output_path,
):
    denominator = np.abs(contributions).sum(axis=1, keepdims=True)
    relative_pct = 100.0 * contributions / np.maximum(denominator, 1e-12)
    sampled_relative = relative_pct[display_indices]
    sampled_values = factor_raw[display_indices]
    plot_order = selected_desc[::-1]
    n_features = len(plot_order)
    x_bound = max(
        1.0,
        float(np.percentile(np.abs(relative_pct[:, selected_desc]), 99)) * 1.05,
    )

    fig, ax = plt.subplots(figsize=(9.0, 0.50 * n_features + 2.2), facecolor="white")
    cmap = plt.get_cmap("RdBu_r")
    for feature_row, channel in enumerate(plot_order):
        x_values = np.clip(sampled_relative[:, channel], -x_bound, x_bound).astype(
            float
        )
        raw_values = sampled_values[:, channel].astype(float)
        value_low, value_high = np.percentile(raw_values, [1, 99])
        normalized_color = np.clip(
            (raw_values - value_low) / (value_high - value_low + 1e-9),
            0,
            1,
        )
        y_offset = beeswarm_y(x_values, row_height=0.52)
        ax.scatter(
            x_values,
            feature_row + y_offset,
            c=cmap(normalized_color),
            s=3,
            alpha=0.32,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(
        [DISPLAY_NAMES.get(CHANNEL_NAMES[c], CHANNEL_NAMES[c]) for c in plot_order],
        fontsize=11,
    )
    ax.axvline(0, color="#666666", linewidth=0.9, zorder=1)
    ax.set_xlim(-x_bound, x_bound)
    ax.set_ylim(-0.65, n_features - 0.35)
    ax.set_xlabel("Signed relative DeepLIFT contribution (%)", fontsize=11.5)
    ax.set_title(
        "Study-area-wide DeepLIFT: Top 10 positive geochemical factors "
        "+ ore-controlling factors",
        fontsize=12.5,
        pad=10,
    )
    ax.set_facecolor("#f4f4f4")
    ax.grid(True, axis="x", color="white", linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(n_features):
        ax.axhline(row - 0.5, color="white", linewidth=1.2, zorder=1)

    scalar_map = ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax, fraction=0.028, pad=0.015, aspect=22)
    colorbar.set_ticks([0.02, 0.98])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.ax.set_ylabel("Feature value", fontsize=10)
    colorbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return relative_pct


def main(argv=None):
    global MODEL_PATH, FULL_AREA_DATA, TRAIN_DATA, TRAIN_LABELS
    global TRAIN_SCALER, PREDICTION_MAP, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--study-area-data", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--prediction-map")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    MODEL_PATH = os.path.abspath(args.model)
    FULL_AREA_DATA = os.path.abspath(args.study_area_data)
    TRAIN_DATA = os.path.abspath(args.train_data)
    TRAIN_LABELS = os.path.abspath(args.train_labels)
    TRAIN_SCALER = os.path.abspath(args.scaler)
    PREDICTION_MAP = (
        os.path.abspath(args.prediction_map) if args.prediction_map else None
    )
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    scaler = joblib.load(TRAIN_SCALER)
    model, checkpoint = load_model(device)
    reference_factor, n_negative = set_training_reference(model, scaler, device)
    factor_norm, factor_raw, rows, cols, area_shape = extract_study_area_factor_inputs(
        scaler
    )
    print(f"全区形状: {area_shape}; 有效预测窗口: {len(factor_norm):,}")
    print(f"固定训练基线: {n_negative} 个非矿化训练窗口")

    contributions, max_completeness_error = compute_contributions(
        model, factor_norm, device
    )
    (
        mean_absolute,
        mean_positive,
        absolute_pct,
        positive_pct,
        top_geochemical,
        controls,
        selected_desc,
    ) = aggregate_rankings(contributions)

    n_display = min(DISPLAY_SAMPLE_MAX, len(contributions))
    display_indices = np.unique(
        np.linspace(0, len(contributions) - 1, n_display, dtype=np.int64)
    )
    figure_path = os.path.join(
        OUTPUT_DIR,
        "study_area_deeplift_beeswarm_top10_positive_plus_controls.png",
    )
    relative_pct = plot_beeswarm(
        contributions,
        factor_raw,
        display_indices,
        selected_desc,
        figure_path,
    )

    probability = None
    if PREDICTION_MAP and os.path.isfile(PREDICTION_MAP):
        probability_map = np.load(PREDICTION_MAP)
        if probability_map.shape == area_shape[:2]:
            probability = probability_map[rows, cols].astype(np.float32)

    archive = {
        "center_row": rows,
        "center_col": cols,
        "factor_value_raw": factor_raw,
        "deeplift_contribution": contributions,
        "signed_relative_contribution_pct": relative_pct.astype(np.float32),
        "display_sample_index": display_indices,
        "reference_factor_standardized": reference_factor,
    }
    if probability is not None:
        archive["prediction_probability"] = probability
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "study_area_factor_attributions.npz"),
        **archive,
    )

    top_set = set(top_geochemical.tolist())
    control_set = set(controls.tolist())
    selected_rank = {
        int(channel): rank for rank, channel in enumerate(selected_desc, 1)
    }
    positive_order = np.argsort(-mean_positive)
    positive_rank = {
        int(channel): rank for rank, channel in enumerate(positive_order, 1)
    }
    summary_rows = []
    for channel, name in enumerate(CHANNEL_NAMES):
        summary_rows.append(
            {
                "channel_index": channel,
                "factor": name,
                "study_area_positive_rank_all_42": positive_rank[channel],
                "mean_positive_raw_contribution": mean_positive[channel],
                "global_positive_contribution_pct": positive_pct[channel],
                "mean_absolute_raw_contribution": mean_absolute[channel],
                "global_absolute_importance_pct": absolute_pct[channel],
                "mean_signed_raw_contribution": contributions[:, channel].mean(),
                "median_signed_raw_contribution": np.median(contributions[:, channel]),
                "positive_window_fraction": np.mean(contributions[:, channel] > 0),
                "selected_for_beeswarm": channel in selected_rank,
                "selected_plot_rank": selected_rank.get(channel, np.nan),
                "selection_group": (
                    "geochemical_positive_top10"
                    if channel in top_set
                    else "ore_controlling_factor"
                    if channel in control_set
                    else "not_selected"
                ),
                "n_valid_study_area_windows": len(contributions),
                "n_nonmineralized_training_baseline_windows": n_negative,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "study_area_factor_ranking_all_42.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    summary[summary["selected_for_beeswarm"]].sort_values("selected_plot_rank").to_csv(
        os.path.join(
            OUTPUT_DIR,
            "study_area_selected_top10_positive_plus_controls.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "scope": "all valid 9x9 prediction windows across the study area",
        "aggregation_population": int(len(contributions)),
        "beeswarm_display_sample": int(len(display_indices)),
        "display_sampling": "deterministic systematic sample in raster order",
        "ranking_metric": "mean(max(DeepLIFT contribution, 0))",
        "baseline": "mean of non-mineralized training windows",
        "n_baseline_windows": n_negative,
        "scaler": "training-set StandardScaler reused without refitting",
        "max_completeness_error": max_completeness_error,
        "checkpoint_epoch_zero_based": int(checkpoint.get("epoch", -1)),
        "spatial_dependence_warning": (
            "overlapping windows are not statistically independent"
        ),
    }
    with open(
        os.path.join(OUTPUT_DIR, "study_area_attribution_metadata.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"最大 DeepLIFT 完备性误差: {max_completeness_error:.3e}")
    print(f"Beeswarm显示样本: {len(display_indices):,}/{len(contributions):,}")
    print("正向贡献前10元素/氧化物:")
    for channel in top_geochemical:
        print(f"  {CHANNEL_NAMES[channel]}: {positive_pct[channel]:.4f}%")
    print(f"结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
