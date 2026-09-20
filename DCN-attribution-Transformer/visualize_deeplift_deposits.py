"""Visualize DeepLIFT factor contributions at known deposit locations."""


import argparse
import os
import sys
import importlib.util
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)

COMBINED_NPY = None
TRAIN_DATA = None
TRAIN_LABELS = None
TRAIN_SCALER = None
LABEL_TIF = None
MODEL_PATH = None
CHECKPOINT_MODEL_FILE = None
SAVE_ROOT = None
PRIOR_SAVE_DIR = None
ALL42_SAVE_DIR = None


FEATURE_DIM = 32
TRANSFORMER_DEPTH = 1
TRANSFORMER_HEADS = 2

TARGET_CLASS = 1
WINDOW_HALF = 4


NORM_PERCENTILE = 99
NORM_SCALE = 1.0


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
    18: "Li",
    6: "Be",
    23: "Nb",
    13: "F",
    29: "Sn",
    35: "W",
    16: r"$\mathrm{K_2O}$",
    1: r"$\mathrm{Al_2O_3}$",
    28: r"$\mathrm{SiO_2}$",
    22: r"$\mathrm{Na_2O}$",
    5: "Ba",
    30: "Sr",
    31: "Th",
    33: "U",
    17: "La",
    38: "Zr",
    39: "Strata",
    40: "Faults",
    41: "Granite",
}

TOP_N_ELEMENT = 16
ELEMENT_CHANNELS = list(range(39))
GEO_CHANNELS = [39, 40, 41]


GEOLOGICAL_PRIOR_NAMES = [
    "Li",
    "Be",
    "Nb",
    "Sn",
    "W",
    "F",
    "B",
    "P",
    "K₂O",
    "Na₂O",
    "Al₂O₃",
    "SiO₂",
    "CaO",
    "Fe₂O₃",
    "Ba",
    "Sr",
]
CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNEL_NAMES)}
GEOLOGICAL_PRIOR_CHANNELS = [CHANNEL_INDEX[name] for name in GEOLOGICAL_PRIOR_NAMES]


def display_name(channel):
    return DISPLAY_NAMES.get(channel, CHANNEL_NAMES[channel])


def load_checkpoint_model_factory():

    model_file = CHECKPOINT_MODEL_FILE
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"未找到检查点同期模型定义: {model_file}")

    model_dir = os.path.dirname(model_file)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location(
        "stable_checkpoint_attribution_guided_model", model_file
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法创建模型模块加载器: {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_attribution_guided_model


CHANNEL_COLORS = {
    18: "#B22222",
    6: "#FF6347",
    23: "#FF8C00",
    13: "#FFA500",
    29: "#DAA520",
    35: "#FFD700",
    16: "#4169E1",
    1: "#6495ED",
    28: "#87CEEB",
    22: "#ADD8E6",
    5: "#9370DB",
    30: "#BA55D3",
    31: "#808080",
    33: "#A9A9A9",
    17: "#20B2AA",
    38: "#3CB371",
    39: "#2E8B57",
    40: "#006400",
    41: "#228B22",
}


GROUP_LABELS = {
    "直接成矿元素": [18],
    "伴生/挥发分元素": [6, 23, 13, 29, 35],
    "岩石化学指标": [16, 1, 28, 22],
    "亲石/放射/稀土": [5, 30, 31, 33, 17, 38],
    "控矿要素": [39, 40, 41],
}


def integrated_gradients_single(
    model, inp, baseline, n_steps, target_class, half, spatial_half=0
):

    device = inp.device

    with torch.no_grad():
        out_real = model(inp)
        pred_prob = float(F.softmax(out_real, dim=1)[0, target_class].item())

    alphas = torch.linspace(0.0, 1.0, n_steps, device=device).view(n_steps, 1, 1, 1, 1)
    delta = inp.unsqueeze(0) - baseline.unsqueeze(0)  # (1,1,C,9,9)
    interp_flat = (
        (baseline.unsqueeze(0) + alphas * delta)
        .squeeze(1)
        .detach()
        .requires_grad_(True)
    )  # (n,C,9,9)

    out = model(interp_flat)
    probs = F.softmax(out, dim=1)
    model.zero_grad()
    probs[:, target_class].sum().backward()

    grads = interp_flat.grad.detach()  # (n,C,9,9)

    weights = torch.ones(n_steps, device=device)
    weights[0] = weights[-1] = 0.5
    weights = weights.view(n_steps, 1, 1, 1) / (n_steps - 1)
    avg_grads = (grads * weights).sum(dim=0)  # (C,9,9)

    r0, r1 = half - spatial_half, half + spatial_half + 1
    c0, c1 = half - spatial_half, half + spatial_half + 1

    delta_region = (inp[0] - baseline[0])[:, r0:r1, c0:c1]  # (C,k,k)
    ig_region_map = delta_region * avg_grads[:, r0:r1, c0:c1]  # (C,k,k)
    ig_region = ig_region_map.mean(dim=(1, 2))  # (C,)

    return ig_region.cpu().numpy(), pred_prob


def smoothgrad_ig(
    model,
    inp,
    baseline,
    n_steps,
    target_class,
    half,
    spatial_half=1,
    n_smooth=20,
    noise_std=0.15,
):

    with torch.no_grad():
        out_real = model(inp)
        pred_prob = float(F.softmax(out_real, dim=1)[0, target_class].item())

    C = inp.shape[1]
    ig_accum = np.zeros(C, dtype=np.float64)

    for _ in range(n_smooth):
        noise = torch.randn_like(inp) * noise_std
        inp_noisy = inp + noise
        ig, _ = integrated_gradients_single(
            model,
            inp_noisy,
            baseline,
            n_steps,
            target_class,
            half,
            spatial_half=spatial_half,
        )
        ig_accum += ig.astype(np.float64)

    return (ig_accum / n_smooth).astype(np.float32), pred_prob


def deeplift_factor_single(model, inp, target_class=TARGET_CLASS):

    with torch.no_grad():
        output = model(inp)
        pred_prob = float(F.softmax(output, dim=1)[0, target_class].item())
        contributions, completeness_delta = model.get_factor_attribution(inp)
    return (
        contributions[0].detach().cpu().numpy().astype(np.float32),
        pred_prob,
        float(completeness_delta.abs().max().item()),
    )


def normalize_ig_global(all_ig, norm_percentile=NORM_PERCENTILE, norm_scale=NORM_SCALE):

    concat = np.concatenate([np.abs(ig) for ig in all_ig])
    scale = np.percentile(concat, norm_percentile)
    if scale == 0:
        scale = np.max(concat) if np.max(concat) > 0 else 1.0

    all_ig_norm = [np.clip(ig / scale, -1.0, 1.0) * norm_scale for ig in all_ig]
    return all_ig_norm, float(scale)


def _academic_bar_chart(
    ax,
    x_pos,
    values,
    x_labels,
    bar_color="#2878B5",
    bar_width=0.5,
    ylim_min=0.5,
    label_fontsize=8,
    fixed_ylim=None,
    label_values=None,
    show_value_labels=False,
):

    bars = ax.bar(
        x_pos, values, color=bar_color, edgecolor="none", width=bar_width, zorder=3
    )

    ax.axhline(0, color="black", linewidth=0.8, zorder=4)

    max_abs = max(abs(v) for v in values) if len(values) else 0.01
    y_range = (
        float(fixed_ylim) if fixed_ylim is not None else max(ylim_min, max_abs * 1.2)
    )

    if show_value_labels:
        offset = max_abs * 0.04
        shown_labels = values if label_values is None else label_values
        for bar, val, label_val in zip(bars, values, shown_labels):
            cx = bar.get_x() + bar.get_width() / 2
            if val >= 0:
                text_y = min(val + offset, y_range * 0.96)
                ax.text(
                    cx,
                    text_y,
                    f"{label_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize,
                    color="black",
                )
            else:
                text_y = max(val - offset, -y_range * 0.96)
                ax.text(
                    cx,
                    text_y,
                    f"{label_val:.2f}",
                    ha="center",
                    va="top",
                    fontsize=label_fontsize,
                    color="black",
                )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)

    ax.set_ylim(-y_range, y_range)

    if fixed_ylim is not None and y_range >= 5.0:
        step = 2.0
    elif y_range <= 0.2:
        step = 0.05
    elif y_range <= 0.55:
        step = 0.1
    elif y_range <= 1.1:
        step = 0.2
    else:
        step = 0.5
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")
        spine.set_visible(True)

    ax.tick_params(
        axis="both",
        which="major",
        length=4,
        width=0.8,
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=2,
        width=0.6,
        direction="in",
        top=True,
        right=True,
    )

    ax.set_facecolor("white")
    return y_range


def plot_deposit_histogram(
    signed_strength, pred_prob, deposit_id, deposit_rc, selected_channels, save_path
):

    n = len(selected_channels)
    x_labels = [display_name(k) for k in selected_channels]
    x_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.46), 5.8), facecolor="white")
    _academic_bar_chart(
        ax,
        x_pos,
        signed_strength,
        x_labels,
        bar_color="#2878B5",
        bar_width=0.5,
        ylim_min=1.0,
        fixed_ylim=1.0,
    )
    ax.set_ylabel("Signed local contribution strength (max |C| = 1)", fontsize=10)

    r, c_col = deposit_rc
    ax.set_title(
        f"Deposit #{deposit_id}  (row={r}, col={c_col})  "
        f"P(Li deposit) = {pred_prob:.3f}",
        fontsize=10,
        pad=8,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        f"  [OK] 矿点 #{deposit_id:02d} → {os.path.basename(save_path)}  "
        f"(P={pred_prob:.3f})"
    )


def main(argv=None):
    global COMBINED_NPY, TRAIN_DATA, TRAIN_LABELS, TRAIN_SCALER, LABEL_TIF
    global MODEL_PATH, CHECKPOINT_MODEL_FILE, SAVE_ROOT
    global PRIOR_SAVE_DIR, ALL42_SAVE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-definition", required=True)
    parser.add_argument("--study-area-data", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    MODEL_PATH = os.path.abspath(args.model)
    CHECKPOINT_MODEL_FILE = os.path.abspath(args.model_definition)
    COMBINED_NPY = os.path.abspath(args.study_area_data)
    TRAIN_DATA = os.path.abspath(args.train_data)
    TRAIN_LABELS = os.path.abspath(args.train_labels)
    TRAIN_SCALER = os.path.abspath(args.scaler)
    LABEL_TIF = os.path.abspath(args.labels)
    SAVE_ROOT = os.path.abspath(args.output_dir)
    PRIOR_SAVE_DIR = os.path.join(SAVE_ROOT, "geological_prior")
    ALL42_SAVE_DIR = os.path.join(SAVE_ROOT, "all_42_factors")
    print("=" * 70)
    print("稳定归因模型 DeepLIFT 矿点因子贡献直方图  –  花岗岩型锂矿")
    print("=" * 70)

    os.makedirs(PRIOR_SAVE_DIR, exist_ok=True)
    os.makedirs(ALL42_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if device.type == "cpu":
        print("  [提示] CPU 模式下每个矿点约需几秒，矿点数较多时请耐心等待。")

    print("\n[1/5] 加载模型...")
    create_attribution_guided_model = load_checkpoint_model_factory()
    print(f"  模型定义快照: {CHECKPOINT_MODEL_FILE}")

    ckpt = torch.load(MODEL_PATH, map_location=device)
    saved_config = ckpt.get("model_config", {})
    regularization = ckpt.get("regularization", {})
    model = create_attribution_guided_model(
        num_classes=int(saved_config.get("num_classes", 2)),
        feature_dim=int(saved_config.get("feature_dim", FEATURE_DIM)),
        transformer_depth=int(saved_config.get("transformer_depth", TRANSFORMER_DEPTH)),
        transformer_heads=int(saved_config.get("transformer_heads", TRANSFORMER_HEADS)),
        factor_pooling=saved_config.get("factor_pooling", "mean"),
        factor_branch_type=saved_config.get("factor_branch_type", "mlp"),
        dropout=float(saved_config.get("dropout", regularization.get("dropout", 0.30))),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f'  [OK] 模型加载成功  (val_acc={ckpt.get("val_acc", 0):.2f}%，'
        f'val_f1={ckpt.get("val_f1", 0):.4f})'
    )

    print("\n[2/5] 加载全图数据...")
    combined = np.load(COMBINED_NPY).astype(np.float32)
    H, W, C = combined.shape
    print(f"  Combined_array 形状: {combined.shape}  (H={H}, W={W}, C={C})")

    print("\n[3/5] 数据标准化...")
    train_data = np.load(TRAIN_DATA)  # (N, 9, 9, 42)
    try:
        scaler = joblib.load(TRAIN_SCALER)
        print(f"  [OK] 已加载训练时 StandardScaler: {TRAIN_SCALER}")
    except Exception as exc:

        print(f"  [兼容] scaler 文件无法读取 ({type(exc).__name__})，按训练流程重建")
        scaler = StandardScaler().fit(train_data.reshape(-1, C))
    combined_norm = scaler.transform(combined.reshape(-1, C)).reshape(H, W, C)

    half = WINDOW_HALF
    train_labels = np.asarray(np.load(TRAIN_LABELS)).reshape(-1)
    if len(train_labels) != len(train_data):
        raise ValueError(f"训练数据与标签数量不一致: {len(train_data)} != {len(train_labels)}")
    negative_mask = train_labels == 0
    n_negative = int(negative_mask.sum())
    if n_negative == 0:
        raise ValueError("训练集中没有标签为0的非矿化样本，无法构建DeepLIFT基线")
    train_norm = scaler.transform(train_data.reshape(-1, C)).reshape(train_data.shape)
    train_mean_win = train_norm[negative_mask].mean(axis=0)
    baseline_win = (
        torch.from_numpy(train_mean_win.astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    _bl = train_mean_win[half, half]
    print(f"  [OK] 基准 = 非矿化训练样本均值窗口" f"（{n_negative}/{len(train_data)} 个样本）")
    print(f"       标准化后中心像素举例: " f"Li={_bl[18]:.4f}  Be={_bl[6]:.4f}  W={_bl[35]:.4f}")

    baseline_factor = baseline_win.mean(dim=(2, 3)).squeeze(0)
    model.attribution_network.factor_branch.set_reference(baseline_factor)

    print("\n[4/5] 读取矿点坐标...")
    import cv2

    label_img = cv2.imread(LABEL_TIF, cv2.IMREAD_UNCHANGED)
    if label_img is None:
        raise FileNotFoundError(f"无法读取标签文件: {LABEL_TIF}")

    raw_coords = np.array(list(zip(*np.where(label_img == 1))))
    if len(raw_coords) == 0:
        raise ValueError("标签文件中未找到矿点（像素值=1）。")

    MERGE_DIST = 5
    kept = []
    used = [False] * len(raw_coords)
    for i in range(len(raw_coords)):
        if used[i]:
            continue
        kept.append(i)
        for j in range(i + 1, len(raw_coords)):
            if (
                not used[j]
                and np.linalg.norm(raw_coords[i] - raw_coords[j]) < MERGE_DIST
            ):
                used[j] = True
    deposit_rc_all = raw_coords[kept]

    valid_mask = (
        (deposit_rc_all[:, 0] >= half)
        & (deposit_rc_all[:, 0] < H - half)
        & (deposit_rc_all[:, 1] >= half)
        & (deposit_rc_all[:, 1] < W - half)
    )
    deposit_rc = deposit_rc_all[valid_mask]
    skipped = len(deposit_rc_all) - len(deposit_rc)
    print(
        f"  原始矿点数: {len(raw_coords)}  →  合并后: {len(deposit_rc_all)}  "
        f"→  有效（距边界≥{half}px）: {len(deposit_rc)}"
    )
    if skipped > 0:
        print(f"  [跳过] {skipped} 个矿点距图像边界过近，无法构建完整窗口。")

    print(f"\n[5/5] 逐矿点计算因子级 DeepLIFT（共 {len(deposit_rc)} 个矿点）...")

    all_raw_full = []
    all_relative_full = []
    pred_probs = []
    completeness_errors = []

    print("  算法配置: 模型内置 DeepLIFT 因子分支，基线=非矿化训练窗口均值")

    for i, (r, c_col) in enumerate(tqdm(deposit_rc, desc="DeepLIFT", ncols=72)):
        win = combined_norm[
            r - half : r + half + 1, c_col - half : c_col + half + 1, :
        ]  # (9, 9, C)
        inp = (
            torch.FloatTensor(win)
            .permute(2, 0, 1)  # (C, 9, 9)
            .unsqueeze(0)  # (1, C, 9, 9)
            .to(device)
        )

        ig_full, prob, completeness_error = deeplift_factor_single(model, inp)
        denominator = max(float(np.abs(ig_full).sum()), 1e-12)
        relative_full = 100.0 * ig_full / denominator
        all_raw_full.append(ig_full)
        all_relative_full.append(relative_full)
        pred_probs.append(prob)
        completeness_errors.append(completeness_error)

    print(f"  最大 DeepLIFT 完整性误差: {max(completeness_errors):.3e}")

    all_raw_arr = np.stack(all_raw_full)
    all_relative_arr = np.stack(all_relative_full)
    print("  每个矿点42通道 |相对贡献| 之和 = 100%")

    print(f"\n绘制矿点归因贡献直方图...")
    plot_versions = [
        (
            "Mineralization-related factors",
            GEOLOGICAL_PRIOR_CHANNELS + GEO_CHANNELS,
            PRIOR_SAVE_DIR,
        ),
        ("All 42 factors", list(range(42)), ALL42_SAVE_DIR),
    ]
    for i, (r, c_col) in enumerate(deposit_rc):
        local_max = max(float(np.abs(all_raw_arr[i]).max()), 1e-12)
        for _, version_channels, version_dir in plot_versions:
            selected_channels = sorted(
                version_channels, key=lambda ch: abs(all_raw_arr[i, ch]), reverse=True
            )
            save_path = os.path.join(
                version_dir, f"deposit_{i+1:02d}_r{r}_c{c_col}.png"
            )
            plot_deposit_histogram(
                signed_strength=all_raw_arr[i, selected_channels] / local_max,
                pred_prob=pred_probs[i],
                deposit_id=i + 1,
                deposit_rc=(r, c_col),
                selected_channels=selected_channels,
                save_path=save_path,
            )

    import pandas as pd

    rows = []
    for i, (r, c_col) in enumerate(deposit_rc):
        row = {
            "deposit_id": i + 1,
            "row": int(r),
            "col": int(c_col),
            "pred_prob": round(float(pred_probs[i]), 6),
        }
        local_max = max(float(np.abs(all_raw_arr[i]).max()), 1e-12)
        for k in range(42):
            row[f"raw_deeplift_{CHANNEL_NAMES[k]}"] = round(float(all_raw_arr[i, k]), 8)
            row[f"absolute_deeplift_{CHANNEL_NAMES[k]}"] = round(
                float(abs(all_raw_arr[i, k])), 8
            )
            row[f"relative_pct_{CHANNEL_NAMES[k]}"] = round(
                float(all_relative_arr[i, k]), 6
            )
            row[f"local_absolute_strength_{CHANNEL_NAMES[k]}"] = round(
                float(abs(all_raw_arr[i, k]) / local_max), 6
            )
            row[f"local_signed_strength_{CHANNEL_NAMES[k]}"] = round(
                float(all_raw_arr[i, k] / local_max), 6
            )
        row["completeness_error"] = completeness_errors[i]
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(SAVE_ROOT, "deposit_deeplift_summary.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] 汇总 CSV 已保存: {os.path.basename(csv_path)}")

    median_relative = np.median(all_relative_arr, axis=0)
    mean_relative = all_relative_arr.mean(axis=0)
    mean_absolute_relative = np.abs(all_relative_arr).mean(axis=0)
    local_scale = np.maximum(np.abs(all_raw_arr).max(axis=1, keepdims=True), 1e-12)
    signed_strength_arr = all_raw_arr / local_scale
    median_signed_strength = np.median(signed_strength_arr, axis=0)

    for version_name, version_channels, version_dir in plot_versions:
        sort_idx = sorted(
            version_channels, key=lambda ch: median_signed_strength[ch], reverse=True
        )
        sorted_names = [display_name(i) for i in sort_idx]
        sorted_values = [float(median_signed_strength[i]) for i in sort_idx]
        fig_avg, ax_avg = plt.subplots(
            figsize=(max(12, len(sort_idx) * 0.46), 5.8), facecolor="white"
        )
        xp = np.arange(len(sort_idx))
        _academic_bar_chart(
            ax_avg,
            xp,
            sorted_values,
            sorted_names,
            bar_color="#2878B5",
            bar_width=0.5,
            ylim_min=1.0,
            fixed_ylim=1.0,
        )
        ax_avg.set_ylabel(
            "Median signed local contribution strength (max |C| = 1)", fontsize=11
        )
        ax_avg.set_title(
            f"Global DeepLIFT contribution – {version_name} "
            f"({len(deposit_rc)} Li deposits)",
            fontsize=10,
            pad=8,
        )
        fig_avg.tight_layout()
        avg_path = os.path.join(version_dir, "global_signed_strength_ranking.png")
        fig_avg.savefig(avg_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig_avg)
        print(f"  [OK] {version_name}全局排序图 → {avg_path}")

    global_rows = []
    for channel in range(42):
        values = all_relative_arr[:, channel]
        direction = np.sign(median_relative[channel])
        nonzero = values != 0
        consistency = (
            float(np.mean(np.sign(values[nonzero]) == direction))
            if direction != 0 and np.any(nonzero)
            else np.nan
        )
        global_rows.append(
            {
                "channel_index": channel,
                "factor": CHANNEL_NAMES[channel],
                "median_signed_relative_pct": median_relative[channel],
                "mean_signed_relative_pct": mean_relative[channel],
                "mean_absolute_relative_pct": mean_absolute_relative[channel],
                "median_signed_local_strength": median_signed_strength[channel],
                "sign_consistency": consistency,
                "mean_absolute_raw_deeplift": np.abs(all_raw_arr[:, channel]).mean(),
                "selected_for_geological_prior_plot": channel
                in (GEOLOGICAL_PRIOR_CHANNELS + GEO_CHANNELS),
            }
        )
    global_path = os.path.join(SAVE_ROOT, "global_contribution_summary.csv")
    pd.DataFrame(global_rows).sort_values(
        "median_signed_relative_pct", ascending=False
    ).to_csv(global_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] 全局贡献汇总 → {os.path.basename(global_path)}")

    print("\n" + "=" * 70)
    print("全部完成！")
    print("=" * 70)
    print(f"\n结果保存在: {SAVE_ROOT}/")
    print(f"  - geological_prior/：16种成矿相关因子 + 3种控矿要素")
    print(f"  - all_42_factors/：全部42因子排序")
    print(f"  - 每个版本均含 {len(deposit_rc)} 幅矿点图和1幅全局排序图")
    print(f"  - deposit_deeplift_summary.csv ：局部原始、绝对及相对贡献")
    print(f"  - global_contribution_summary.csv：全局统计与符号一致率")
    print(f"\n图表说明:")
    print(f"  X 轴 ：预注册的{TOP_N_ELEMENT}种成矿相关元素/氧化物 + 3种控矿要素")
    print(f'  地球化学因子：{", ".join(GEOLOGICAL_PRIOR_NAMES)}')
    print(f"  审计 ：CSV仍保留全部42因子；Bi等非先验因子不从审计结果中删除")
    print(f"  单图 ：有符号局部贡献强度，统一 -1～1，正值向上、负值向下")
    print(f"  全局 ：各矿点有符号局部贡献强度的中位数，由正到负排序")
    print(f'  正值 ：该特征相对背景值偏高，促进模型预测为"有矿"')
    print(f'  负值 ：该特征相对背景值偏低，抑制模型预测为"有矿"')
    print(f"  颜色 ：所有柱体统一为蓝色，柱体不标数值")


if __name__ == "__main__":
    main()
