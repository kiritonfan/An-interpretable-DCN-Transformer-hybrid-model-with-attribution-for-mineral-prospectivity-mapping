"""Generate study-area Grad-CAM maps from the trained model."""


import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


COMBINED_NPY = None
TRAIN_DATA = None
LABEL_TIF = None
GEO_TIF_REF = None
XX_TIF = None
YY_TIF = None
MODEL_PATH = None
SAVE_DIR = None


FEATURE_DIM = 32
TRANSFORMER_DEPTH = 1
TRANSFORMER_HEADS = 2


BATCH_SIZE = 256
WINDOW_HALF = 4
TARGET_CLASS = 1


MERGE_DIST = 5


FIGURE_DPI = 300
SMOOTH_SIGMA = 1.5
CONTOUR_LEVELS = 8
VMIN_PERCENTILE = 2
VMAX_PERCENTILE = 98


def load_geo_info(tif_path):

    try:
        import rasterio

        with rasterio.open(tif_path) as src:
            return src.transform, src.crs, src.height, src.width
    except Exception as e:
        print(f"  [警告] 无法读取地理信息: {e}")
        return None, None, None, None


def pixel_to_lonlat(row, col, transform, crs):

    try:
        from pyproj import Transformer

        x = transform.c + col * transform.a + row * transform.b
        y = transform.f + col * transform.d + row * transform.e
        t = Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True)
        lon, lat = t.transform(x, y)
        return lon, lat
    except Exception:
        return None, None


def build_latlon_ticks(H, W, transform, crs, n_lon=5, n_lat=5):

    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    lons, lats = [], []
    for r, c in corners:
        lon, lat = pixel_to_lonlat(r, c, transform, crs)
        if lon is not None:
            lons.append(lon)
            lats.append(lat)
    if not lons:
        return None, None, None, None

    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    lon_vals = np.linspace(lon_min, lon_max, n_lon)
    lat_vals = np.linspace(lat_max, lat_min, n_lat)

    try:
        from pyproj import Transformer

        t_inv = Transformer.from_crs(4326, crs.to_epsg(), always_xy=True)
        col_ticks, lon_labels = [], []
        for lon in lon_vals:
            lat_mid = (lat_min + lat_max) / 2
            x, y = t_inv.transform(lon, lat_mid)
            col = (x - transform.c) / transform.a
            if 0 <= col <= W:
                col_ticks.append(col)
                lon_labels.append(f"{lon:.2f}°E" if lon >= 0 else f"{-lon:.2f}°W")

        row_ticks, lat_labels = [], []
        for lat in lat_vals:
            lon_mid = (lon_min + lon_max) / 2
            x, y = t_inv.transform(lon_mid, lat)
            row = (y - transform.f) / transform.e
            if 0 <= row <= H:
                row_ticks.append(row)
                lat_labels.append(f"{lat:.2f}°N" if lat >= 0 else f"{-lat:.2f}°S")

        return col_ticks, lon_labels, row_ticks, lat_labels
    except Exception:
        return None, None, None, None


def compute_full_gradcam_map(model, combined_norm, device):

    H, W, C = combined_norm.shape
    half = WINDOW_HALF

    cam_map = np.full((H, W), np.nan, dtype=np.float32)
    pred_map = np.full((H, W), np.nan, dtype=np.float32)

    valid_pos = [(r, c) for r in range(half, H - half) for c in range(half, W - half)]
    n_valid = len(valid_pos)
    print(f"  有效格点数: {n_valid:,}（{H}×{W}={H*W:,} 中去除边缘 {half} 像素后）")

    model.eval()
    n_batches = (n_valid + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in tqdm(range(n_batches), desc="全图 Grad-CAM 计算", ncols=72):
        batch_pos = valid_pos[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        bsz = len(batch_pos)

        windows = np.stack(
            [
                combined_norm[r - half : r + half + 1, c - half : c + half + 1, :]
                for r, c in batch_pos
            ]
        )  # (bsz, 9, 9, 42)
        inp = (
            torch.FloatTensor(windows).permute(0, 3, 1, 2).to(device)  # (bsz, 42, 9, 9)
        )

        acts_buf = {}
        grads_buf = {}

        def fwd_hook(module, inp_t, out_t):
            acts_buf["A"] = out_t  # (bsz, 128, 9, 9)
            out_t.register_hook(lambda g: grads_buf.update({"G": g.detach()}))

        handle = model.dcn.block2.register_forward_hook(fwd_hook)

        output = model(inp)  # (bsz, 2)
        probs = F.softmax(output, dim=1)  # (bsz, 2)

        model.zero_grad()
        output[:, TARGET_CLASS].sum().backward()

        handle.remove()

        if "A" not in acts_buf or "G" not in grads_buf:
            continue

        A = acts_buf["A"].detach()  # (bsz, 128, 9, 9)
        G = grads_buf["G"]  # (bsz, 128, 9, 9)

        alpha = G.mean(dim=[2, 3])  # (bsz, 128)

        cam_batch = torch.einsum("bc,bchw->bhw", alpha, A)  # (bsz, 9, 9)
        cam_batch = F.relu(cam_batch)  # (bsz, 9, 9)

        cam_np = cam_batch.cpu().numpy()  # (bsz, 9, 9)
        pred_np = probs[:, TARGET_CLASS].detach().cpu().numpy()  # (bsz,)

        for i, (r, c) in enumerate(batch_pos):

            cam_single = cam_np[i]  # (9, 9)
            mn, mx = cam_single.min(), cam_single.max()
            if mx > mn:
                cam_val = (cam_single[half, half] - mn) / (mx - mn)
            else:
                cam_val = 0.0

            cam_map[r, c] = cam_val
            pred_map[r, c] = pred_np[i]

    return cam_map, pred_map


def export_to_csv(cam_map, pred_map, cam_smooth, save_path):

    import pandas as pd
    from PIL import Image

    H, W = cam_map.shape
    rows_idx, cols_idx = np.where(~np.isnan(cam_map))
    n = len(rows_idx)
    print(f"  有效格点数: {n:,}，正在构建表格...")

    coord_source = "行列索引（回退）"
    X = cols_idx.astype(np.float64)
    Y = rows_idx.astype(np.float64)

    if XX_TIF and YY_TIF and os.path.exists(XX_TIF) and os.path.exists(YY_TIF):
        try:
            XX = np.array(Image.open(XX_TIF))
            YY = np.array(Image.open(YY_TIF))
            if XX.shape == (H, W) and YY.shape == (H, W):
                X = XX[rows_idx, cols_idx].astype(np.float64)
                Y = YY[rows_idx, cols_idx].astype(np.float64)
                coord_source = "XX.tif / YY.tif（UTM，与预测代码一致）"
            else:
                print(f"  [警告] XX/YY.tif 形状 {XX.shape} 与栅格 ({H},{W}) 不符")
        except Exception as e:
            print(f"  [警告] 读取 XX/YY.tif 失败: {e}")
    else:

        try:
            import rasterio

            with rasterio.open(GEO_TIF_REF) as src:
                tf = src.transform

                X = tf.c + cols_idx * tf.a + rows_idx * tf.b
                Y = tf.f + cols_idx * tf.d + rows_idx * tf.e
                coord_source = f"rasterio 仿射变换（{GEO_TIF_REF}，投影坐标）"
        except Exception as e:
            print(f"  [警告] rasterio 读取失败: {e}，改用行列索引")

    print(f"  坐标来源: {coord_source}")

    data = {
        "X": np.round(X, 2),
        "Y": np.round(Y, 2),
        "row": rows_idx,
        "col": cols_idx,
        "gradcam": np.round(cam_map[rows_idx, cols_idx].astype(np.float64), 6),
        "pred_prob": np.round(pred_map[rows_idx, cols_idx].astype(np.float64), 6),
    }

    if cam_smooth is not None:
        data["gradcam_smooth"] = np.round(
            cam_smooth[rows_idx, cols_idx].astype(np.float64), 6
        )

    df = pd.DataFrame(data)

    df.sort_values(["row", "col"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] CSV 已保存: {os.path.basename(save_path)}")
    print(f"       行数: {len(df):,}  列数: {len(df.columns)}")
    print(f"       列名: {list(df.columns)}")
    print("\n  前 3 行预览:")
    print(df.head(3).to_string(index=False))
    return df


def _apply_geo_ticks(ax, H, W, transform, crs, n_lon=6, n_lat=6):

    if transform is not None and crs is not None:
        col_ticks, lon_labels, row_ticks, lat_labels = build_latlon_ticks(
            H, W, transform, crs, n_lon=n_lon, n_lat=n_lat
        )
        if col_ticks:
            ax.set_xticks(col_ticks)
            ax.set_xticklabels(lon_labels, fontsize=9)
            ax.set_yticks(row_ticks)
            ax.set_yticklabels(lat_labels, fontsize=9)
            ax.tick_params(direction="in", length=4, width=0.8)
            return
    ax.set_xlabel("Column (pixel)", fontsize=10)
    ax.set_ylabel("Row (pixel)", fontsize=10)


def plot_anomaly_map(
    cam_map, pred_map, deposit_rc, transform, crs, save_path, smooth=False
):

    H, W = cam_map.shape
    data = cam_map.copy()

    if smooth and SMOOTH_SIGMA > 0:
        valid_mask = ~np.isnan(data)
        tmp = np.where(valid_mask, data, 0.0)
        smoothed = gaussian_filter(tmp, sigma=SMOOTH_SIGMA)

        weight = gaussian_filter(valid_mask.astype(np.float32), sigma=SMOOTH_SIGMA)
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = np.where(weight > 0.01, smoothed / weight, np.nan)
        data = smoothed

    valid = data[~np.isnan(data)]
    vmin = np.percentile(valid, VMIN_PERCENTILE)
    vmax = np.percentile(valid, VMAX_PERCENTILE)

    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor="white")

    im = ax.imshow(
        data,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
        origin="upper",
    )

    data_clean = np.where(np.isnan(data), vmin, data)
    contour_lvls = np.linspace(vmin, vmax, CONTOUR_LEVELS + 2)[1:-1]
    cs = ax.contour(
        data_clean,
        levels=contour_lvls,
        colors="black",
        linewidths=0.55,
        linestyles="-",
        alpha=0.55,
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f", inline_spacing=3)

    if pred_map is not None:
        pred_clean = np.where(np.isnan(pred_map), 0, pred_map)
        ax.contour(
            pred_clean,
            levels=[0.5],
            colors="white",
            linewidths=1.2,
            linestyles="--",
            alpha=0.85,
        )

    if len(deposit_rc) > 0:
        dep_rows, dep_cols = deposit_rc[:, 0], deposit_rc[:, 1]
        ax.scatter(
            dep_cols,
            dep_rows,
            marker="*",
            s=160,
            c="gold",
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
            label="Known Li deposits",
        )
        ax.legend(
            loc="lower right",
            fontsize=10,
            framealpha=0.88,
            edgecolor="gray",
            handletextpad=0.4,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Grad-CAM intensity\n(normalized, DCN block2)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    _apply_geo_ticks(ax, H, W, transform, crs)

    ax.grid(True, color="white", linewidth=0.4, linestyle="--", alpha=0.45)

    title_suffix = "(Gaussian smoothed)" if smooth else ""
    ax.set_title(
        f"Grad-CAM Anomaly Map  –  DCNv2 Block2  {title_suffix}\n"
        "(Class: Li deposits  |  Center-pixel activation × gradient)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] 已保存: {os.path.basename(save_path)}")


def plot_prediction_map(pred_map, deposit_rc, transform, crs, save_path):

    H, W = pred_map.shape
    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor="white")

    im = ax.imshow(
        pred_map,
        cmap="Reds",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
        aspect="equal",
        origin="upper",
    )

    if len(deposit_rc) > 0:
        ax.scatter(
            deposit_rc[:, 1],
            deposit_rc[:, 0],
            marker="*",
            s=160,
            c="gold",
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
            label="Known Li deposits",
        )
        ax.legend(loc="lower right", fontsize=10, framealpha=0.88, edgecolor="gray")

    pred_clean = np.where(np.isnan(pred_map), 0, pred_map)
    ax.contour(
        pred_clean,
        levels=[0.5],
        colors="steelblue",
        linewidths=1.0,
        linestyles="--",
        alpha=0.8,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("P(Li deposit)", fontsize=11)
    cbar.ax.tick_params(labelsize=8)

    _apply_geo_ticks(ax, H, W, transform, crs)
    ax.grid(True, color="white", linewidth=0.4, linestyle="--", alpha=0.45)
    ax.set_title(
        "Mineral Prospectivity Map  –  Li deposits\n"
        "(Model prediction probability, DCN-Transformer)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] 已保存: {os.path.basename(save_path)}")


def plot_combined_map(cam_data, pred_map, deposit_rc, transform, crs, save_path):

    H, W = cam_data.shape

    valid = cam_data[~np.isnan(cam_data)]
    vmin = np.percentile(valid, VMIN_PERCENTILE)
    vmax = np.percentile(valid, VMAX_PERCENTILE)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), facecolor="white")

    ax = axes[0]
    im0 = ax.imshow(
        cam_data,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
        origin="upper",
    )
    data_clean = np.where(np.isnan(cam_data), vmin, cam_data)
    contour_lvls = np.linspace(vmin, vmax, CONTOUR_LEVELS + 2)[1:-1]
    cs0 = ax.contour(
        data_clean,
        levels=contour_lvls,
        colors="black",
        linewidths=0.5,
        linestyles="-",
        alpha=0.5,
    )
    ax.clabel(cs0, inline=True, fontsize=5.5, fmt="%.2f", inline_spacing=3)
    if pred_map is not None:
        pred_clean = np.where(np.isnan(pred_map), 0, pred_map)
        ax.contour(
            pred_clean,
            levels=[0.5],
            colors="white",
            linewidths=1.0,
            linestyles="--",
            alpha=0.8,
        )
    if len(deposit_rc) > 0:
        ax.scatter(
            deposit_rc[:, 1],
            deposit_rc[:, 0],
            marker="*",
            s=130,
            c="gold",
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
            label="Known Li deposits",
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    cbar0 = fig.colorbar(im0, ax=ax, fraction=0.035, pad=0.02)
    cbar0.set_label("Grad-CAM intensity", fontsize=9)
    cbar0.ax.tick_params(labelsize=7.5)
    _apply_geo_ticks(ax, H, W, transform, crs, n_lon=5, n_lat=5)
    ax.grid(True, color="white", linewidth=0.35, linestyle="--", alpha=0.4)
    ax.set_title(
        "Grad-CAM Anomaly Map\n(DCNv2 Block2, Gaussian smoothed)",
        fontsize=11,
        fontweight="bold",
        pad=6,
    )

    ax = axes[1]
    im1 = ax.imshow(
        pred_map,
        cmap="Reds",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
        aspect="equal",
        origin="upper",
    )
    if pred_map is not None:
        pred_clean = np.where(np.isnan(pred_map), 0, pred_map)
        ax.contour(
            pred_clean,
            levels=[0.5],
            colors="steelblue",
            linewidths=1.0,
            linestyles="--",
            alpha=0.8,
        )
    if len(deposit_rc) > 0:
        ax.scatter(
            deposit_rc[:, 1],
            deposit_rc[:, 0],
            marker="*",
            s=130,
            c="gold",
            edgecolors="black",
            linewidths=0.7,
            zorder=6,
            label="Known Li deposits",
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.035, pad=0.02)
    cbar1.set_label("P(Li deposit)", fontsize=9)
    cbar1.ax.tick_params(labelsize=7.5)
    _apply_geo_ticks(ax, H, W, transform, crs, n_lon=5, n_lat=5)
    ax.grid(True, color="white", linewidth=0.35, linestyle="--", alpha=0.4)
    ax.set_title(
        "Mineral Prospectivity Map\n(DCN-Transformer prediction probability)",
        fontsize=11,
        fontweight="bold",
        pad=6,
    )

    fig.suptitle(
        "Study Area Grad-CAM Anomaly  vs.  Prediction Probability  –  Li Deposits",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] 已保存: {os.path.basename(save_path)}")


def main(argv=None):
    global COMBINED_NPY, TRAIN_DATA, LABEL_TIF, GEO_TIF_REF
    global XX_TIF, YY_TIF, MODEL_PATH, SAVE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--study-area-data", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--georeference")
    parser.add_argument("--x-coordinates")
    parser.add_argument("--y-coordinates")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    MODEL_PATH = os.path.abspath(args.model)
    COMBINED_NPY = os.path.abspath(args.study_area_data)
    TRAIN_DATA = os.path.abspath(args.train_data)
    LABEL_TIF = os.path.abspath(args.labels) if args.labels else None
    GEO_TIF_REF = os.path.abspath(args.georeference) if args.georeference else None
    XX_TIF = os.path.abspath(args.x_coordinates) if args.x_coordinates else None
    YY_TIF = os.path.abspath(args.y_coordinates) if args.y_coordinates else None
    SAVE_DIR = os.path.abspath(args.output_dir)
    print("=" * 72)
    print("Grad-CAM 全图可视化：研究区动态卷积特征重要性异常图")
    print("=" * 72)

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if device.type == "cpu":
        print("  [提示] CPU 模式下计算较慢，建议 GPU 环境运行。")

    print("\n[1/5] 加载模型...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}\n" "请修改脚本顶部 MODEL_PATH 变量。")
    from attribution_guided_model import create_attribution_guided_model

    model = create_attribution_guided_model(
        num_classes=2,
        feature_dim=FEATURE_DIM,
        transformer_depth=TRANSFORMER_DEPTH,
        transformer_heads=TRANSFORMER_HEADS,
    ).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  [OK] 模型加载成功（最佳验证准确率: " f'{ckpt.get("val_acc", 0):.2f}%）')

    print("\n[2/5] 加载全图数据...")
    if not os.path.exists(COMBINED_NPY):
        raise FileNotFoundError(f"找不到全图数据: {COMBINED_NPY}")
    combined = np.load(COMBINED_NPY).astype(np.float32)
    H, W, C = combined.shape
    print(f"  形状: {combined.shape}  ({H}行 × {W}列 × {C}波段)")

    print("\n[3/5] 数据标准化...")
    from sklearn.preprocessing import StandardScaler

    train_data = np.load(TRAIN_DATA).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(train_data.reshape(-1, C))
    combined_norm = scaler.transform(combined.reshape(-1, C)).reshape(H, W, C)
    del train_data
    print("  [OK] 标准化完成")

    print("\n[4/5] 读取地理信息与矿点坐标...")
    import cv2

    deposit_rc = np.empty((0, 2), dtype=int)
    if LABEL_TIF and os.path.exists(LABEL_TIF):
        label1 = cv2.imread(LABEL_TIF, 2)
        if label1 is not None:
            raw_coords = np.array(list(zip(*np.where(label1 == 1))))
            kept, used = [], [False] * len(raw_coords)
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
            deposit_rc = raw_coords[kept]
            print(f"  矿点数: {len(deposit_rc)} 个（去重后）")
    else:
        print(f"  [警告] 未找到标签文件: {LABEL_TIF}，矿点将不显示")

    geo_transform, geo_crs, _, _ = load_geo_info(GEO_TIF_REF)
    if geo_transform is not None:
        print("  [OK] 地理信息加载成功")
    else:
        print("  [警告] 地理信息不可用，将使用行列索引为坐标轴")

    print(
        f"\n[5/5] 计算全图 Grad-CAM（共 {H * W:,} 格点，"
        f"批大小 {BATCH_SIZE}，跳过 {WINDOW_HALF} 像素边缘）..."
    )
    cam_map, pred_map = compute_full_gradcam_map(model, combined_norm, device)
    valid_cnt = np.sum(~np.isnan(cam_map))
    print(f"  [OK] 计算完成（有效格点: {valid_cnt:,}）")

    np.save(os.path.join(SAVE_DIR, "gradcam_map.npy"), cam_map)
    np.save(os.path.join(SAVE_DIR, "pred_map.npy"), pred_map)
    print(f"  [OK] 中间数据已保存至 {SAVE_DIR}/")

    cam_smooth = None
    if SMOOTH_SIGMA > 0:
        valid_mask = ~np.isnan(cam_map)
        tmp = np.where(valid_mask, cam_map, 0.0)
        sm = gaussian_filter(tmp, sigma=SMOOTH_SIGMA)
        wt = gaussian_filter(valid_mask.astype(np.float32), sigma=SMOOTH_SIGMA)
        with np.errstate(invalid="ignore", divide="ignore"):
            cam_smooth = np.where(wt > 0.01, sm / wt, np.nan)

    print("\n[7/6] 导出地理坐标 CSV...")
    import pandas as pd

    csv_path = os.path.join(SAVE_DIR, "gradcam_results.csv")
    export_to_csv(cam_map, pred_map, cam_smooth, csv_path)

    print("\n[8/6] 生成可视化图像...")

    plot_anomaly_map(
        cam_map,
        pred_map,
        deposit_rc,
        geo_transform,
        geo_crs,
        save_path=os.path.join(SAVE_DIR, "gradcam_anomaly_map.png"),
        smooth=False,
    )

    if cam_smooth is not None:
        plot_anomaly_map(
            cam_smooth,
            pred_map,
            deposit_rc,
            geo_transform,
            geo_crs,
            save_path=os.path.join(SAVE_DIR, "gradcam_anomaly_map_smooth.png"),
            smooth=False,
        )

    plot_prediction_map(
        pred_map,
        deposit_rc,
        geo_transform,
        geo_crs,
        save_path=os.path.join(SAVE_DIR, "pred_probability_map.png"),
    )

    plot_combined_map(
        cam_smooth if cam_smooth is not None else cam_map,
        pred_map,
        deposit_rc,
        geo_transform,
        geo_crs,
        save_path=os.path.join(SAVE_DIR, "gradcam_combined.png"),
    )

    print("\n" + "=" * 72)
    print("全部完成！")
    print("=" * 72)
    print(f"\n结果保存在: {SAVE_DIR}/")
    print("  gradcam_anomaly_map.png        ：原始 Grad-CAM 异常图")
    if SMOOTH_SIGMA > 0:
        print("  gradcam_anomaly_map_smooth.png ：高斯平滑异常图")
    print("  pred_probability_map.png       ：模型矿化概率图（对比参考）")
    print("  gradcam_combined.png           ：异常图 + 概率图双图对比版")
    print("  gradcam_map.npy / pred_map.npy ：中间数据（可用于重新出图）")
    print("  gradcam_results.csv            ：地理坐标 + Grad-CAM + 预测概率（GIS 投图用）")
    print("\nCSV 列说明:")
    print("  X             : UTM 东向坐标（米）或列索引（取决于坐标文件是否存在）")
    print("  Y             : UTM 北向坐标（米）或行索引")
    print("  row / col     : 栅格行列号（0-indexed，可用于像素索引对应）")
    print("  gradcam       : 归一化 Grad-CAM 值（0~1，中心像素）")
    if SMOOTH_SIGMA > 0:
        print("  gradcam_smooth: 高斯平滑后的 Grad-CAM 值（σ={:.1f}）".format(SMOOTH_SIGMA))
    print("  pred_prob     : 模型预测为 Li 矿点的概率（0~1）")
    print("\n图例说明:")
    print("  · 颜色（jet）：蓝色→低 Grad-CAM 值（低特征重要性），" "红色→高值（高特征重要性）")
    print("  · 黑色等值线：Grad-CAM 强度等值线（类似地球化学异常线）")
    print("  · 白色虚线  ：模型预测概率 P=0.5 边界（矿/非矿分界线）")
    print("  · 金色五角星：已知 Li 矿点位置")


if __name__ == "__main__":
    main()
