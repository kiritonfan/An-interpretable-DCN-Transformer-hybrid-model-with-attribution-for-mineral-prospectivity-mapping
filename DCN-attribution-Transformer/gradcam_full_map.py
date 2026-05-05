"""This script generates a full-map Grad-CAM anomaly visualization for the study area."""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── Paths ─────────────────────────────────────────────────────────────────────
NANLING_DIR  = r'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling'
COMBINED_NPY = os.path.join(NANLING_DIR, 'Combined_array.npy')
TRAIN_DATA   = os.path.join(NANLING_DIR, 'train_data.npy')
LABEL_TIF    = os.path.join(NANLING_DIR, 'New_data/label/Li_deposits0.tif')
GEO_TIF_REF  = os.path.join(NANLING_DIR, 'New_data/Geochemical_data/Extract_cu1.tif')
XX_TIF       = os.path.join(NANLING_DIR, 'New_data/coordinate/XX.tif')
YY_TIF       = os.path.join(NANLING_DIR, 'New_data/coordinate/YY.tif')

MODEL_PATH = os.path.join(current_dir, 'best_attribution_guided_model.pth')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(current_dir),
                              'best_attribution_guided_model.pth')

SAVE_DIR = os.path.join(current_dir, 'gradcam_full_map')

# ── Model hyperparameters (must match training configuration) ─────────────────
FEATURE_DIM       = 32
TRANSFORMER_DEPTH = 1
TRANSFORMER_HEADS = 2

# ── Computation parameters ────────────────────────────────────────────────────
BATCH_SIZE   = 256   # patches per batch (GPU: 512 recommended, CPU: 128)
WINDOW_HALF  = 4     # window radius: 2×4+1 = 9
TARGET_CLASS = 1     # deposit class index

# ── Deposit merging threshold (de-duplicate radius in pixels) ─────────────────
MERGE_DIST = 5

# ── Visualization parameters ──────────────────────────────────────────────────
FIGURE_DPI      = 300
SMOOTH_SIGMA    = 1.5   # Gaussian smooth sigma (pixels); set to 0 to disable
CONTOUR_LEVELS  = 8     # number of anomaly contour lines
VMIN_PERCENTILE = 2     # colorscale lower bound percentile
VMAX_PERCENTILE = 98    # colorscale upper bound percentile


def load_geo_info(tif_path):
    """Read affine transform and CRS from a reference GeoTIFF."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            return src.transform, src.crs, src.height, src.width
    except Exception as e:
        print(f'  [Warning] Cannot read geo info: {e}')
        return None, None, None, None


def pixel_to_lonlat(row, col, transform, crs):
    """Convert pixel coordinates to longitude / latitude (degrees)."""
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
    """Compute pixel positions and labels for lon/lat axis ticks."""
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
                lon_labels.append(f'{lon:.2f}°E' if lon >= 0 else f'{-lon:.2f}°W')

        row_ticks, lat_labels = [], []
        for lat in lat_vals:
            lon_mid = (lon_min + lon_max) / 2
            x, y = t_inv.transform(lon_mid, lat)
            row = (y - transform.f) / transform.e
            if 0 <= row <= H:
                row_ticks.append(row)
                lat_labels.append(f'{lat:.2f}°N' if lat >= 0 else f'{-lat:.2f}°S')

        return col_ticks, lon_labels, row_ticks, lat_labels
    except Exception:
        return None, None, None, None


def compute_full_gradcam_map(model, combined_norm, device):
    """
    Compute per-pixel Grad-CAM scalars over the full study area using DCNv2 block2.

    For each valid pixel a 9x9 patch is extracted, and the Grad-CAM value at the
    center position (4, 4) is used as the feature-importance score for that pixel.

    Args:
        model         : trained AttributionGuidedDCNTransformer in eval mode
        combined_norm : (H, W, 42) standardized full-area data
        device        : torch.device

    Returns:
        cam_map  : (H, W) ndarray with NaN at border pixels
        pred_map : (H, W) ndarray with NaN at border pixels
    """
    H, W, C = combined_norm.shape
    half = WINDOW_HALF

    cam_map  = np.full((H, W), np.nan, dtype=np.float32)
    pred_map = np.full((H, W), np.nan, dtype=np.float32)

    # Collect valid pixel coordinates (exclude border where full 9x9 window is unavailable)
    valid_pos = [
        (r, c)
        for r in range(half, H - half)
        for c in range(half, W - half)
    ]
    n_valid = len(valid_pos)
    print(f'  Valid pixels: {n_valid:,} '
          f'(from {H}x{W}={H*W:,} total, excluding {half}-pixel border)')

    model.eval()
    n_batches = (n_valid + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in tqdm(range(n_batches), desc='Full-map Grad-CAM', ncols=72):
        batch_pos = valid_pos[b_idx * BATCH_SIZE: (b_idx + 1) * BATCH_SIZE]
        bsz = len(batch_pos)

        # ── Build batch windows: (bsz, 42, 9, 9) ────────────────────────────
        windows = np.stack([
            combined_norm[r - half: r + half + 1,
                          c - half: c + half + 1, :]
            for r, c in batch_pos
        ])                                           # (bsz, 9, 9, 42)
        inp = (torch.FloatTensor(windows)
               .permute(0, 3, 1, 2)                 # (bsz, 42, 9, 9)
               .to(device))

        # ── Register hooks to capture block2 activations and gradients ───────
        acts_buf  = {}
        grads_buf = {}

        def fwd_hook(module, inp_t, out_t):
            acts_buf['A'] = out_t                    # (bsz, 128, 9, 9)
            out_t.register_hook(
                lambda g: grads_buf.update({'G': g.detach()})
            )

        handle = model.dcn.block2.register_forward_hook(fwd_hook)

        # ── Forward pass ─────────────────────────────────────────────────────
        output = model(inp)                          # (bsz, 2)
        probs  = F.softmax(output, dim=1)            # (bsz, 2)

        # ── Backward pass (sum over target-class logits; samples are independent
        #    in eval mode) ───────────────────────────────────────────────────
        model.zero_grad()
        output[:, TARGET_CLASS].sum().backward()

        handle.remove()

        if 'A' not in acts_buf or 'G' not in grads_buf:
            continue

        # ── Grad-CAM computation (batched) ───────────────────────────────────
        A = acts_buf['A'].detach()       # (bsz, 128, 9, 9)
        G = grads_buf['G']               # (bsz, 128, 9, 9)

        alpha     = G.mean(dim=[2, 3])                        # (bsz, 128)
        cam_batch = torch.einsum('bc,bchw->bhw', alpha, A)    # (bsz, 9, 9)
        cam_batch = F.relu(cam_batch)                         # (bsz, 9, 9)

        cam_np  = cam_batch.cpu().numpy()                     # (bsz, 9, 9)
        pred_np = probs[:, TARGET_CLASS].detach().cpu().numpy()  # (bsz,)

        for i, (r, c) in enumerate(batch_pos):
            # Take center pixel (half, half) = (4, 4) as the scalar Grad-CAM value
            cam_single = cam_np[i]
            mn, mx = cam_single.min(), cam_single.max()
            cam_val = (cam_single[half, half] - mn) / (mx - mn) if mx > mn else 0.0
            cam_map[r, c]  = cam_val
            pred_map[r, c] = pred_np[i]

    return cam_map, pred_map


def export_to_csv(cam_map, pred_map, cam_smooth, save_path):
    """
    Export full-map Grad-CAM values and prediction probabilities to a geo-referenced CSV.

    Coordinate source priority:
      1. XX.tif / YY.tif (per-pixel UTM X/Y, consistent with prediction pipeline)
      2. rasterio affine transform (derived from GEO_TIF_REF)
      3. Row / column index (fallback)

    CSV columns:
        X             : UTM easting (m) or column index
        Y             : UTM northing (m) or row index
        row           : raster row number (0-indexed)
        col           : raster column number (0-indexed)
        gradcam       : normalized Grad-CAM value at center pixel [0, 1]
        gradcam_smooth: Gaussian-smoothed Grad-CAM value (if computed)
        pred_prob     : model probability of deposit class [0, 1]

    Only valid (non-NaN) pixels are written; border pixels are excluded.
    """
    import pandas as pd
    from PIL import Image

    H, W = cam_map.shape
    rows_idx, cols_idx = np.where(~np.isnan(cam_map))
    n = len(rows_idx)
    print(f'  Valid pixels: {n:,} — building table...')

    # ── Geographic coordinates (prefer XX/YY.tif) ────────────────────────────
    coord_source = 'row/col index (fallback)'
    X = cols_idx.astype(np.float64)
    Y = rows_idx.astype(np.float64)

    if os.path.exists(XX_TIF) and os.path.exists(YY_TIF):
        try:
            XX = np.array(Image.open(XX_TIF))
            YY = np.array(Image.open(YY_TIF))
            if XX.shape == (H, W) and YY.shape == (H, W):
                X = XX[rows_idx, cols_idx].astype(np.float64)
                Y = YY[rows_idx, cols_idx].astype(np.float64)
                coord_source = 'XX.tif / YY.tif (UTM, consistent with prediction pipeline)'
            else:
                print(f'  [Warning] XX/YY.tif shape {XX.shape} does not match raster ({H},{W})')
        except Exception as e:
            print(f'  [Warning] Failed to read XX/YY.tif: {e}')
    else:
        try:
            import rasterio
            with rasterio.open(GEO_TIF_REF) as src:
                tf = src.transform
                X  = tf.c + cols_idx * tf.a + rows_idx * tf.b
                Y  = tf.f + cols_idx * tf.d + rows_idx * tf.e
                coord_source = f'rasterio affine transform ({GEO_TIF_REF}, projected coords)'
        except Exception as e:
            print(f'  [Warning] rasterio read failed: {e} — using row/col index')

    print(f'  Coordinate source: {coord_source}')

    # ── Build DataFrame ───────────────────────────────────────────────────────
    data = {
        'X'         : np.round(X, 2),
        'Y'         : np.round(Y, 2),
        'row'       : rows_idx,
        'col'       : cols_idx,
        'gradcam'   : np.round(cam_map[rows_idx, cols_idx].astype(np.float64), 6),
        'pred_prob' : np.round(pred_map[rows_idx, cols_idx].astype(np.float64), 6),
    }

    if cam_smooth is not None:
        data['gradcam_smooth'] = np.round(
            cam_smooth[rows_idx, cols_idx].astype(np.float64), 6
        )

    df = pd.DataFrame(data)
    df.sort_values(['row', 'col'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f'  [OK] CSV saved: {os.path.basename(save_path)}')
    print(f'       Rows: {len(df):,}  Columns: {len(df.columns)}')
    print(f'       Column names: {list(df.columns)}')
    print('\n  First 3 rows preview:')
    print(df.head(3).to_string(index=False))
    return df


def _apply_geo_ticks(ax, H, W, transform, crs, n_lon=6, n_lat=6):
    """Apply lon/lat axis ticks if geo info is available, otherwise use pixel indices."""
    if transform is not None and crs is not None:
        col_ticks, lon_labels, row_ticks, lat_labels = build_latlon_ticks(
            H, W, transform, crs, n_lon=n_lon, n_lat=n_lat
        )
        if col_ticks:
            ax.set_xticks(col_ticks)
            ax.set_xticklabels(lon_labels, fontsize=9)
            ax.set_yticks(row_ticks)
            ax.set_yticklabels(lat_labels, fontsize=9)
            ax.tick_params(direction='in', length=4, width=0.8)
            return
    ax.set_xlabel('Column (pixel)', fontsize=10)
    ax.set_ylabel('Row (pixel)', fontsize=10)


def plot_anomaly_map(cam_map, pred_map, deposit_rc,
                     transform, crs, save_path, smooth=False):
    """
    Draw a geochemical-anomaly-style Grad-CAM map for the study area.

    Parameters
    ----------
    cam_map    : (H, W) ndarray with NaN at borders, normalized Grad-CAM values
    pred_map   : (H, W) ndarray with NaN at borders, prediction probabilities
    deposit_rc : (N, 2) known deposit pixel coordinates [(row, col), ...]
    smooth     : if True, apply Gaussian smoothing to cam_map before plotting
    """
    H, W = cam_map.shape
    data = cam_map.copy()

    if smooth and SMOOTH_SIGMA > 0:
        valid_mask = ~np.isnan(data)
        tmp      = np.where(valid_mask, data, 0.0)
        smoothed = gaussian_filter(tmp, sigma=SMOOTH_SIGMA)
        weight   = gaussian_filter(valid_mask.astype(np.float32), sigma=SMOOTH_SIGMA)
        with np.errstate(invalid='ignore', divide='ignore'):
            smoothed = np.where(weight > 0.01, smoothed / weight, np.nan)
        data = smoothed

    # ── Colorscale range (percentile clipping) ───────────────────────────────
    valid = data[~np.isnan(data)]
    vmin  = np.percentile(valid, VMIN_PERCENTILE)
    vmax  = np.percentile(valid, VMAX_PERCENTILE)

    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor='white')

    # ── Main image: Grad-CAM heatmap (jet colormap, anomaly-map style) ────────
    im = ax.imshow(
        data,
        cmap='jet',
        vmin=vmin, vmax=vmax,
        interpolation='bilinear',
        aspect='equal',
        origin='upper'
    )

    # ── Contour lines (key element of anomaly maps) ───────────────────────────
    data_clean   = np.where(np.isnan(data), vmin, data)
    contour_lvls = np.linspace(vmin, vmax, CONTOUR_LEVELS + 2)[1:-1]
    cs = ax.contour(
        data_clean,
        levels=contour_lvls,
        colors='black',
        linewidths=0.55,
        linestyles='-',
        alpha=0.55
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f', inline_spacing=3)

    # ── Prediction probability P=0.5 contour (deposit / non-deposit boundary) ─
    if pred_map is not None:
        pred_clean = np.where(np.isnan(pred_map), 0, pred_map)
        ax.contour(
            pred_clean,
            levels=[0.5],
            colors='white',
            linewidths=1.2,
            linestyles='--',
            alpha=0.85
        )

    # ── Deposit locations ────────────────────────────────────────────────────
    if len(deposit_rc) > 0:
        dep_rows, dep_cols = deposit_rc[:, 0], deposit_rc[:, 1]
        ax.scatter(
            dep_cols, dep_rows,
            marker='*', s=160,
            c='gold', edgecolors='black', linewidths=0.7,
            zorder=6, label='Known Li deposits'
        )
        ax.legend(
            loc='lower right', fontsize=10,
            framealpha=0.88, edgecolor='gray',
            handletextpad=0.4
        )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(
        'Grad-CAM intensity\n(normalized, DCN block2)',
        fontsize=10
    )
    cbar.ax.tick_params(labelsize=8)

    # ── Axis ticks and grid ───────────────────────────────────────────────────
    _apply_geo_ticks(ax, H, W, transform, crs)
    ax.grid(True, color='white', linewidth=0.4, linestyle='--', alpha=0.45)

    title_suffix = '(Gaussian smoothed)' if smooth else ''
    ax.set_title(
        f'Grad-CAM Anomaly Map  –  DCNv2 Block2  {title_suffix}\n'
        '(Class: Li deposits  |  Center-pixel activation × gradient)',
        fontsize=12, fontweight='bold', pad=8
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] Saved: {os.path.basename(save_path)}')


def plot_prediction_map(pred_map, deposit_rc, transform, crs, save_path):
    """Draw the full-area mineralization probability map (reference comparison)."""
    H, W = pred_map.shape
    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor='white')

    im = ax.imshow(
        pred_map,
        cmap='Reds',
        vmin=0, vmax=1,
        interpolation='bilinear',
        aspect='equal',
        origin='upper'
    )

    if len(deposit_rc) > 0:
        ax.scatter(
            deposit_rc[:, 1], deposit_rc[:, 0],
            marker='*', s=160, c='gold',
            edgecolors='black', linewidths=0.7,
            zorder=6, label='Known Li deposits'
        )
        ax.legend(loc='lower right', fontsize=10, framealpha=0.88, edgecolor='gray')

    pred_clean = np.where(np.isnan(pred_map), 0, pred_map)
    ax.contour(pred_clean, levels=[0.5],
               colors='steelblue', linewidths=1.0, linestyles='--', alpha=0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('P(Li deposit)', fontsize=11)
    cbar.ax.tick_params(labelsize=8)

    _apply_geo_ticks(ax, H, W, transform, crs)
    ax.grid(True, color='white', linewidth=0.4, linestyle='--', alpha=0.45)
    ax.set_title(
        'Mineral Prospectivity Map  –  Li deposits\n'
        '(Model prediction probability, DCN-Transformer)',
        fontsize=12, fontweight='bold', pad=8
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] Saved: {os.path.basename(save_path)}')


def plot_combined_map(cam_data, pred_map, deposit_rc,
                      transform, crs, save_path):
    """
    Draw a side-by-side comparison of the Grad-CAM anomaly map and the
    mineralization probability map.  cam_data should already be smoothed
    (computed once in main and passed in).
    """
    H, W = cam_data.shape

    valid = cam_data[~np.isnan(cam_data)]
    vmin  = np.percentile(valid, VMIN_PERCENTILE)
    vmax  = np.percentile(valid, VMAX_PERCENTILE)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), facecolor='white')

    # ── Left panel: Grad-CAM anomaly map ─────────────────────────────────────
    ax  = axes[0]
    im0 = ax.imshow(cam_data, cmap='jet', vmin=vmin, vmax=vmax,
                    interpolation='bilinear', aspect='equal', origin='upper')
    data_clean   = np.where(np.isnan(cam_data), vmin, cam_data)
    contour_lvls = np.linspace(vmin, vmax, CONTOUR_LEVELS + 2)[1:-1]
    cs0 = ax.contour(data_clean, levels=contour_lvls,
                     colors='black', linewidths=0.5, linestyles='-', alpha=0.5)
    ax.clabel(cs0, inline=True, fontsize=5.5, fmt='%.2f', inline_spacing=3)
    if pred_map is not None:
        pred_clean = np.where(np.isnan(pred_map), 0, pred_map)
        ax.contour(pred_clean, levels=[0.5], colors='white',
                   linewidths=1.0, linestyles='--', alpha=0.8)
    if len(deposit_rc) > 0:
        ax.scatter(deposit_rc[:, 1], deposit_rc[:, 0],
                   marker='*', s=130, c='gold', edgecolors='black',
                   linewidths=0.7, zorder=6, label='Known Li deposits')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.85)
    cbar0 = fig.colorbar(im0, ax=ax, fraction=0.035, pad=0.02)
    cbar0.set_label('Grad-CAM intensity', fontsize=9)
    cbar0.ax.tick_params(labelsize=7.5)
    _apply_geo_ticks(ax, H, W, transform, crs, n_lon=5, n_lat=5)
    ax.grid(True, color='white', linewidth=0.35, linestyle='--', alpha=0.4)
    ax.set_title('Grad-CAM Anomaly Map\n(DCNv2 Block2, Gaussian smoothed)',
                 fontsize=11, fontweight='bold', pad=6)

    # ── Right panel: mineralization probability map ───────────────────────────
    ax  = axes[1]
    im1 = ax.imshow(pred_map, cmap='Reds', vmin=0, vmax=1,
                    interpolation='bilinear', aspect='equal', origin='upper')
    if pred_map is not None:
        pred_clean = np.where(np.isnan(pred_map), 0, pred_map)
        ax.contour(pred_clean, levels=[0.5], colors='steelblue',
                   linewidths=1.0, linestyles='--', alpha=0.8)
    if len(deposit_rc) > 0:
        ax.scatter(deposit_rc[:, 1], deposit_rc[:, 0],
                   marker='*', s=130, c='gold', edgecolors='black',
                   linewidths=0.7, zorder=6, label='Known Li deposits')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.85)
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.035, pad=0.02)
    cbar1.set_label('P(Li deposit)', fontsize=9)
    cbar1.ax.tick_params(labelsize=7.5)
    _apply_geo_ticks(ax, H, W, transform, crs, n_lon=5, n_lat=5)
    ax.grid(True, color='white', linewidth=0.35, linestyle='--', alpha=0.4)
    ax.set_title('Mineral Prospectivity Map\n(DCN-Transformer prediction probability)',
                 fontsize=11, fontweight='bold', pad=6)

    fig.suptitle('Study Area Grad-CAM Anomaly  vs.  Prediction Probability  –  Li Deposits',
                 fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] Saved: {os.path.basename(save_path)}')


def main():
    print('=' * 72)
    print('Grad-CAM full-map visualization: dynamic convolution feature importance')
    print('=' * 72)

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    if device.type == 'cpu':
        print('  [Note] CPU mode is slow; a GPU environment is recommended.')

    # ── Step 1: Load model ────────────────────────────────────────────────────
    print('\n=== Step 1/8: Loading model ===')
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f'Model file not found: {MODEL_PATH}\n'
            'Please update MODEL_PATH at the top of this script.'
        )
    from attribution_guided_model import create_attribution_guided_model
    model = create_attribution_guided_model(
        num_classes=2,
        feature_dim=FEATURE_DIM,
        transformer_depth=TRANSFORMER_DEPTH,
        transformer_heads=TRANSFORMER_HEADS
    ).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  [OK] Model loaded (best val acc: {ckpt.get("val_acc", 0):.2f}%)')

    # ── Step 2: Load full-area data ───────────────────────────────────────────
    print('\n=== Step 2/8: Loading full-area data (Combined_array.npy) ===')
    if not os.path.exists(COMBINED_NPY):
        raise FileNotFoundError(
            f'Full-area data not found: {COMBINED_NPY}\n'
            'Please ensure Combined_array.npy has been generated.'
        )
    combined = np.load(COMBINED_NPY).astype(np.float32)
    H, W, C  = combined.shape
    print(f'  Shape: {combined.shape}  ({H} rows × {W} cols × {C} bands)')

    # ── Step 3: Standardize data (consistent with training) ───────────────────
    print('\n=== Step 3/8: Standardizing data ===')
    from sklearn.preprocessing import StandardScaler
    train_data    = np.load(TRAIN_DATA).astype(np.float32)
    scaler        = StandardScaler()
    scaler.fit(train_data.reshape(-1, C))
    combined_norm = scaler.transform(combined.reshape(-1, C)).reshape(H, W, C)
    del train_data
    print('  [OK] Standardization complete')

    # ── Step 4: Load deposit coordinates and geo info ─────────────────────────
    print('\n=== Step 4/8: Loading geo info and deposit coordinates ===')
    import cv2
    deposit_rc = np.empty((0, 2), dtype=int)
    if os.path.exists(LABEL_TIF):
        label1 = cv2.imread(LABEL_TIF, 2)
        if label1 is not None:
            raw_coords = np.array(list(zip(*np.where(label1 == 1))))
            kept, used = [], [False] * len(raw_coords)
            for i in range(len(raw_coords)):
                if used[i]:
                    continue
                kept.append(i)
                for j in range(i + 1, len(raw_coords)):
                    if not used[j] and \
                       np.linalg.norm(raw_coords[i] - raw_coords[j]) < MERGE_DIST:
                        used[j] = True
            deposit_rc = raw_coords[kept]
            print(f'  Deposit points: {len(deposit_rc)} (after de-duplication)')
    else:
        print(f'  [Warning] Label file not found: {LABEL_TIF} — deposits will not be shown')

    geo_transform, geo_crs, _, _ = load_geo_info(GEO_TIF_REF)
    if geo_transform is not None:
        print('  [OK] Geo info loaded successfully')
    else:
        print('  [Warning] Geo info unavailable — row/col indices will be used as axis labels')

    # ── Step 5: Compute full-map Grad-CAM ─────────────────────────────────────
    print(f'\n=== Step 5/8: Computing full-map Grad-CAM '
          f'({H * W:,} pixels, batch size {BATCH_SIZE}, '
          f'{WINDOW_HALF}-pixel border excluded) ===')
    cam_map, pred_map = compute_full_gradcam_map(model, combined_norm, device)
    valid_cnt = np.sum(~np.isnan(cam_map))
    print(f'  [OK] Computation complete (valid pixels: {valid_cnt:,})')

    np.save(os.path.join(SAVE_DIR, 'gradcam_map.npy'), cam_map)
    np.save(os.path.join(SAVE_DIR, 'pred_map.npy'),    pred_map)
    print(f'  [OK] Intermediate arrays saved to {SAVE_DIR}/')

    # ── Step 6: Pre-compute Gaussian smoothing ────────────────────────────────
    print('\n=== Step 6/8: Pre-computing Gaussian smoothing ===')
    cam_smooth = None
    if SMOOTH_SIGMA > 0:
        valid_mask = ~np.isnan(cam_map)
        tmp = np.where(valid_mask, cam_map, 0.0)
        sm  = gaussian_filter(tmp, sigma=SMOOTH_SIGMA)
        wt  = gaussian_filter(valid_mask.astype(np.float32), sigma=SMOOTH_SIGMA)
        with np.errstate(invalid='ignore', divide='ignore'):
            cam_smooth = np.where(wt > 0.01, sm / wt, np.nan)
        print(f'  [OK] Gaussian smooth applied (sigma={SMOOTH_SIGMA})')
    else:
        print('  Smoothing disabled (SMOOTH_SIGMA=0)')

    # ── Step 7: Export geo-referenced CSV ─────────────────────────────────────
    print('\n=== Step 7/8: Exporting geo-referenced CSV ===')
    csv_path = os.path.join(SAVE_DIR, 'gradcam_results.csv')
    export_to_csv(cam_map, pred_map, cam_smooth, csv_path)

    # ── Step 8: Generate visualization outputs ────────────────────────────────
    print('\n=== Step 8/8: Generating visualization outputs ===')

    plot_anomaly_map(
        cam_map, pred_map, deposit_rc,
        geo_transform, geo_crs,
        save_path=os.path.join(SAVE_DIR, 'gradcam_anomaly_map.png'),
        smooth=False
    )

    if cam_smooth is not None:
        plot_anomaly_map(
            cam_smooth, pred_map, deposit_rc,
            geo_transform, geo_crs,
            save_path=os.path.join(SAVE_DIR, 'gradcam_anomaly_map_smooth.png'),
            smooth=False   # data already smoothed; do not apply again
        )

    plot_prediction_map(
        pred_map, deposit_rc,
        geo_transform, geo_crs,
        save_path=os.path.join(SAVE_DIR, 'pred_probability_map.png')
    )

    plot_combined_map(
        cam_smooth if cam_smooth is not None else cam_map,
        pred_map, deposit_rc,
        geo_transform, geo_crs,
        save_path=os.path.join(SAVE_DIR, 'gradcam_combined.png')
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('All steps complete.')
    print('=' * 72)
    print(f'\nResults saved in: {SAVE_DIR}/')
    print('  gradcam_anomaly_map.png        : raw Grad-CAM anomaly map')
    if SMOOTH_SIGMA > 0:
        print('  gradcam_anomaly_map_smooth.png : Gaussian-smoothed anomaly map')
    print('  pred_probability_map.png       : model mineralization probability map')
    print('  gradcam_combined.png           : side-by-side anomaly + probability comparison')
    print('  gradcam_map.npy / pred_map.npy : intermediate arrays (for re-plotting)')
    print('  gradcam_results.csv            : geo coords + Grad-CAM + pred probability (for GIS)')
    print('\nCSV column descriptions:')
    print('  X             : UTM easting (m) or column index')
    print('  Y             : UTM northing (m) or row index')
    print('  row / col     : raster row/col number (0-indexed)')
    print('  gradcam       : normalized Grad-CAM value [0, 1] at center pixel')
    if SMOOTH_SIGMA > 0:
        print(f'  gradcam_smooth: Gaussian-smoothed Grad-CAM value (sigma={SMOOTH_SIGMA:.1f})')
    print('  pred_prob     : model probability of Li deposit [0, 1]')
    print('\nMap legend:')
    print('  Color (jet)       : blue = low Grad-CAM (low importance), '
          'red = high (high importance)')
    print('  Black contours    : Grad-CAM intensity isolines (anomaly lines)')
    print('  White dashed line : model prediction boundary P=0.5')
    print('  Gold stars        : known Li deposit locations')


if __name__ == '__main__':
    main()
