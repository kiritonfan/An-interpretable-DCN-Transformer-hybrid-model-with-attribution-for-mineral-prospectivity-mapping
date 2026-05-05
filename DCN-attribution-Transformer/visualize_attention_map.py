"""This script visualizes attention maps produced by the trained model."""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8', errors='replace')

                                                                             
          
                                                                             
NANLING_DIR     = r'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling'
COMBINED_NPY    = os.path.join(NANLING_DIR, 'Combined_array.npy')
TRAIN_DATA      = os.path.join(NANLING_DIR, 'train_data.npy')
LABEL_TIF       = os.path.join(NANLING_DIR, 'New_data/label/Li_deposits0.tif')
                   
XX_TIF          = os.path.join(NANLING_DIR, 'New_data/coordinate/XX.tif')
YY_TIF          = os.path.join(NANLING_DIR, 'New_data/coordinate/YY.tif')
SAVE_DIR        = os.path.join(current_dir, 'attention_map')
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(current_dir, 'best_attribution_guided_model.pth')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(current_dir),
                              'best_attribution_guided_model.pth')

               
FEATURE_DIM       = 32
TRANSFORMER_DEPTH = 1
TRANSFORMER_HEADS = 2

      
WINDOW_HALF  = 4             

                     
BATCH_SIZE = 256

                       
DEPOSIT_BUFFER_RADIUS = 10               
MERGE_DIST            = 5                

                                        
PROCESS_FULL_MAP = True

                                                                    
                                                
                                                    
CENTER_TOKEN_IDX = 4 * 9 + 4 + 1         


                                                                             
       
                                                                             

def extract_attention(model, windows_t, device):
    windows_t = windows_t.to(device)
    with torch.no_grad():
        logits, attn_maps = model(windows_t, return_attention=True)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy().astype(np.float32)
                                             
    attn = attn_maps[0].cpu().numpy().astype(np.float32)                      
    return attn, probs


def aggregate_attention(attn):
                                    
    cls_row = attn[:, :, 0, :]                          
    cls_to_spatial = cls_row[:, :, 1:]                              

                                                       
    center_idx = CENTER_TOKEN_IDX - 1                              

    results = {
                                 
        'attn_cls_center': cls_to_spatial[:, :, center_idx].mean(axis=1),
                          
        'attn_cls_mean':   cls_to_spatial.mean(axis=(1, 2)),
                                     
        'attn_cls_max':    cls_to_spatial.mean(axis=1).max(axis=1),
                                      
        'attn_center_cls': attn[:, :, CENTER_TOKEN_IDX, 0].mean(axis=1),
    }

                        
    num_heads = attn.shape[1]
    for h in range(num_heads):
        results[f'attn_head{h}_cls_center'] = cls_to_spatial[:, h, center_idx]

    return results


def load_geo_coords():
    from PIL import Image
    try:
        XX = np.array(Image.open(XX_TIF)).astype(np.float64)
        YY = np.array(Image.open(YY_TIF)).astype(np.float64)
        print(f'  [OK] Geographic coordinate rasters loaded: XX{XX.shape}, YY{YY.shape}')
        print(f'       X range: [{XX.min():.2f}, {XX.max():.2f}]')
        print(f'       Y range: [{YY.min():.2f}, {YY.max():.2f}]')
        return XX, YY
    except Exception as e:
        print(f'  [Warning] Could not load geographic coordinate files; using pixel coordinates: {e}')
        return None, None


def run_inference(model, combined_norm, rows, cols, device, XX=None, YY=None):
    N = len(rows)
    half = WINDOW_HALF

             
    all_probs   = np.zeros(N, dtype=np.float32)
    all_metrics = {}                              

    n_batches = (N + BATCH_SIZE - 1) // BATCH_SIZE
    model.eval()

    for b in tqdm(range(n_batches), desc='Attention inference', ncols=72):
        s = b * BATCH_SIZE
        e = min(s + BATCH_SIZE, N)
        brows, bcols = rows[s:e], cols[s:e]

                                                     
        windows = np.stack([
            combined_norm[r - half: r + half + 1,
                          c - half: c + half + 1, :]
            for r, c in zip(brows, bcols)
        ])
        inp = torch.FloatTensor(windows).permute(0, 3, 1, 2)

        attn, probs = extract_attention(model, inp, device)
        metrics     = aggregate_attention(attn)

        all_probs[s:e] = probs
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = np.zeros(N, dtype=np.float32)
            all_metrics[k][s:e] = v

            
    if XX is not None and YY is not None:
        x_vals = XX[rows, cols].astype(np.float64)
        y_vals = YY[rows, cols].astype(np.float64)
        coord_note = 'Geographic coordinates (from XX.tif / YY.tif)'
    else:
        x_vals = cols.astype(np.float64)
        y_vals = rows.astype(np.float64)
        coord_note = 'Pixel coordinates (x=col, y=row)'
    print(f'  Coordinate type: {coord_note}')

                  
    df = pd.DataFrame({
        'row':      rows,
        'col':      cols,
        'x':        x_vals,
        'y':        y_vals,
        'pred_prob': all_probs,
    })
    for k, v in all_metrics.items():
        df[k] = v

    return df


                                                                             
        
                                                                             

def plot_spatial_map(data_grid, title, save_path, cmap='hot', label=None,
                     deposit_rc=None):
    fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
    im = ax.imshow(data_grid, cmap=cmap, aspect='equal',
                   origin='upper', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label(label or title, fontsize=10)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel('Column (X)', fontsize=10)
    ax.set_ylabel('Row (Y)', fontsize=10)

    if deposit_rc is not None:
        ax.scatter(deposit_rc[:, 1], deposit_rc[:, 0],
                   c='cyan', s=40, marker='*', zorder=5,
                   label='Known deposits', edgecolors='navy', linewidths=0.5)
        ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] Saved: {save_path}')


def plot_head_comparison(df, H, W, num_heads, deposit_rc, save_path):
    cols_list = [f'attn_head{h}_cls_center' for h in range(num_heads)]
    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 5.5),
                             facecolor='white')
    if num_heads == 1:
        axes = [axes]

    for h, (ax, col) in enumerate(zip(axes, cols_list)):
        grid = np.full((H, W), np.nan, dtype=np.float32)
        grid[df['row'].values, df['col'].values] = df[col].values
        im = ax.imshow(grid, cmap='hot', aspect='equal',
                       origin='upper', interpolation='nearest')
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
        ax.set_title(f'Head {h}  —  CLS→Center Attention', fontsize=11)
        ax.set_xlabel('Column (X)', fontsize=9)
        ax.set_ylabel('Row (Y)', fontsize=9)
        if deposit_rc is not None:
            ax.scatter(deposit_rc[:, 1], deposit_rc[:, 0],
                       c='cyan', s=30, marker='*', zorder=5,
                       edgecolors='navy', linewidths=0.5)

    plt.suptitle('Per-Head Transformer Attention  (CLS → Center Patch)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] Saved: {save_path}')


def plot_deposit_scatter(df_dep, metric, xlabel, save_path):
    vals = df_dep[metric].values
    fig, ax = plt.subplots(figsize=(7, 3), facecolor='white')
    ax.scatter(vals, np.random.uniform(-0.4, 0.4, len(vals)),
               c=vals, cmap='hot', s=8, alpha=0.6,
               linewidths=0, rasterized=True)
    ax.axvline(vals.mean(), color='#2266cc', linewidth=1.5,
               linestyle='--', label=f'Mean={vals.mean():.4f}')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title('Transformer Attention — Deposit Buffer Area', fontsize=12)
    ax.set_yticks([])
    ax.legend(fontsize=9)
    ax.set_facecolor('#f4f4f4')
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK] Saved: {save_path}')


                                                                             
      
                                                                             

def main():
    print('=' * 65)
    print('Spatial visualization of Transformer attention weights')
    print('=' * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice in Use: {device}')

                                                                         
    print('\n[1/5] Loading model...')
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
    print(f'  [OK] Validation accuracy: {ckpt.get("val_acc", 0):.2f}%')
    print(f'  Transformer layers: {TRANSFORMER_DEPTH}, attention heads: {TRANSFORMER_HEADS}')
    print(f'  Attention matrix shape: (B, {TRANSFORMER_HEADS}, 82, 82)')
    print(f'  CLS token: index 0, center patch: index {CENTER_TOKEN_IDX}')

                                                                        
    print('\n[2/5] Standardizing data...')
    combined_raw  = np.load(COMBINED_NPY).astype(np.float32)               
    H, W, C = combined_raw.shape
    train_data    = np.load(TRAIN_DATA)
    scaler        = StandardScaler()
    scaler.fit(train_data.reshape(-1, C))
    combined_norm = scaler.transform(combined_raw.reshape(-1, C)).reshape(H, W, C)
    print(f'  [OK] Full image size: {H} × {W}, channels: {C}')

                                                                       
    print('\n[3/5] Determining processing extent...')
    import cv2
    label1 = cv2.imread(LABEL_TIF, 2)
    half   = WINDOW_HALF

                
    raw_coords = np.array(list(zip(*np.where(label1 == 1))))
    kept, used = [], [False] * len(raw_coords)
    for i in range(len(raw_coords)):
        if used[i]:
            continue
        kept.append(i)
        for j in range(i + 1, len(raw_coords)):
            if not used[j] and np.linalg.norm(raw_coords[i] - raw_coords[j]) < MERGE_DIST:
                used[j] = True
    deposit_rc = raw_coords[kept]
    print(f'  Number of deposits: {len(deposit_rc)} (deduplicated)')

    if PROCESS_FULL_MAP:
                            
        all_r, all_c = np.meshgrid(
            np.arange(half, H - half),
            np.arange(half, W - half),
            indexing='ij'
        )
        proc_rows = all_r.ravel().astype(int)
        proc_cols = all_c.ravel().astype(int)
        print(f'  Valid grid cells in full map: {len(proc_rows):,} ')
    else:
                
        buf_set = set()
        for (dr, dc) in deposit_rc:
            for r in range(max(half, dr - DEPOSIT_BUFFER_RADIUS),
                           min(H - half, dr + DEPOSIT_BUFFER_RADIUS + 1)):
                for c in range(max(half, dc - DEPOSIT_BUFFER_RADIUS),
                               min(W - half, dc + DEPOSIT_BUFFER_RADIUS + 1)):
                    if (r - dr)**2 + (c - dc)**2 <= DEPOSIT_BUFFER_RADIUS**2:
                        buf_set.add((r, c))
        buf_coords = np.array(sorted(buf_set))
        proc_rows  = buf_coords[:, 0].astype(int)
        proc_cols  = buf_coords[:, 1].astype(int)
        print(f'  Deposit buffer grid cells: {len(proc_rows):,} ')

                                                                       
    print('\n[3b/5] Loading geographic coordinate rasters...')
    XX, YY = load_geo_coords()

                                                                   
    print(f'\n[4/5] Batch inference (batch_size={BATCH_SIZE}, no gradients)...')
    df = run_inference(model, combined_norm, proc_rows, proc_cols, device, XX, YY)

    print(f'\n  Attention metric statistics (all {len(df):,} grid cells):')
    for col in ['attn_cls_center', 'attn_cls_mean', 'attn_cls_max',
                'attn_center_cls']:
        print(f'    {col:25s}  '
              f'mean={df[col].mean():.5f}  '
              f'std={df[col].std():.5f}  '
              f'[{df[col].min():.5f}, {df[col].max():.5f}]')

                                                                           
    csv_path = os.path.join(SAVE_DIR, 'attention_map.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig', float_format='%.6f')
    print(f'\n  [OK] CSV saved: {csv_path}')
    print(f'       Column names: {list(df.columns)}')
    print(f'       Rows: {len(df):,}(each row corresponds to a grid cell)')

                                                                         
    print('\n[5/5] Spatial visualization...')

          
    rows_arr = df['row'].values
    cols_arr = df['col'].values

    def make_grid(col_name):
        g = np.full((H, W), np.nan, dtype=np.float32)
        g[rows_arr, cols_arr] = df[col_name].values
        return g

                        
    plot_spatial_map(
        make_grid('attn_cls_center'),
        title='Transformer Attention: CLS → Center Patch',
        save_path=os.path.join(SAVE_DIR, 'attention_cls_center_map.png'),
        cmap='hot',
        label='Attention Weight (mean over heads)',
        deposit_rc=deposit_rc
    )

                           
    plot_spatial_map(
        make_grid('attn_cls_mean'),
        title='Transformer Attention: CLS → Spatial Mean',
        save_path=os.path.join(SAVE_DIR, 'attention_cls_mean_map.png'),
        cmap='YlOrRd',
        label='Mean Spatial Attention Weight',
        deposit_rc=deposit_rc
    )

                    
    plot_spatial_map(
        make_grid('pred_prob'),
        title='Model Prediction Probability (Class=Deposit)',
        save_path=os.path.join(SAVE_DIR, 'attention_pred_prob_map.png'),
        cmap='RdYlGn',
        label='Prediction Probability',
        deposit_rc=deposit_rc
    )

                 
    plot_head_comparison(
        df, H, W, TRANSFORMER_HEADS, deposit_rc,
        save_path=os.path.join(SAVE_DIR, 'attention_head_comparison.png')
    )

                                
    if PROCESS_FULL_MAP and len(deposit_rc) > 0:
        buf_mask = np.zeros(len(df), dtype=bool)
        for (dr, dc) in deposit_rc:
            dist2 = (rows_arr - dr)**2 + (cols_arr - dc)**2
            buf_mask |= (dist2 <= DEPOSIT_BUFFER_RADIUS**2)
        df_dep = df[buf_mask]
        print(f'\n  Deposit buffer grid count (for scatter plots): {buf_mask.sum():,}')
        plot_deposit_scatter(
            df_dep, 'attn_cls_center',
            xlabel='Attention Weight  (CLS → Center Patch,  mean over heads)',
            save_path=os.path.join(SAVE_DIR, 'deposit_attention_scatter.png')
        )

    print('\n' + '=' * 65)
    print('Notes:')
    print(f'  · Model: FEATURE_DIM={FEATURE_DIM}, DEPTH={TRANSFORMER_DEPTH}, '
          f'HEADS={TRANSFORMER_HEADS}')
    print(f'  · Attention matrix: (B, {TRANSFORMER_HEADS}, 82, 82)  '
          f'[82 = 81 spatial patches + 1 CLS token]')
    print(f'  · Center patch sequence index: {CENTER_TOKEN_IDX}  '
          f'(corresponding to the center of the 9×9 window at (4,4))')
    print(f'  · Main metric attn_cls_center = attention weight from the CLS token to the center patch')
    print(f'    → High values: the Transformer considers this location more important for classification')
    coord_src = 'XX.tif / YY.tif (geographic coordinates)' if XX is not None else 'Pixel coordinates (x=col, y=row)'
    print(f'  · CSV coordinate notes: x/y  {coord_src}')
    print(f'  · Output directory: {SAVE_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()
