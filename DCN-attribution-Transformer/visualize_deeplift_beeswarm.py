"""This script visualizes DeepLIFT feature attributions with beeswarm plots."""

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

                                                                             
          
                                                                             
NANLING_DIR      = r'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling'
COMBINED_NPY     = os.path.join(NANLING_DIR, 'Combined_array.npy')
TRAIN_DATA       = os.path.join(NANLING_DIR, 'train_data.npy')
LABEL_TIF        = os.path.join(NANLING_DIR, 'New_data/label/Li_deposits0.tif')
NON_DEPOSIT_TIF  = os.path.join(NANLING_DIR, 'New_data/label/non_deposits.tif')
SAVE_DIR         = os.path.join(current_dir, 'attribution_full_map')

                     
N_BASELINE = 3000

MODEL_PATH = os.path.join(current_dir, 'best_attribution_guided_model.pth')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(current_dir),
                              'best_attribution_guided_model.pth')

               
FEATURE_DIM       = 32
TRANSFORMER_DEPTH = 1
TRANSFORMER_HEADS = 2

                        
BUFFER_RADIUS = 10          

                            
MERGE_DIST    = 5

                           
N_SAMPLES     = None

      
WINDOW_HALF  = 4
TARGET_CLASS = 1

        
                                                                    
IG_STEPS = 20                                  
IG_BATCH = 64                                                  

                                          
TOP_N_ELEM   = 8                                 
FORCE_ELEM   = ['Li', 'Be', 'Nb', 'W', 'Sn', 'Al₂O₃', 'Na₂O']                           

                                     
                                           
EXCLUDE_ELEM = ['Ni', 'Co']

            
GEO_NAMES = ['Granite', 'Faults', 'Strata']

                                                             
CHANNEL_NAMES = [
    'Ag',  'Al₂O₃','As',   'Au',   'B',    'Ba',   'Be',   'Bi',
    'CaO', 'Cd',   'Co',   'Cr',   'Cu',   'F',    'Fe₂O₃','Hg',
    'K₂O', 'La',   'Li',   'MgO',  'Mn',   'Mo',   'Na₂O', 'Nb',
    'Ni',  'P',    'Pb',   'Sb',   'SiO₂', 'Sn',   'Sr',   'Th',
    'Ti',  'U',    'V',    'W',    'Y',    'Zn',   'Zr',
    'Strata', 'Faults', 'Granite'                 
]
        
CH = {name: i for i, name in enumerate(CHANNEL_NAMES)}

                   
DISPLAY = {
    'Al₂O₃': 'Al$_2$O$_3$', 'Fe₂O₃': 'Fe$_2$O$_3$',
    'K₂O':   'K$_2$O',      'Na₂O':  'Na$_2$O',
    'SiO₂':  'SiO$_2$',
}


                                                                             
                
                                                                             

def beeswarm_y(x_vals, row_height=0.50, n_bins=120):
    n = len(x_vals)
    y = np.zeros(n)
    if n == 0 or x_vals.max() == x_vals.min():
        return y
    edges   = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(x_vals, edges, right=True), 0, n_bins - 1)
    rng     = np.random.default_rng(seed=0)
    for b in np.unique(bin_idx):
        idx = np.where(bin_idx == b)[0]
        k   = len(idx)
        if k == 1:
            continue
        offsets = np.linspace(-row_height / 2, row_height / 2, k)
        rng.shuffle(offsets)
        y[idx] = offsets
    return y


                                                                             
                           
                                                                             

def compute_ig_deposit(model, combined_norm, deposit_rows, deposit_cols,
                       device, baseline_window):
    N = len(deposit_rows)
    combined_raw = np.load(COMBINED_NPY).astype(np.float32)               

    attr_matrix = np.zeros((N, 42), dtype=np.float32)
    orig_matrix = combined_raw[deposit_rows, deposit_cols, :]           

    half      = WINDOW_HALF
    n_batches = (N + IG_BATCH - 1) // IG_BATCH
    S         = IG_STEPS

                                   
    base_t = torch.FloatTensor(baseline_window).to(device)                 

                                        
    alphas = torch.linspace(1.0 / S, 1.0, S, device=device)         

    model.eval()
    for b in tqdm(range(n_batches), desc=f'Integrated Gradients S={S}(deposit area)', ncols=72):
        s = b * IG_BATCH
        e = min(s + IG_BATCH, N)
        brows = deposit_rows[s:e]
        bcols = deposit_cols[s:e]
        bsz   = e - s

                                    
        windows = np.stack([
            combined_norm[r - half: r + half + 1,
                          c - half: c + half + 1, :]
            for r, c in zip(brows, bcols)
        ])                                                                 
        inp = (torch.FloatTensor(windows)
               .permute(0, 3, 1, 2)                                       
               .to(device))

                        
        baseline = base_t.unsqueeze(0).expand(bsz, -1, -1, -1)                   
        delta    = inp - baseline                                                   

                                                       
        interp = (baseline.unsqueeze(1)
                  + alphas.view(1, S, 1, 1, 1) * delta.unsqueeze(1))
        interp = interp.reshape(bsz * S, 42, 9, 9).requires_grad_(True)

        output = model(interp)
        model.zero_grad()
        F.softmax(output, dim=1)[:, TARGET_CLASS].sum().backward()

                                      
        grad     = interp.grad.detach().view(bsz, S, 42, 9, 9)
        avg_grad = grad.mean(dim=1)                                         
        ig_vals  = (avg_grad * delta)[:, :, half, half]               

        attr_matrix[s:e] = ig_vals.cpu().numpy()

    return attr_matrix, orig_matrix


                                                                             
      
                                                                             

def main():
    print('=' * 65)
    print(f'DeepLIFT Beeswarm — deposit buffer（radius {BUFFER_RADIUS} pixels）')
    print('=' * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice in Use: {device}')

                                                      
    print(f'\n[1/5] Reading original deposit coordinates (Li_deposits0.tif)...')
    import cv2
    label1     = cv2.imread(LABEL_TIF, 2)
    raw_coords = np.array(list(zip(*np.where(label1 == 1))))            

                                 
    kept, used = [], [False] * len(raw_coords)
    for i in range(len(raw_coords)):
        if used[i]:
            continue
        kept.append(i)
        for j in range(i + 1, len(raw_coords)):
            if not used[j]:
                if np.linalg.norm(raw_coords[i] - raw_coords[j]) < MERGE_DIST:
                    used[j] = True
    deposit_rc = raw_coords[kept]            
    print(f'  Original labeled pixels: {len(raw_coords)}  →  deduplicated deposits: {len(deposit_rc)} ')

                                                   
    H, W = label1.shape
    half = WINDOW_HALF
    buf_set = set()
    for (dr, dc) in deposit_rc:
        for r in range(max(half, dr - BUFFER_RADIUS),
                       min(H - half, dr + BUFFER_RADIUS + 1)):
            for c in range(max(half, dc - BUFFER_RADIUS),
                           min(W - half, dc + BUFFER_RADIUS + 1)):
                if (r - dr) ** 2 + (c - dc) ** 2 <= BUFFER_RADIUS ** 2:
                    buf_set.add((r, c))

    buf_coords = np.array(sorted(buf_set))               
    print(f'  （{BUFFER_RADIUS}  valid grid cells within pixel radius: {len(buf_coords):,} ')

                              
    np.random.seed(42)
    if N_SAMPLES is not None and len(buf_coords) > N_SAMPLES:
        idx = np.random.choice(len(buf_coords), N_SAMPLES, replace=False)
        buf_coords = buf_coords[idx]
    print(f'  Used this run: {len(buf_coords):,} points（all buffer grid cells）')

    dep_rows = buf_coords[:, 0].astype(int)
    dep_cols = buf_coords[:, 1].astype(int)

                                                                         
    print('\n[2/5] Model...')
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

                                                                        
    print('\n[3/5] ...')
    combined_raw  = np.load(COMBINED_NPY).astype(np.float32)               
    H, W, C = combined_raw.shape
    train_data    = np.load(TRAIN_DATA)
    scaler        = StandardScaler()
    scaler.fit(train_data.reshape(-1, C))
    combined_norm = scaler.transform(combined_raw.reshape(-1, C)).reshape(H, W, C)
    print('  [OK]')

                                                                      
    print(f'\n[3b/5] Building non-mineral baseline (N_BASELINE={N_BASELINE}）...')
    import cv2 as _cv2
    half = WINDOW_HALF

    non_dep_tif = NON_DEPOSIT_TIF
    if os.path.exists(non_dep_tif):
        non_label = _cv2.imread(non_dep_tif, 2)
        non_rows_all, non_cols_all = np.where(non_label > 0)
    else:
                                     
        print(f'  [Warning] Could not find {non_dep_tif}，using all non-mineral grid cells in the full map as fallback')
        deposit_set = set(zip(dep_rows.tolist(), dep_cols.tolist()))
        all_r, all_c = np.meshgrid(
            np.arange(half, H - half), np.arange(half, W - half), indexing='ij'
        )
        all_r = all_r.ravel(); all_c = all_c.ravel()
        mask  = np.array([(r, c) not in deposit_set
                          for r, c in zip(all_r, all_c)])
        non_rows_all = all_r[mask]; non_cols_all = all_c[mask]

                             
    valid = ((non_rows_all >= half) & (non_rows_all < H - half) &
             (non_cols_all >= half) & (non_cols_all < W - half))
    non_rows_all = non_rows_all[valid]
    non_cols_all = non_cols_all[valid]

          
    np.random.seed(0)
    n_avail = len(non_rows_all)
    n_sample = min(N_BASELINE, n_avail)
    idx_bl = np.random.choice(n_avail, n_sample, replace=False)
    bl_rows = non_rows_all[idx_bl]
    bl_cols = non_cols_all[idx_bl]
    print(f'  Available non-mineral grid cells: {n_avail:,}  →  sampled: {n_sample:,} ')

                                 
    win_size = 2 * half + 1
    bl_windows = np.stack([
        combined_norm[r - half: r + half + 1,
                      c - half: c + half + 1, :]
        for r, c in zip(bl_rows, bl_cols)
    ])                                                               
    baseline_window = bl_windows.mean(axis=0).transpose(2, 0, 1)              
    print(f'  [OK] Baseline window shape: {baseline_window.shape}  '
          f'Center-pixel means (first 5 channels): '
          f'{baseline_window[:5, half, half].round(3)}')

                                                                 
    print(f'\n[4/5] Computing full-channel Integrated Gradients attribution (IG_STEPS={IG_STEPS}，'
          f'baseline = mean non-mineral window)...')
    attr_mat, orig_mat = compute_ig_deposit(
        model, combined_norm, dep_rows, dep_cols, device, baseline_window
    )
                                          

                                                                 
    print('\n[5/5] Selecting features and plotting...')

                                            
    elem_mean_attr = {
        CHANNEL_NAMES[i]: attr_mat[:, i].mean()
        for i in range(39)
    }
                                     
    pos_elem_sorted = sorted(
        [e for e in elem_mean_attr
         if elem_mean_attr[e] > 0 and e not in EXCLUDE_ELEM],
        key=elem_mean_attr.get, reverse=True
    )

                                                            
    selected_elem = [e for e in FORCE_ELEM
                     if elem_mean_attr.get(e, 0) > 0 and e not in EXCLUDE_ELEM]
    for e in pos_elem_sorted:
        if len(selected_elem) >= TOP_N_ELEM:
            break
        if e not in selected_elem:
            selected_elem.append(e)

                  
    geo_importance = {
        name: attr_mat[:, CH[name]].mean()
        for name in GEO_NAMES
    }
    elem_importance = elem_mean_attr                 
    all_importance  = {**elem_importance, **geo_importance}

    all_features = selected_elem + GEO_NAMES
    all_sorted   = sorted(all_features,
                          key=lambda k: all_importance.get(k, 0), reverse=True)
    plot_order   = list(reversed(all_sorted))                       

    print('  Positive-driving element ranking (mean attribution > 0, sorted descending after geological filtering):')
    print(f'  Geological exclusion list: {EXCLUDE_ELEM}')
    print(f'  Available positive elements ({len(pos_elem_sorted)}  total): {pos_elem_sorted}')
    for rank, name in enumerate(all_sorted, 1):
        tag = '★ Priority' if name in FORCE_ELEM else\
              ('● Ore-control' if name in GEO_NAMES else '○ Element')
        sign = '↑Promotes' if all_importance.get(name, 0) > 0 else '↓Suppresses'
        print(f'    {rank}. {DISPLAY.get(name, name):12s}  '
              f'mean_attr={all_importance.get(name,0):+.6f}  {sign}  {tag}')

                                                             
                                                   
    attr_norm = attr_mat.copy()
    norm_scale = {}
    for name in all_features:
        ch  = CH[name]
        v99 = np.percentile(np.abs(attr_mat[:, ch]), 99)
        if v99 > 0:
            attr_norm[:, ch] = np.clip(attr_mat[:, ch] / v99, -1, 1)
        norm_scale[name] = v99
    print('\n  Normalization scale（99th pct of |attr|）:')
    for name in all_sorted:
        print(f'    {DISPLAY.get(name, name):12s}  scale={norm_scale[name]:.2e}')

                                                                         
    n_feat = len(plot_order)
    fig, ax = plt.subplots(figsize=(8.0, 0.72 * n_feat + 2.0),
                           facecolor='white')
    cmap = plt.get_cmap('RdBu_r')

                               
    x_bound = 1.08

    for feat_i, name in enumerate(plot_order):
        ch      = CH[name]
        x_vals  = attr_norm[:, ch].astype(float)                 
        raw_val = orig_mat[:, ch].astype(float)

                    
        v_lo, v_hi = np.percentile(raw_val, 1), np.percentile(raw_val, 99)
        norm_c = np.clip((raw_val - v_lo) / (v_hi - v_lo + 1e-9), 0, 1)
        colors = cmap(norm_c)

                       
        y_off = beeswarm_y(x_vals, row_height=0.52)
        ax.scatter(
            x_vals, feat_i + y_off,
            c=colors, s=7, alpha=0.70,
            linewidths=0, rasterized=True, zorder=2
        )

        
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(
        [DISPLAY.get(f, f) for f in plot_order], fontsize=12.5
    )
    ax.axvline(0, color='#666666', linewidth=0.9, zorder=1)
    ax.set_xlim(-x_bound, x_bound)
    ax.set_ylim(-0.65, n_feat - 0.35)
    ax.set_xlabel('Attribution value  (Integrated Gradients,  normalized to [−1, 1])',
                  fontsize=11.5, labelpad=6)
    ax.set_facecolor('#f4f4f4')
    ax.grid(True, axis='x', color='white', linewidth=0.8, zorder=0)
    ax.tick_params(axis='y', length=0, labelsize=12)
    ax.tick_params(axis='x', labelsize=10)
    for sp in ['top', 'right', 'left', 'bottom']:
        ax.spines[sp].set_visible(False)
    for i in range(n_feat):
        ax.axhline(i - 0.5, color='white', linewidth=1.2, zorder=1)

         
    sm   = ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.015, aspect=22)
    cbar.set_ticks([0.02, 0.98])
    cbar.set_ticklabels(['Low', 'High'], fontsize=10)
    cbar.ax.set_ylabel('Feature value', fontsize=10, labelpad=4)
    cbar.ax.yaxis.set_label_position('right')
    cbar.outline.set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'deeplift_beeswarm_deposit.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\n[OK] Saved: {save_path}')

                                    
    csv_out = os.path.join(SAVE_DIR, 'attribution_deposit_area.csv')
    cols = {'row': dep_rows, 'col': dep_cols}
    for name in all_features:
        cols[f'attr_{name}']      = attr_mat [:, CH[name]].round(8)
        cols[f'attr_norm_{name}'] = attr_norm[:, CH[name]].round(6)
    pd.DataFrame(cols).to_csv(csv_out, index=False, encoding='utf-8-sig')
    print(f'[OK] Deposit-buffer attribution CSV: {csv_out}')

    print('\nNotes:')
    print(f'  · Attribution method: Integrated Gradients, steps S={IG_STEPS}')
    print(f'  ·  x\': sampled {n_sample:,}  randomly sampled grid cells in the non-mineral area')
    print(f'  · Sampling range: in the original Li_deposits0.tif, {len(deposit_rc)}  deposits, '
          f'{BUFFER_RADIUS} all grid cells within the pixel buffer')
    print(f'  · Element selection: mean attribution > 0 (positive-driving) and not in the geological exclusion list; keep the top {TOP_N_ELEM} ；'
          f'Li/Be/Nb kept')
    print(f'  · Geological exclusion: {EXCLUDE_ELEM}（，，Granite）')
    print(f'  · Selected elements: {selected_elem}')
    print(f'  · Ore-controlling factors: {GEO_NAMES}(forced inclusion)')
    print('  · Normalization: independent per feature, divide by the 99th percentile absolute value, and clip to [-1,1]')
    print('  · Red = high feature value, blue = low feature value; right-skewed = promotes mineralization, left-skewed = suppresses mineralization')
    print('=' * 65)


if __name__ == '__main__':
    main()
