"""This script visualizes integrated gradients deposit-level histograms for the study."""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

                                                                 
plt.rcParams.update({
    'font.family'        : 'Times New Roman',
    'mathtext.fontset'   : 'stix',                                        
    'axes.unicode_minus' : False,
})

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

                                                                                 
          
                                                                                 
NANLING_DIR  = r'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling'
COMBINED_NPY = os.path.join(NANLING_DIR, 'Combined_array.npy')
TRAIN_DATA   = os.path.join(NANLING_DIR, 'train_data.npy')
LABEL_TIF    = os.path.join(NANLING_DIR, 'New_data/label/Li_deposits0.tif')

MODEL_PATH = os.path.join(current_dir, 'best_attribution_guided_model.pth')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(current_dir),
                              'best_attribution_guided_model.pth')

SAVE_DIR = os.path.join(current_dir, 'ig_deposit_histograms')

                
FEATURE_DIM       = 32
TRANSFORMER_DEPTH = 1
TRANSFORMER_HEADS = 2

        
IG_STEPS         = 200                               
TARGET_CLASS     = 1             
WINDOW_HALF      = 4                   

                  
                                  
                                                    
SMOOTH_N         = 20                         
SMOOTH_NOISE_STD = 0.15         

          
                                          
                                     
SPATIAL_HALF     = 1

           
NORM_PERCENTILE = 99                 
NORM_SCALE      = 1.0                             

                                                                
CHANNEL_NAMES = [
    'Ag',  'Al₂O₃','As',   'Au',   'B',    'Ba',   'Be',   'Bi',
    'CaO', 'Cd',   'Co',   'Cr',   'Cu',   'F',    'Fe₂O₃','Hg',
    'K₂O', 'La',   'Li',   'MgO',  'Mn',   'Mo',   'Na₂O', 'Nb',
    'Ni',  'P',    'Pb',   'Sb',   'SiO₂', 'Sn',   'Sr',   'Th',
    'Ti',  'U',    'V',    'W',    'Y',    'Zn',   'Zr',
    'Strata', 'Faults', 'Granite'
]

                                                                     
                                                                   
            
DISPLAY_NAMES = {
    18: 'Li',
    6:  'Be',
    23: 'Nb',
    13: 'F',
    29: 'Sn',
    35: 'W',
    16: r'$\mathrm{K_2O}$',
    1:  r'$\mathrm{Al_2O_3}$',
    28: r'$\mathrm{SiO_2}$',
    22: r'$\mathrm{Na_2O}$',
    5:  'Ba',
    30: 'Sr',
    31: 'Th',
    33: 'U',
    17: 'La',
    38: 'Zr',
    39: 'Strata',        
    40: 'Faults',        
    41: 'Granite',        
}

                                                            
                                       
SELECTED_CHANNELS = [
    18,                     
    6,                       
    23,                             
    13,                            
    29,                                
    35,                                
    16,                           
    1,                                 
    28,                       
    22,                        
    5,                               
    30,                            
    31,                           
    33,                           
    17,                               
    38,                              
    39,                    
    40,                    
    41,                   
]

                                                                
                                  
CHANNEL_COLORS = {
    18: '#B22222',                    
    6:  '#FF6347',                    
    23: '#FF8C00',                  
    13: '#FFA500',                 
    29: '#DAA520',                  
    35: '#FFD700',                  
    16: '#4169E1',                  
    1:  '#6495ED',                  
    28: '#87CEEB',                  
    22: '#ADD8E6',                  
    5:  '#9370DB',                  
    30: '#BA55D3',                  
    31: '#808080',                 
    33: '#A9A9A9',                 
    17: '#20B2AA',                
    38: '#3CB371',                
    39: '#2E8B57',                
    40: '#006400',                
    41: '#228B22',               
}

                
GROUP_LABELS = {
    'Direct ore-forming elements':  [18],
    'Associated / volatile elements': [6, 23, 13, 29, 35],
    'Geochemical indices':  [16, 1, 28, 22],
    'Lithophile / radioactive / rare-earth elements': [5, 30, 31, 33, 17, 38],
    'Ore-controlling factors':     [39, 40, 41],
}


                                                                                 
                                                 
                                                                                 

def integrated_gradients_single(model, inp, baseline, n_steps, target_class,
                                half, spatial_half=0):
    device = inp.device

    with torch.no_grad():
        out_real  = model(inp)
        pred_prob = float(F.softmax(out_real, dim=1)[0, target_class].item())

                                                         
    alphas      = torch.linspace(0.0, 1.0, n_steps, device=device
                                 ).view(n_steps, 1, 1, 1, 1)
    delta       = inp.unsqueeze(0) - baseline.unsqueeze(0)                       
    interp_flat = (baseline.unsqueeze(0) + alphas * delta
                   ).squeeze(1).detach().requires_grad_(True)                  

    out   = model(interp_flat)
    probs = F.softmax(out, dim=1)
    model.zero_grad()
    probs[:, target_class].sum().backward()

    grads = interp_flat.grad.detach()                                          

            
    weights    = torch.ones(n_steps, device=device)
    weights[0] = weights[-1] = 0.5
    weights    = weights.view(n_steps, 1, 1, 1) / (n_steps - 1)
    avg_grads  = (grads * weights).sum(dim=0)                                

                                                             
    r0, r1 = half - spatial_half, half + spatial_half + 1
    c0, c1 = half - spatial_half, half + spatial_half + 1

                                           
    delta_region = (inp[0] - baseline[0])[:, r0:r1, c0:c1]                  
    ig_region_map = delta_region * avg_grads[:, r0:r1, c0:c1]               
    ig_region     = ig_region_map.mean(dim=(1, 2))                        

    return ig_region.cpu().numpy(), pred_prob


def smoothgrad_ig(model, inp, baseline, n_steps, target_class, half,
                  spatial_half=SPATIAL_HALF,
                  n_smooth=SMOOTH_N,
                  noise_std=SMOOTH_NOISE_STD):
                       
    with torch.no_grad():
        out_real  = model(inp)
        pred_prob = float(F.softmax(out_real, dim=1)[0, target_class].item())

    C         = inp.shape[1]
    ig_accum  = np.zeros(C, dtype=np.float64)

    for _ in range(n_smooth):
        noise     = torch.randn_like(inp) * noise_std
        inp_noisy = inp + noise                                 
        ig, _     = integrated_gradients_single(
            model, inp_noisy, baseline, n_steps, target_class,
            half, spatial_half=spatial_half
        )
        ig_accum += ig.astype(np.float64)

    return (ig_accum / n_smooth).astype(np.float32), pred_prob


                                                                                 
                 
                                                                                 

def normalize_ig_global(all_ig, norm_percentile=NORM_PERCENTILE,
                        norm_scale=NORM_SCALE):
    concat = np.concatenate([np.abs(ig) for ig in all_ig])
    scale  = np.percentile(concat, norm_percentile)
    if scale == 0:
        scale = np.max(concat) if np.max(concat) > 0 else 1.0

    all_ig_norm = [
        np.clip(ig / scale, -1.0, 1.0) * norm_scale
        for ig in all_ig
    ]
    return all_ig_norm, float(scale)


                                                                                 
             
                                                                                 

def _academic_bar_chart(ax, x_pos, values, x_labels,
                        bar_color='#2878B5', bar_width=0.5,
                        ylim_min=0.5, label_fontsize=8):
    bars = ax.bar(x_pos, values,
                  color=bar_color, edgecolor='none',
                  width=bar_width, zorder=3)

          
    ax.axhline(0, color='black', linewidth=0.8, zorder=4)

                
    max_abs = max(abs(v) for v in values) if len(values) else 0.01
    offset  = max_abs * 0.04
    for bar, val in zip(bars, values):
        cx = bar.get_x() + bar.get_width() / 2
        if val >= 0:
            ax.text(cx, val + offset, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=label_fontsize,
                    color='black')
        else:
            ax.text(cx, val - offset, f'{val:.2f}',
                    ha='center', va='top', fontsize=label_fontsize,
                    color='black')

           
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right',
                       fontsize=10)

                                                    
    y_range = max(ylim_min, max_abs * 1.2)
    ax.set_ylim(-y_range, y_range)

                      
    if y_range <= 0.2:
        step = 0.05
    elif y_range <= 0.55:
        step = 0.1
    elif y_range <= 1.1:
        step = 0.2
    else:
        step = 0.5
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

                    
    ax.yaxis.grid(True, color='#CCCCCC', linewidth=0.6,
                  linestyle='-', zorder=0)
    ax.set_axisbelow(True)

                      
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
        spine.set_visible(True)

    ax.tick_params(axis='both', which='major', length=4, width=0.8,
                   direction='in', top=True, right=True)
    ax.tick_params(axis='both', which='minor', length=2, width=0.6,
                   direction='in', top=True, right=True)

    ax.set_facecolor('white')
    return y_range


def plot_deposit_histogram(ig_norm, pred_prob, deposit_id,
                           deposit_rc, selected_channels,
                           save_path, norm_percentile=NORM_PERCENTILE,
                           scale_factor=None):
    n        = len(selected_channels)
    x_labels = [DISPLAY_NAMES[k] for k in selected_channels]
    x_pos    = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.7), 5), facecolor='white')

    _academic_bar_chart(ax, x_pos, ig_norm, x_labels,
                        bar_color='#2878B5', bar_width=0.5,
                        ylim_min=0.5, label_fontsize=8)

    ax.set_ylabel('Attribution contribution', fontsize=11)

    r, c_col = deposit_rc
    scale_note = f'  (scale={scale_factor:.2e})' if scale_factor is not None else ''
    ax.set_title(
        f'Deposit #{deposit_id}  (row={r}, col={c_col})  '
        f'P(Li deposit) = {pred_prob:.3f}{scale_note}',
        fontsize=10, pad=8
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [OK]  #{deposit_id:02d} → {os.path.basename(save_path)}  '
          f'(P={pred_prob:.3f})')


                                                                                 
      
                                                                                 

def main():
    print('=' * 70)
    print('（IG）  –  Granite')
    print('=' * 70)

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice in Use: {device}')
    if device.type == 'cpu':
        print('  [] CPU  deposits，。')

                                                                             
    print('\n[1/5] Loading model...')
    from attribution_guided_model import create_attribution_guided_model

    model = create_attribution_guided_model(
        num_classes=2,
        feature_dim=FEATURE_DIM,
        transformer_depth=TRANSFORMER_DEPTH,
        transformer_heads=TRANSFORMER_HEADS
    ).to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  [OK] Model  (val_acc={ckpt.get("val_acc", 0):.2f}%，'
          f'val_f1={ckpt.get("val_f1", 0):.4f})')

                                                                           
    print('\n[2/5] Loading full-map data...')
    combined = np.load(COMBINED_NPY).astype(np.float32)
    H, W, C  = combined.shape
    print(f'  Combined_array : {combined.shape}  (H={H}, W={W}, C={C})')

                                                                      
    print('\n[3/5] ...')
    from sklearn.preprocessing import StandardScaler
    train_data = np.load(TRAIN_DATA)                                  
    scaler     = StandardScaler()
    scaler.fit(train_data.reshape(-1, C))
    combined_norm = scaler.transform(
        combined.reshape(-1, C)
    ).reshape(H, W, C)

                                                
                                                
                                    
    TRAIN_LABELS_PATH = os.path.join(NANLING_DIR, 'train_labels.npy')
    half = WINDOW_HALF

    if os.path.exists(TRAIN_LABELS_PATH):
        train_labels_arr = np.load(TRAIN_LABELS_PATH)                    
        no_mineral_idx   = np.where(train_labels_arr == 0)[0]
        if len(no_mineral_idx) > 0:
                                                                  
            no_mineral_win    = train_data[no_mineral_idx].mean(axis=0)           
            no_mineral_norm   = scaler.transform(
                no_mineral_win.reshape(-1, C)
            ).reshape(2*half+1, 2*half+1, C)                                
            baseline_win = (
                torch.FloatTensor(no_mineral_norm)
                    .permute(2, 0, 1)                                       
                    .unsqueeze(0)                                              
                    .clone()
                    .to(device)
            )
            print(f'  [OK]  = Non-mineralized（{len(no_mineral_idx)} Non-mineralized）')
            _bl = no_mineral_norm[half, half]
            print(f'       : '
                  f'Li={_bl[18]:.4f}  Be={_bl[6]:.4f}  W={_bl[35]:.4f}')
        else:
            print('  [Warning] Non-mineralized，Training Set')
            no_mineral_win = None
    else:
        print(f'  [Warning] Could not find {TRAIN_LABELS_PATH}，Training Set')
        no_mineral_win = None

    if no_mineral_win is None:
                               
        train_mean_original = scaler.mean_.astype(np.float32)
        baseline_norm_1d    = scaler.transform(
            train_mean_original.reshape(1, -1)
        ).reshape(C).astype(np.float32)
        baseline_win = (
            torch.FloatTensor(baseline_norm_1d)
                .view(1, C, 1, 1)
                .expand(1, C, 2*half+1, 2*half+1)
                .clone()
                .to(device)
        )
        print('  [OK]  = Training Set（）')

                                                                
    print('\n[4/5] ...')
    import cv2
    label_img  = cv2.imread(LABEL_TIF, cv2.IMREAD_UNCHANGED)
    if label_img is None:
        raise FileNotFoundError(f'label file: {LABEL_TIF}')

    raw_coords = np.array(list(zip(*np.where(label_img == 1))))
    if len(raw_coords) == 0:
        raise ValueError('label file（=1）。')

                                   
    MERGE_DIST = 5
    kept = []; used = [False] * len(raw_coords)
    for i in range(len(raw_coords)):
        if used[i]:
            continue
        kept.append(i)
        for j in range(i + 1, len(raw_coords)):
            if (not used[j] and
                    np.linalg.norm(raw_coords[i] - raw_coords[j]) < MERGE_DIST):
                used[j] = True
    deposit_rc_all = raw_coords[kept]

                                                      
    valid_mask = (
        (deposit_rc_all[:, 0] >= half) &
        (deposit_rc_all[:, 0] <  H - half) &
        (deposit_rc_all[:, 1] >= half) &
        (deposit_rc_all[:, 1] <  W - half)
    )
    deposit_rc = deposit_rc_all[valid_mask]
    skipped    = len(deposit_rc_all) - len(deposit_rc)
    print(f'  Number of deposits: {len(raw_coords)}  →  After merging: {len(deposit_rc_all)}  '
          f'→  （≥{half}px）: {len(deposit_rc)}')
    if skipped > 0:
        print(f'  [] {skipped}  deposits，。')

                                                                   
    print(f'\n[5/5]  IG （ {len(deposit_rc)}  deposits，'
          f'IG_STEPS={IG_STEPS}）...')

    all_ig_raw   = []                                 
    pred_probs   = []                  

    print(f'  : SmoothGrad-IG  '
          f'(steps={IG_STEPS}, smooth_n={SMOOTH_N}, '
          f'noise_std={SMOOTH_NOISE_STD}, spatial={2*SPATIAL_HALF+1}×{2*SPATIAL_HALF+1})')

    for i, (r, c_col) in enumerate(tqdm(deposit_rc, desc='SmoothGrad-IG', ncols=72)):
        win = combined_norm[r - half: r + half + 1,
                            c_col - half: c_col + half + 1, :]             
        inp = (torch.FloatTensor(win)
                   .permute(2, 0, 1)                         
                   .unsqueeze(0)                                
                   .to(device))

                                                  
        ig_full, prob = smoothgrad_ig(
            model, inp, baseline_win, IG_STEPS, TARGET_CLASS, half
        )
                 
        ig_sel = ig_full[SELECTED_CHANNELS]
        all_ig_raw.append(ig_sel)
        pred_probs.append(prob)

                                                              
    print(f'\n  （{NORM_PERCENTILE}th percentile clipping -> [-1, 1]）...')
    all_ig_norm, scale_factor = normalize_ig_global(all_ig_raw)
    print(f'  : {scale_factor:.4e}'
          f'  (SmoothGrad the smoothing stage should yield a larger scale factor than a single raw IG run)')

                                                                   
    print(f'\nPlotting deposit attribution contribution histograms...')
    for i, (r, c_col) in enumerate(deposit_rc):
        save_path = os.path.join(SAVE_DIR,
                                 f'deposit_{i+1:02d}_r{r}_c{c_col}.png')
        plot_deposit_histogram(
            ig_norm           = all_ig_norm[i],
            pred_prob         = pred_probs[i],
            deposit_id        = i + 1,
            deposit_rc        = (r, c_col),
            selected_channels = SELECTED_CHANNELS,
            save_path         = save_path,
            norm_percentile   = NORM_PERCENTILE,
            scale_factor      = scale_factor
        )

                                                                             
    import pandas as pd
    rows = []
    for i, (r, c_col) in enumerate(deposit_rc):
        row = {
            'deposit_id'  : i + 1,
            'row'         : int(r),
            'col'         : int(c_col),
            'pred_prob'   : round(float(pred_probs[i]), 6),
        }
        for j, k in enumerate(SELECTED_CHANNELS):
            row[f'ig_{CHANNEL_NAMES[k]}'] = round(float(all_ig_norm[i][j]), 6)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path   = os.path.join(SAVE_DIR, 'deposit_ig_summary.csv')
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n  [OK]  CSV saved: {os.path.basename(csv_path)}')

                                                     
    avg_ig   = np.stack(all_ig_norm).mean(axis=0)
    sort_idx = np.argsort(np.abs(avg_ig))[::-1]

    sorted_names   = [DISPLAY_NAMES[SELECTED_CHANNELS[i]] for i in sort_idx]
    sorted_mean_ig = [float(avg_ig[i]) for i in sort_idx]

    fig_avg, ax_avg = plt.subplots(
        figsize=(max(12, len(SELECTED_CHANNELS) * 0.7), 5), facecolor='white')
    xp = np.arange(len(SELECTED_CHANNELS))

    _academic_bar_chart(ax_avg, xp, sorted_mean_ig, sorted_names,
                        bar_color='#2878B5', bar_width=0.5,
                        ylim_min=0.5, label_fontsize=8)

    ax_avg.set_ylabel('Attribution contribution', fontsize=11)
    ax_avg.set_title(
        f'Average IG attribution  –  all {len(deposit_rc)} Li deposits  '
        f'(sorted by |mean|, {NORM_PERCENTILE}th-pct normalization)',
        fontsize=10, pad=8
    )
    fig_avg.tight_layout()
    avg_path = os.path.join(SAVE_DIR, 'all_deposits_avg_ig.png')
    fig_avg.savefig(avg_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig_avg)
    print(f'  [OK] Average contribution ranking plot -> {os.path.basename(avg_path)}')

                                                                            
    print('\n' + '=' * 70)
    print('All tasks completed!')
    print('=' * 70)
    print(f'\nResults saved to: {SAVE_DIR}/')
    print(f'  - deposit_XX_rRR_cCC.png  ：IG attribution contribution histograms for each deposit ({len(deposit_rc)} ）')
    print(f'  - all_deposits_avg_ig.png  ：Average IG contribution ranking across all deposits')
    print(f'  - deposit_ig_summary.csv   ：Summary table of normalized IG values for each deposit')
    print(f'\nNotes:')
    print(f'  X  ：Granite Li  {len(SELECTED_CHANNELS)-3}  + 3 Ore-controlling factors')
    print(f'  Y axis: normalized IG attribution contribution values (shared scaling across all deposits, range [-1, 1])')
    print(f'  Positive values: this feature is higher than the background value and promotes a mineral prediction')
    print(f'  Negative values: this feature is lower than the background value and suppresses a mineral prediction')
    print(f'   ：Ore-controlling factors')
    print(f'\nColor legend:')
    for grp, chs in GROUP_LABELS.items():
        names = [CHANNEL_NAMES[k] for k in chs if k in SELECTED_CHANNELS]
        print(f'  {grp}: {", ".join(names)}')


if __name__ == '__main__':
    main()
