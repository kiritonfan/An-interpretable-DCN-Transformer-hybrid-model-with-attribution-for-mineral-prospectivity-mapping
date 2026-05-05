"""This script visualizes Grad-CAM results for DCN features across deposit samples."""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

                                                                  
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

plt.rcParams['font.family'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

                                                                             
             
                                                                             
                        
MODEL_PATH = os.path.join(current_dir, 'best_attribution_guided_model.pth')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(current_dir),
                              'best_attribution_guided_model.pth')

                                                    
ORIGINAL_DEPOSIT_DIR = r'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/New_data/sample/1'

                                                            
TRAIN_DATA_PATH   = r'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/train_data.npy'

SAVE_DIR          = os.path.join(current_dir, 'gradcam_dcn_all_deposits')
EXPECTED_DEPOSITS = 21                   
MERGE_DIST        = 5                                 

              
FIGURE_DPI        = 300
INTERP_METHOD     = 'nearest'                 

                
FEATURE_DIM        = 32
TRANSFORMER_DEPTH  = 1
TRANSFORMER_HEADS  = 2


                                                                             
                
                                                                             

def compute_gradcam_dcn_block2(model, inp_tensor, target_class=1):
    model.eval()

    acts_buf  = {}
    grads_buf = {}

    def fwd_hook(module, inp_t, out_t):
                                    
        acts_buf['A'] = out_t
        out_t.register_hook(lambda g: grads_buf.update({'G': g.detach().clone()}))

    handle = model.dcn.block2.register_forward_hook(fwd_hook)

                   
    output = model(inp_tensor)
    probs  = F.softmax(output, dim=1)
    prob   = probs[0, target_class].item()

                 
    model.zero_grad()
    output[0, target_class].backward()

    handle.remove()

                                                                         
    if 'A' not in acts_buf or 'G' not in grads_buf:
        return np.zeros((9, 9)), prob

    A = acts_buf['A'].detach()                   
    G = grads_buf['G']                           

                      
    alpha = G.mean(dim=[2, 3], keepdim=True)                          

                 
    cam = (alpha * A).sum(dim=1).squeeze(0)                   
    cam = F.relu(cam).cpu().numpy()

                 
    mn, mx = cam.min(), cam.max()
    if mx > mn:
        cam = (cam - mn) / (mx - mn)
    else:
        cam = np.zeros_like(cam)

    return cam, prob


                                                                             
             
                                                                             

def make_spatial_composite(window_nhwc):
    comp = window_nhwc.mean(axis=2).astype(np.float32)
    mn, mx = comp.min(), comp.max()
    if mx > mn:
        comp = (comp - mn) / (mx - mn)
    return comp


                                                                             
           
                                                                             

def plot_single(rec, n_total, save_dir):
                                
    fig_w, fig_h = 6.0, 4.5       
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

                                      
    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        left=0.06, right=0.96,
        top=0.84,  bottom=0.22,
        wspace=0.10
    )

    ax_pat = fig.add_subplot(gs[0, 0])
    ax_cam = fig.add_subplot(gs[0, 1])

                                                               
    ax_pat.set_title(
        'Deposits\nspatial pattern',
        fontsize=11, fontweight='bold', color='#1A237E', pad=5
    )
    ax_cam.set_title(
        'Grad-CAM\n(DCN)',
        fontsize=11, fontweight='bold', color='#1A237E', pad=5
    )

                                                                       
    im_pat = ax_pat.imshow(
        rec['pattern'], cmap='viridis',
        vmin=0, vmax=1,
        interpolation=INTERP_METHOD,
        aspect='equal'
    )
    ax_pat.plot(4, 4, 'kx', markersize=11, markeredgewidth=2.5)
    ax_pat.axis('off')

                                                                           
    im_cam = ax_cam.imshow(
        rec['cam'], cmap='jet',
        vmin=0, vmax=1,
        interpolation=INTERP_METHOD,
        aspect='equal'
    )
    ax_cam.plot(4, 4, 'kx', markersize=11, markeredgewidth=2.5)
    ax_cam.axis('off')

                                                      
    ax_pat.text(
        -0.10, 0.5, f"No.{rec['no']}",
        transform=ax_pat.transAxes,
        fontsize=13, fontweight='bold',
        va='center', ha='right', color='#1A237E'
    )

                                                            
                   
    cax_pat = fig.add_axes([0.06, 0.09, 0.40, 0.030])
    cb_pat = fig.colorbar(im_pat, cax=cax_pat, orientation='horizontal')
    cb_pat.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb_pat.ax.tick_params(labelsize=8)

                       
    cax_cam = fig.add_axes([0.54, 0.09, 0.40, 0.030])
    cb_cam = fig.colorbar(im_cam, cax=cax_cam, orientation='horizontal')
    cb_cam.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb_cam.ax.tick_params(labelsize=8)

                                                                         
    fig.suptitle(
        f'Deposit No.{rec["no"]} / {n_total}  —  Deposits Spatial Pattern vs. DCN Grad-CAM',
        fontsize=11, fontweight='bold', y=0.97
    )

    fname = f'gradcam_dcn_deposit_{rec["no"]:02d}.png'
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight',
                facecolor='white', pad_inches=0.12)
    plt.close(fig)
    return save_path


                                                                             
      
                                                                             

def main():
    print('=' * 70)
    print('Grad-CAM ：DCN （block2）sampled')
    print('=' * 70)

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice in Use: {device}')

                                                                         
    print('\n[1/5] Loading model...')
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f'Model: {MODEL_PATH}\n'
            'Please update the MODEL_PATH variable at the top of the script to the correct path.'
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
    print(f'  [OK] Model（best validation accuracy: '
          f'{ckpt.get("val_acc", "unknown"):.2f}%）')

                                                           
    print('\n[2/5] Loading original deposit samples...')
    if not os.path.isdir(ORIGINAL_DEPOSIT_DIR):
        raise FileNotFoundError(
            f'Original deposit sample directory not found: {ORIGINAL_DEPOSIT_DIR}\n'
            'Please make sure the CNN sample-generation script has been run and that the sample/1/ directory exists.'
        )

                       
    npy_files = sorted(
        glob.glob(os.path.join(ORIGINAL_DEPOSIT_DIR, '*.npy')),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
    if len(npy_files) == 0:
        raise RuntimeError(f'{ORIGINAL_DEPOSIT_DIR} contains no .npy files.')

    all_raw = np.stack([np.load(f) for f in npy_files], axis=0)                  
    n_loaded = len(all_raw)
    print(f'  sample/1/  {n_loaded} sample files')

                                                    
    label_path = os.path.join(
        os.path.dirname(ORIGINAL_DEPOSIT_DIR), '..', 'label', 'Li_deposits0.tif'
    )
    import cv2
    label1 = cv2.imread(label_path, 2)
    if label1 is None:
                                             
        print(f'  [Warning] label file {label_path}，'
              f'will use the first {EXPECTED_DEPOSITS} samples directly.')
        ore_raw_data = all_raw[:EXPECTED_DEPOSITS]
    else:
        raw_coords = np.array(list(zip(*np.where(label1 == 1))))                       

                                                
        kept_indices = []
        used = [False] * len(raw_coords)
        for i in range(len(raw_coords)):
            if used[i]:
                continue
            kept_indices.append(i)
            for j in range(i + 1, len(raw_coords)):
                if not used[j]:
                    d = np.linalg.norm(raw_coords[i] - raw_coords[j])
                    if d < MERGE_DIST:
                        used[j] = True               

        print(f'  Before deduplication: {len(raw_coords)}  pixels  →  '
              f'After merging: {len(kept_indices)}  independent deposits'
              f'（merge-distance threshold: {MERGE_DIST} pixels）')
        for skip in sorted(set(range(len(raw_coords))) - set(kept_indices)):
            r, c = raw_coords[skip]
            keeper = kept_indices[
                min(range(len(kept_indices)),
                    key=lambda k: np.linalg.norm(raw_coords[kept_indices[k]] - np.array([r, c])))]
            kr, kc = raw_coords[keeper]
            print(f'    : pixels{skip+1}(row={r},col={c}) → '
                  f'kept{keeper+1}(row={kr},col={kc})')

        ore_raw_data = all_raw[kept_indices]

    n_ore = len(ore_raw_data)
    print(f'  [OK] Final number of deposit samples: {n_ore} ')
    print(f'  Data shape: {ore_raw_data.shape}')

                                                       
    print('\n[3/5] ...')
    train_data = np.load(TRAIN_DATA_PATH)                  
    n_ch = ore_raw_data.shape[3]                

    scaler = StandardScaler()
    scaler.fit(train_data.reshape(-1, n_ch))            

    ore_norm_data = scaler.transform(
        ore_raw_data.reshape(-1, n_ch)
    ).reshape(ore_raw_data.shape)                              
    print('  [OK] Standardization complete')

                                                                       
    print(f'\n[4/5] Preparing to generate {n_ore} Grad-CAM heatmaps for original deposits...')
    if n_ore == 0:
        print('  [Error] No deposit samples were found.')
        return

                                                                      
    print(f'\n[5/5] Computing Grad-CAM and generating visualizations for {n_ore}  deposits）...\n')

    records = []
    for i in tqdm(range(n_ore), desc='Grad-CAM progress', ncols=70):
        window_raw  = ore_raw_data[i]                               
        window_norm = ore_norm_data[i]                              

                            
        inp = (torch.FloatTensor(window_norm)
               .permute(2, 0, 1)
               .unsqueeze(0)
               .to(device))

        cam, prob = compute_gradcam_dcn_block2(model, inp, target_class=1)
        spatial   = make_spatial_composite(window_raw)

        records.append({
            'no'     : i + 1,
            'pattern': spatial,
            'cam'    : cam,
            'prob'   : prob
        })

                                                                  
    print(f'\n[6/5] Generating standalone high-resolution images for {n_ore} images)...\n')
    saved_files = []

    for rec in tqdm(records, desc='Plotting progress', ncols=70):
        path = plot_single(rec, n_ore, SAVE_DIR)
        saved_files.append(path)

                                                                          
    print('\n' + '=' * 70)
    print('All tasks completed!')
    print('=' * 70)
    print(f'\nGenerated {len(saved_files)} standalone high-resolution images, saved in:')
    print(f'  {SAVE_DIR}/')
    for f in saved_files:
        print(f'    - {os.path.basename(f)}')
    print('\nNotes:')
    print('  · Data source  ：sample/1/ folder (original deposits before augmentation, not augmented samples)')
    print('  · Left panel "Deposits spatial pattern": mean composite of all channels in the 9×9 window (raw values)')
    print('  · Right panel "Grad-CAM (DCN)": Grad-CAM heatmap for DCNv2 block2')
    print(f'  · Interpolation method                        ：{INTERP_METHOD}(smoothed and sharpened)')
    print('  · p-value                            ：Model')
    print('  · Heatmap colors: blue (low importance) -> red (high importance)')
    print('  · The × marker indicates the deposit center (the exact center pixel of the 9×9 window)')


if __name__ == '__main__':
    main()
