"""Visualize Grad-CAM responses at known deposit locations."""


import argparse
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


MODEL_PATH = None
ORIGINAL_DEPOSIT_DIR = None
TRAIN_DATA_PATH = None
LABEL_PATH = None
SAVE_DIR = None
EXPECTED_DEPOSITS = 21
MERGE_DIST = 5


FIGURE_DPI = 300
INTERP_METHOD = "nearest"


FEATURE_DIM = 32
TRANSFORMER_DEPTH = 1
TRANSFORMER_HEADS = 2


def compute_gradcam_dcn_block2(model, inp_tensor, target_class=1):

    model.eval()

    acts_buf = {}
    grads_buf = {}

    def fwd_hook(module, inp_t, out_t):

        acts_buf["A"] = out_t
        out_t.register_hook(lambda g: grads_buf.update({"G": g.detach().clone()}))

    handle = model.dcn.block2.register_forward_hook(fwd_hook)

    output = model(inp_tensor)
    probs = F.softmax(output, dim=1)
    prob = probs[0, target_class].item()

    model.zero_grad()
    output[0, target_class].backward()

    handle.remove()

    if "A" not in acts_buf or "G" not in grads_buf:
        return np.zeros((9, 9)), prob

    A = acts_buf["A"].detach()  # (1, 128, 9, 9)
    G = grads_buf["G"]  # (1, 128, 9, 9)

    alpha = G.mean(dim=[2, 3], keepdim=True)  # (1, 128, 1, 1)

    cam = (alpha * A).sum(dim=1).squeeze(0)  # (9, 9)
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
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = gridspec.GridSpec(
        1, 2, figure=fig, left=0.06, right=0.96, top=0.84, bottom=0.22, wspace=0.10
    )

    ax_pat = fig.add_subplot(gs[0, 0])
    ax_cam = fig.add_subplot(gs[0, 1])

    ax_pat.set_title(
        "Deposits\nspatial pattern",
        fontsize=11,
        fontweight="bold",
        color="#1A237E",
        pad=5,
    )
    ax_cam.set_title(
        "Grad-CAM\n(DCN)", fontsize=11, fontweight="bold", color="#1A237E", pad=5
    )

    im_pat = ax_pat.imshow(
        rec["pattern"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation=INTERP_METHOD,
        aspect="equal",
    )
    ax_pat.plot(4, 4, "kx", markersize=11, markeredgewidth=2.5)
    ax_pat.axis("off")

    im_cam = ax_cam.imshow(
        rec["cam"],
        cmap="jet",
        vmin=0,
        vmax=1,
        interpolation=INTERP_METHOD,
        aspect="equal",
    )
    ax_cam.plot(4, 4, "kx", markersize=11, markeredgewidth=2.5)
    ax_cam.axis("off")

    ax_pat.text(
        -0.10,
        0.5,
        f"No.{rec['no']}",
        transform=ax_pat.transAxes,
        fontsize=13,
        fontweight="bold",
        va="center",
        ha="right",
        color="#1A237E",
    )

    cax_pat = fig.add_axes([0.06, 0.09, 0.40, 0.030])
    cb_pat = fig.colorbar(im_pat, cax=cax_pat, orientation="horizontal")
    cb_pat.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb_pat.ax.tick_params(labelsize=8)

    cax_cam = fig.add_axes([0.54, 0.09, 0.40, 0.030])
    cb_cam = fig.colorbar(im_cam, cax=cax_cam, orientation="horizontal")
    cb_cam.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb_cam.ax.tick_params(labelsize=8)

    fig.suptitle(
        f'Deposit No.{rec["no"]} / {n_total}  —  Deposits Spatial Pattern vs. DCN Grad-CAM',
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )

    fname = f'gradcam_dcn_deposit_{rec["no"]:02d}.png'
    save_path = os.path.join(save_dir, fname)
    plt.savefig(
        save_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.12,
    )
    plt.close(fig)
    return save_path


def main(argv=None):
    global MODEL_PATH, ORIGINAL_DEPOSIT_DIR, TRAIN_DATA_PATH, LABEL_PATH, SAVE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deposit-dir", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    MODEL_PATH = os.path.abspath(args.model)
    ORIGINAL_DEPOSIT_DIR = os.path.abspath(args.deposit_dir)
    TRAIN_DATA_PATH = os.path.abspath(args.train_data)
    LABEL_PATH = os.path.abspath(args.labels) if args.labels else None
    SAVE_DIR = os.path.abspath(args.output_dir)
    print("=" * 70)
    print("Grad-CAM 可视化：DCN 分支（block2）动态采样特征重要性热力图")
    print("=" * 70)

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    print("\n[1/5] 加载模型...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"找不到模型文件: {MODEL_PATH}\n" "请修改脚本顶部的 MODEL_PATH 变量指向正确路径。"
        )

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
    print(f"  [OK] 模型加载成功（最佳验证准确率: " f'{ckpt.get("val_acc", "未知"):.2f}%）')

    print("\n[2/5] 加载原始矿点样本...")
    if not os.path.isdir(ORIGINAL_DEPOSIT_DIR):
        raise FileNotFoundError(
            f"找不到原始矿点样本目录: {ORIGINAL_DEPOSIT_DIR}\n"
            "请确认卷积神经网络样本制作代码.py 已运行，且 sample/1/ 目录存在。"
        )

    npy_files = sorted(
        glob.glob(os.path.join(ORIGINAL_DEPOSIT_DIR, "*.npy")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    if len(npy_files) == 0:
        raise RuntimeError(f"{ORIGINAL_DEPOSIT_DIR} 目录下没有 .npy 文件。")

    all_raw = np.stack([np.load(f) for f in npy_files], axis=0)  # (N, 9, 9, 42)
    n_loaded = len(all_raw)
    print(f"  sample/1/ 共 {n_loaded} 个样本文件")

    import cv2

    label1 = cv2.imread(LABEL_PATH, 2) if LABEL_PATH else None
    if label1 is None:

        print(f"  [警告] 找不到标签文件 {LABEL_PATH}，" f"将直接使用前 {EXPECTED_DEPOSITS} 个样本。")
        ore_raw_data = all_raw[:EXPECTED_DEPOSITS]
    else:
        raw_coords = np.array(list(zip(*np.where(label1 == 1))))  # (N, 2) → (row, col)

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

        print(
            f"  去重前: {len(raw_coords)} 个像素  →  "
            f"合并后: {len(kept_indices)} 个独立矿点"
            f"（合并距离阈值: {MERGE_DIST} 像素）"
        )
        for skip in sorted(set(range(len(raw_coords))) - set(kept_indices)):
            r, c = raw_coords[skip]
            keeper = kept_indices[
                min(
                    range(len(kept_indices)),
                    key=lambda k: np.linalg.norm(
                        raw_coords[kept_indices[k]] - np.array([r, c])
                    ),
                )
            ]
            kr, kc = raw_coords[keeper]
            print(
                f"    合并: 像素{skip+1}(row={r},col={c}) → "
                f"保留为像素{keeper+1}(row={kr},col={kc})"
            )

        ore_raw_data = all_raw[kept_indices]

    n_ore = len(ore_raw_data)
    print(f"  [OK] 最终矿点样本数: {n_ore} 个")
    print(f"  数据形状: {ore_raw_data.shape}")

    print("\n[3/5] 数据标准化...")
    train_data = np.load(TRAIN_DATA_PATH)  # (N, 9, 9, 42)
    n_ch = ore_raw_data.shape[3]  # 42

    scaler = StandardScaler()
    scaler.fit(train_data.reshape(-1, n_ch))

    ore_norm_data = scaler.transform(ore_raw_data.reshape(-1, n_ch)).reshape(
        ore_raw_data.shape
    )  # (21, 9, 9, 42)
    print("  [OK] 标准化完成")

    print(f"\n[4/5] 准备生成 {n_ore} 个原始矿点的 Grad-CAM 热力图...")
    if n_ore == 0:
        print("  [错误] 未找到矿点样本。")
        return

    print(f"\n[5/5] 计算 Grad-CAM 并生成可视化（共 {n_ore} 个矿点）...\n")

    records = []
    for i in tqdm(range(n_ore), desc="Grad-CAM 计算进度", ncols=70):
        window_raw = ore_raw_data[i]
        window_norm = ore_norm_data[i]

        inp = torch.FloatTensor(window_norm).permute(2, 0, 1).unsqueeze(0).to(device)

        cam, prob = compute_gradcam_dcn_block2(model, inp, target_class=1)
        spatial = make_spatial_composite(window_raw)

        records.append({"no": i + 1, "pattern": spatial, "cam": cam, "prob": prob})

    print(f"\n[6/5] 逐矿点生成独立高清图像（共 {n_ore} 张）...\n")
    saved_files = []

    for rec in tqdm(records, desc="绘图进度", ncols=70):
        path = plot_single(rec, n_ore, SAVE_DIR)
        saved_files.append(path)

    print("\n" + "=" * 70)
    print("全部完成！")
    print("=" * 70)
    print(f"\n共生成 {len(saved_files)} 张独立高清图，保存在:")
    print(f"  {SAVE_DIR}/")
    for f in saved_files:
        print(f"    - {os.path.basename(f)}")
    print("\n说明:")
    print("  · 数据来源  ：sample/1/ 文件夹（扩增前的原始矿点，非增强样本）")
    print('  · 左图 "Deposits spatial pattern"：9×9 窗口所有通道均值合成图（原始值）')
    print('  · 右图 "Grad-CAM (DCN)"          ：DCNv2 block2 的 Grad-CAM 热力图')
    print(f"  · 插值方式                        ：{INTERP_METHOD}（平滑锐化）")
    print("  · p 值                            ：模型预测为矿点的概率")
    print("  · 热力图颜色：蓝色（低重要性）→ 红色（高重要性）")
    print("  · × 符号标记矿点中心位置（9×9 窗口正中心像素）")


if __name__ == "__main__":
    main()
