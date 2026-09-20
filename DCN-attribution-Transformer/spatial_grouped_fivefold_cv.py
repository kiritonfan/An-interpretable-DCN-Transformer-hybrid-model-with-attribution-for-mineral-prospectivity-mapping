"""Run leakage-free repeated spatial-group cross-validation experiments."""


import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import sklearn
import torch
import torchvision
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OUT = None
INPUT_CUBE = None
POSITIVE_LABELS = None
NEGATIVE_LABELS = None

from Attribution_Guided_CNN_Transformer import create_attribution_guided_cnn_model
from Attribution_Guided_DCN_Transformer import create_attribution_guided_model
from CNN import CNN
from CNN_Transformer import CNNTransformerModel
from DCN import create_dcn_model
from DCN_Transformer import DCNTransformerModel
from Transformer import TransformerModel

ALL_MODELS = [
    "attribution_guided",
    "dcn_transformer",
    "dcn_only",
    "transformer_only",
    "cnn_only",
    "cnn_transformer",
    "attribution_guided_cnn",
]


def extract(cube, coords):
    return np.stack(
        [cube[row - 4 : row + 5, col - 4 : col + 5, :] for row, col in coords]
    ).astype(np.float32)


def augment_training(cube, coords, labels, groups):
    records = {}
    for (row, col), label, group in zip(coords, labels, groups):
        for row_shift in (-1, 0, 1):
            for col_shift in (-1, 0, 1):
                key = (int(row + row_shift), int(col + col_shift), int(label))
                records[key] = int(group)
    augmented_coords = np.asarray(
        [[row, col] for row, col, _ in records], dtype=np.int64
    )
    augmented_labels = np.asarray([label for _, _, label in records], dtype=np.int64)
    augmented_groups = np.asarray(list(records.values()), dtype=np.int64)
    return extract(cube, augmented_coords), augmented_labels, augmented_groups


def tensor(array):
    return torch.from_numpy(array).float().permute(0, 3, 1, 2)


def make_model(name):
    if name == "cnn_only":
        return CNN(num_classes=2, output_dim=32)
    if name == "cnn_transformer":
        return CNNTransformerModel(num_classes=2, feature_dim=32)
    if name == "attribution_guided_cnn":
        return create_attribution_guided_cnn_model(
            num_classes=2,
            feature_dim=32,
            transformer_depth=1,
            transformer_heads=2,
            factor_branch_type="linear",
        )
    if name == "dcn_only":
        return create_dcn_model(num_classes=2, feature_dim=32)
    if name == "dcn_transformer":
        return DCNTransformerModel(num_classes=2, feature_dim=32)
    if name == "transformer_only":
        return TransformerModel(num_classes=2, feature_dim=32)
    if name == "attribution_guided":
        return create_attribution_guided_model(
            num_classes=2,
            feature_dim=32,
            transformer_depth=1,
            transformer_heads=2,
            factor_branch_type="linear",
        )
    raise ValueError(f"Unknown model: {name}")


def training_objective(
    model, x, y, criterion, factor_loss_weight=0.35, transformer_loss_weight=0.50
):
    if hasattr(model, "attribution_network"):
        output, _, factor_info, branches = model(
            x,
            return_attribution=True,
            return_factor_attribution=True,
            return_branch_logits=True,
        )
        main_loss = criterion(output, y)
        factor_loss = nn.functional.binary_cross_entropy_with_logits(
            factor_info["branch_logit"].squeeze(-1), y.float()
        )
        transformer_loss = criterion(branches["transformer"], y)
        loss = (
            main_loss
            + factor_loss_weight * factor_loss
            + transformer_loss_weight * transformer_loss
        )
        return (
            loss,
            main_loss.detach(),
            {
                "factor": factor_loss.detach(),
                "transformer": transformer_loss.detach(),
            },
        )
    if hasattr(model, "logit_fusion"):
        output, branches = model(x, return_branch_logits=True)
        main_loss = criterion(output, y)
        transformer_loss = criterion(branches["transformer"], y)
        return (
            main_loss + transformer_loss_weight * transformer_loss,
            main_loss.detach(),
            {"transformer": transformer_loss.detach()},
        )
    main_loss = criterion(model(x), y)
    return main_loss, main_loss.detach(), None


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_sources():
    positive = np.argwhere(cv2.imread(POSITIVE_LABELS, 0) == 1)
    negative = np.argwhere(cv2.imread(NEGATIVE_LABELS, 0) == 0)
    coords = np.concatenate([negative, positive]).astype(np.int64)
    labels = np.concatenate(
        [
            np.zeros(len(negative), dtype=np.int64),
            np.ones(len(positive), dtype=np.int64),
        ]
    )
    groups, offset = [], 0
    for points in (negative, positive):
        distance = np.max(np.abs(points[:, None, :] - points[None, :, :]), axis=2)
        n_groups, local = connected_components(
            (distance <= 12).astype(np.uint8), directed=False
        )
        groups.append(local + offset)
        offset += n_groups
    return coords, labels, np.concatenate(groups).astype(np.int64)


def tta_windows(cube, coords):
    windows, owners = [], []
    for owner, (row, col) in enumerate(coords):
        shifted = np.asarray(
            [(row + dr, col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)],
            dtype=np.int64,
        )
        windows.append(extract(cube, shifted))
        owners.extend([owner] * 9)
    return np.concatenate(windows), np.asarray(owners, dtype=np.int64)


def scale_fit(train, *others):
    scaler = StandardScaler().fit(train.reshape(-1, train.shape[-1]))
    arrays = [train, *others]
    scaled = [
        scaler.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape) for a in arrays
    ]
    return scaler, scaled


def class_weights(labels, device):
    counts = np.bincount(labels, minlength=2)
    return torch.tensor(
        len(labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device
    )


def make_loader(data, labels, batch_size, shuffle, generator_seed=None):
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(int(generator_seed))
    return DataLoader(
        TensorDataset(tensor(data), torch.from_numpy(labels).long()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def initialize_attribution_reference(model, loader, device):
    if hasattr(model, "attribution_network"):
        model.attribution_network.update_reference_features(loader, device)


def train_with_inner_stopping(
    model,
    train_loader,
    stop_loader,
    device,
    weights,
    epochs,
    patience,
    factor_loss_weight,
    transformer_loss_weight,
):
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    best_loss, best_epoch, best_state, stale = float("inf"), 1, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            loss, _, _ = training_objective(
                model,
                xb,
                yb,
                criterion,
                factor_loss_weight=factor_loss_weight,
                transformer_loss_weight=transformer_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            batches += 1
        model.eval()
        stop_total = 0.0
        stop_batches = 0
        with torch.no_grad():
            for xb, yb in stop_loader:
                stop_total += criterion(model(xb.to(device)), yb.to(device)).item()
                stop_batches += 1
        stop_loss = stop_total / max(stop_batches, 1)
        history.append((epoch, total / max(batches, 1), stop_loss))
        if stop_loss < best_loss:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_loss, history


def train_fixed_epochs(
    model,
    train_loader,
    device,
    weights,
    epochs,
    factor_loss_weight,
    transformer_loss_weight,
):
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            loss, _, _ = training_objective(
                model,
                xb,
                yb,
                criterion,
                factor_loss_weight=factor_loss_weight,
                transformer_loss_weight=transformer_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            batches += 1
        history.append((epoch, total / max(batches, 1)))
    return model, history


def evaluate_tta(model, scaled_windows, owners, labels, device):
    loader = make_loader(
        scaled_windows, np.zeros(len(scaled_windows), dtype=np.int64), 96, False
    )
    batches = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            batches.append(torch.softmax(model(xb.to(device)), 1)[:, 1].cpu().numpy())
    window_probs = np.concatenate(batches)
    point_probs = np.asarray(
        [window_probs[owners == index].mean() for index in range(len(labels))]
    )
    return metric_values(labels, point_probs), point_probs, window_probs


def evaluate_tta_branches(model, scaled_windows, owners, labels, device):
    """Evaluate each fused classifier branch independently on outer-fold TTA."""
    if not hasattr(model, "logit_fusion"):
        return {}
    loader = make_loader(
        scaled_windows, np.zeros(len(scaled_windows), dtype=np.int64), 96, False
    )
    collected = {}
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            _, branch_info = model(xb.to(device), return_branch_logits=True)
            for name in ("base", "transformer", "factor"):
                if name in branch_info:
                    collected.setdefault(name, []).append(
                        torch.softmax(branch_info[name], 1)[:, 1].cpu().numpy()
                    )
    results = {}
    for name, batches in collected.items():
        window_probs = np.concatenate(batches)
        point_probs = np.asarray(
            [window_probs[owners == index].mean() for index in range(len(labels))]
        )
        results[name] = metric_values(labels, point_probs)
    return results


def evaluate_tta_module_effects(model, scaled_windows, owners, labels, device):
    """Measure counterfactual prediction changes when each module is bypassed."""
    if not hasattr(model, "logit_fusion"):
        return []
    loader = make_loader(
        scaled_windows, np.zeros(len(scaled_windows), dtype=np.int64), 96, False
    )
    collected = {"full": [], "no_transformer": []}
    is_attribution_model = hasattr(model, "attribution_network")
    if is_attribution_model:
        collected.update({"no_factor": [], "no_attribution_guidance": []})
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            full_logits, branches = model(xb, return_branch_logits=True)
            weights = branches["fusion_weights"].to(full_logits)
            collected["full"].append(torch.softmax(full_logits, 1)[:, 1].cpu().numpy())
            if "factor" in branches:
                without_transformer = (
                    weights[0] * branches["base"] + weights[2] * branches["factor"]
                ) / (weights[0] + weights[2])
                without_factor = (
                    weights[0] * branches["base"] + weights[1] * branches["transformer"]
                ) / (weights[0] + weights[1])
                no_guidance_logits = model(xb, disable_attribution_guidance=True)
                collected["no_factor"].append(
                    torch.softmax(without_factor, 1)[:, 1].cpu().numpy()
                )
                collected["no_attribution_guidance"].append(
                    torch.softmax(no_guidance_logits, 1)[:, 1].cpu().numpy()
                )
            else:
                without_transformer = branches["base"]
            collected["no_transformer"].append(
                torch.softmax(without_transformer, 1)[:, 1].cpu().numpy()
            )

    point_probabilities = {}
    for variant, batches in collected.items():
        window_probs = np.concatenate(batches)
        point_probabilities[variant] = np.asarray(
            [window_probs[owners == index].mean() for index in range(len(labels))]
        )
    full_probs = point_probabilities.pop("full")
    full_metrics = metric_values(labels, full_probs)
    rows = []
    for variant, probabilities in point_probabilities.items():
        variant_metrics = metric_values(labels, probabilities)
        rows.append(
            {
                "comparison": f"full_vs_{variant}",
                "probability_mae": float(np.mean(np.abs(full_probs - probabilities))),
                "probability_max_abs": float(
                    np.max(np.abs(full_probs - probabilities))
                ),
                "classification_flip_rate": float(
                    np.mean((full_probs >= 0.5) != (probabilities >= 0.5))
                ),
                **{
                    f"delta_{name}": full_metrics[name] - variant_metrics[name]
                    for name in full_metrics
                },
            }
        )
    return rows


def metric_values(labels, probabilities):
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "auc": roc_auc_score(labels, probabilities),
        "kappa": cohen_kappa_score(labels, predictions),
        "mcc": matthews_corrcoef(labels, predictions),
    }


def write_csv(name, rows):
    if not rows:
        return
    path = os.path.join(OUT, name)
    temporary = path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_and_protocol(args, device, coords, labels, groups):
    snapshot_dir = os.path.join(OUT, "code_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    hashes = {}
    for source in sorted(Path(args.source_dir).resolve().glob("*.py")):
        target = os.path.join(snapshot_dir, source.name)
        shutil.copy2(source, target)
        hashes[source.name] = sha256(target)
    cube_path = INPUT_CUBE
    label_paths = [POSITIVE_LABELS, NEGATIVE_LABELS]
    protocol = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_points": len(labels),
        "positive_points": int(labels.sum()),
        "negative_points": int((labels == 0).sum()),
        "spatial_groups": int(len(np.unique(groups))),
        "models": args.models,
        "outer_cv": "StratifiedGroupKFold",
        "outer_folds": 5,
        "seeds": args.seeds,
        "outer_validation_used_for_model_selection": False,
        "epoch_selection": "one 4-fold StratifiedGroupKFold holdout inside outer training fold",
        "final_fit": "fresh initialization on full outer-training fold for selected epoch count",
        "training_augmentation": "3x3 center shifts on fitting data only",
        "outer_evaluation": "3x3 TTA averaged to one probability per source point",
        "factor_branch_supervision": "BCEWithLogits auxiliary loss plus direct logit fusion",
        "transformer_design": "mean pooling, zero position init, constrained convex logit fusion",
        "attribution_baseline": "fixed mean of negative windows in the fitting fold only",
        "attribution_spatial_map": "DeepLIFT multipliers mapped to local input deviations",
        "attribution_fusion": "positive bounded attention strength and constrained feature mixing",
        "shared_dropout": 0.15,
        "factor_loss_weight": args.factor_loss_weight,
        "transformer_loss_weight": args.transformer_loss_weight,
        "attribution_variance_weight": 0.0,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "threshold": 0.5,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "sklearn": sklearn.__version__,
        "opencv": cv2.__version__,
        "deterministic_algorithms": "warn_only",
        "code_snapshot_sha256": hashes,
        "data_sha256": {
            os.path.basename(cube_path): sha256(cube_path),
            **{os.path.basename(path): sha256(path) for path in label_paths},
        },
    }
    with open(os.path.join(OUT, "protocol.json"), "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)


def verify_module_gradients(device, factor_loss_weight, transformer_loss_weight):
    checks = {}
    for model_name in (
        "attribution_guided",
        "attribution_guided_cnn",
        "dcn_transformer",
        "cnn_transformer",
    ):
        seed_all(99991)
        model = make_model(model_name).to(device)
        xb = torch.randn(2, 42, 9, 9, device=device)
        yb = torch.tensor([0, 1], device=device)
        reference_loader = DataLoader(
            TensorDataset(xb.detach().cpu(), yb.detach().cpu()),
            batch_size=2,
            shuffle=False,
        )
        initialize_attribution_reference(model, reference_loader, device)
        loss, _, _ = training_objective(
            model,
            xb,
            yb,
            nn.CrossEntropyLoss(),
            factor_loss_weight=factor_loss_weight,
            transformer_loss_weight=transformer_loss_weight,
        )
        loss.backward()
        required_groups = {
            "transformer": [
                p.grad for name, p in model.named_parameters() if "transformer" in name
            ],
            "logit_fusion": [
                p.grad for name, p in model.named_parameters() if "logit_fusion" in name
            ],
        }
        if hasattr(model, "attribution_network"):
            required_groups.update(
                {
                    "factor_branch": [
                        p.grad
                        for name, p in model.named_parameters()
                        if "factor_branch" in name
                    ],
                    "attribution_attention": [
                        p.grad
                        for name, p in model.named_parameters()
                        if "attribution_attention" in name
                    ],
                }
            )
        for group_name, gradients in required_groups.items():
            passed = bool(gradients) and all(
                grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
                for grad in gradients
            )
            if not passed:
                raise RuntimeError(f"{group_name} gradient check failed: {model_name}")
        checks[model_name] = {name: "PASS" for name in required_groups}
        del model
    return checks


def verify_module_effects(device):
    """Fail fast if a supposedly active module has no numerical forward effect."""
    checks = {}
    xb = torch.randn(3, 42, 9, 9, device=device)
    for model_name in (
        "attribution_guided",
        "attribution_guided_cnn",
        "dcn_transformer",
        "cnn_transformer",
    ):
        seed_all(99992)
        model = make_model(model_name).to(device).eval()
        reference_loader = DataLoader(
            TensorDataset(xb.detach().cpu(), torch.tensor([0, 1, 0])),
            batch_size=3,
            shuffle=False,
        )
        initialize_attribution_reference(model, reference_loader, device)
        with torch.no_grad():
            full, branches = model(xb, return_branch_logits=True)
            transformer_delta = float(
                (
                    torch.softmax(full, 1)[:, 1]
                    - torch.softmax(branches["base"], 1)[:, 1]
                )
                .abs()
                .mean()
            )
            if not np.isfinite(transformer_delta) or transformer_delta <= 1e-7:
                raise RuntimeError(
                    f"Transformer forward-effect check failed: {model_name}"
                )
            result = {"transformer_probability_mae": transformer_delta}
            if hasattr(model, "attribution_network"):
                no_guidance = model(xb, disable_attribution_guidance=True)
                attribution_delta = float(
                    (torch.softmax(full, 1)[:, 1] - torch.softmax(no_guidance, 1)[:, 1])
                    .abs()
                    .mean()
                )
                if not np.isfinite(attribution_delta) or attribution_delta <= 1e-7:
                    raise RuntimeError(
                        f"Attribution forward-effect check failed: {model_name}"
                    )
                result["attribution_probability_mae"] = attribution_delta
        checks[model_name] = result
        del model
    return checks


def assert_group_disjoint(groups, left, right, name):
    overlap = set(groups[left].tolist()) & set(groups[right].tolist())
    if overlap:
        raise RuntimeError(f"{name} group leakage: {sorted(overlap)}")


def main():
    global OUT, INPUT_CUBE, POSITIVE_LABELS, NEGATIVE_LABELS
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--factor-loss-weight", type=float, default=0.35)
    parser.add_argument("--transformer-loss-weight", type=float, default=0.50)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--cube", required=True)
    parser.add_argument("--positive-labels", required=True)
    parser.add_argument("--negative-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dir", default=".")
    args = parser.parse_args()
    OUT = os.path.abspath(args.output_dir)
    INPUT_CUBE = os.path.abspath(args.cube)
    POSITIVE_LABELS = os.path.abspath(args.positive_labels)
    NEGATIVE_LABELS = os.path.abspath(args.negative_labels)
    if len(args.models) != len(set(args.models)):
        raise ValueError("Duplicate model names are not allowed")
    if args.factor_loss_weight < 0 or args.transformer_loss_weight < 0:
        raise ValueError("Auxiliary loss weights must be non-negative")
    os.makedirs(OUT, exist_ok=True)
    configure_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This unified experiment requires the DL1 CUDA environment")

    cube = np.load(INPUT_CUBE, mmap_mode="r")
    coords, labels, groups = build_sources()
    snapshot_and_protocol(args, device, coords, labels, groups)
    gradient_checks = verify_module_gradients(
        device, args.factor_loss_weight, args.transformer_loss_weight
    )
    effect_checks = verify_module_effects(device)
    with open(os.path.join(OUT, "preflight.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cuda": True,
                "module_gradient_checks": gradient_checks,
                "module_forward_effect_checks": effect_checks,
            },
            handle,
            indent=2,
        )

    prediction_rows, window_rows, fold_rows, branch_metric_rows = [], [], [], []
    epoch_rows, history_rows, parameter_rows, fusion_rows = [], [], [], []
    module_effect_rows = []
    for model_name in args.models:
        seed_all(12345)
        parameter_rows.append(
            {
                "model": model_name,
                "trainable_parameters": sum(
                    p.numel()
                    for p in make_model(model_name).parameters()
                    if p.requires_grad
                ),
            }
        )
    write_csv("parameter_counts.csv", parameter_rows)

    for seed in range(args.seeds):
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (outer_train, outer_test) in enumerate(
            outer.split(coords, labels, groups), 1
        ):
            assert_group_disjoint(groups, outer_train, outer_test, "outer")
            inner_splitter = StratifiedGroupKFold(
                4, shuffle=True, random_state=10000 + seed * 100 + fold
            )
            local_fit, local_stop = next(
                inner_splitter.split(
                    coords[outer_train], labels[outer_train], groups[outer_train]
                )
            )
            inner_fit, inner_stop = outer_train[local_fit], outer_train[local_stop]
            assert_group_disjoint(groups, inner_fit, inner_stop, "inner")

            inner_train_x, inner_train_y, _ = augment_training(
                cube, coords[inner_fit], labels[inner_fit], groups[inner_fit]
            )
            inner_stop_x = extract(cube, coords[inner_stop])
            _, (inner_train_x, inner_stop_x) = scale_fit(inner_train_x, inner_stop_x)
            final_train_x, final_train_y, _ = augment_training(
                cube, coords[outer_train], labels[outer_train], groups[outer_train]
            )
            outer_tta, owners = tta_windows(cube, coords[outer_test])
            final_scaler, (final_train_x, outer_tta) = scale_fit(
                final_train_x, outer_tta
            )

            for model_name in args.models:
                base_seed = seed * 1000 + fold * 10
                seed_all(base_seed)
                inner_train_loader = make_loader(
                    inner_train_x, inner_train_y, args.batch_size, True, base_seed + 1
                )
                inner_stop_loader = make_loader(
                    inner_stop_x, labels[inner_stop], args.batch_size, False
                )
                model = make_model(model_name).to(device)
                initialize_attribution_reference(
                    model,
                    make_loader(inner_train_x, inner_train_y, args.batch_size, False),
                    device,
                )
                model, best_epoch, best_loss, inner_history = train_with_inner_stopping(
                    model,
                    inner_train_loader,
                    inner_stop_loader,
                    device,
                    class_weights(inner_train_y, device),
                    args.epochs,
                    args.patience,
                    args.factor_loss_weight,
                    args.transformer_loss_weight,
                )
                for epoch, train_loss, stop_loss in inner_history:
                    history_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "fold": fold,
                            "phase": "inner_epoch_selection",
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "validation_loss": stop_loss,
                        }
                    )
                del model
                torch.cuda.empty_cache()

                final_seed = base_seed + 500000
                seed_all(final_seed)
                final_loader = make_loader(
                    final_train_x, final_train_y, args.batch_size, True, final_seed + 1
                )
                model = make_model(model_name).to(device)
                initialize_attribution_reference(
                    model,
                    make_loader(final_train_x, final_train_y, args.batch_size, False),
                    device,
                )
                model, final_history = train_fixed_epochs(
                    model,
                    final_loader,
                    device,
                    class_weights(final_train_y, device),
                    best_epoch,
                    args.factor_loss_weight,
                    args.transformer_loss_weight,
                )
                for epoch, train_loss in final_history:
                    history_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "fold": fold,
                            "phase": "full_outer_training",
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "validation_loss": "",
                        }
                    )

                fold_metric, point_probs, window_probs = evaluate_tta(
                    model, outer_tta, owners, labels[outer_test], device
                )
                branch_metrics = evaluate_tta_branches(
                    model, outer_tta, owners, labels[outer_test], device
                )
                module_effects = evaluate_tta_module_effects(
                    model, outer_tta, owners, labels[outer_test], device
                )
                for effect in module_effects:
                    module_effect_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "fold": fold,
                            "n": len(outer_test),
                            **effect,
                        }
                    )
                for branch_name, branch_metric in branch_metrics.items():
                    branch_metric_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "fold": fold,
                            "branch": branch_name,
                            "n": len(outer_test),
                            **branch_metric,
                        }
                    )
                if hasattr(model, "fusion_diagnostics"):
                    fusion_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "fold": fold,
                            "base_weight": "",
                            "transformer_weight": "",
                            "factor_attribution_weight": "",
                            "attribution_strength": "",
                            "guided_feature_weight": "",
                            **model.fusion_diagnostics(),
                        }
                    )
                fold_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "n": len(outer_test),
                        **fold_metric,
                    }
                )
                epoch_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "inner_fit_points": len(inner_fit),
                        "inner_stop_points": len(inner_stop),
                        "inner_fit_groups": len(np.unique(groups[inner_fit])),
                        "inner_stop_groups": len(np.unique(groups[inner_stop])),
                        "selected_epoch": best_epoch,
                        "best_inner_validation_loss": best_loss,
                    }
                )
                for local, index in enumerate(outer_test):
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "fold": fold,
                            "sample": int(index),
                            "label": int(labels[index]),
                            "group_id": int(groups[index]),
                            "row": int(coords[index, 0]),
                            "col": int(coords[index, 1]),
                            "probability": float(point_probs[local]),
                        }
                    )
                    for aug, probability in enumerate(window_probs[owners == local]):
                        window_rows.append(
                            {
                                "model": model_name,
                                "seed": seed,
                                "fold": fold,
                                "sample": int(index),
                                "tta_window": aug,
                                "probability": float(probability),
                            }
                        )

                checkpoint_dir = os.path.join(OUT, "checkpoints", model_name)
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "selected_epoch": best_epoch,
                        "state_dict": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        "scaler_mean": final_scaler.mean_,
                        "scaler_scale": final_scaler.scale_,
                    },
                    os.path.join(checkpoint_dir, f"seed_{seed}_fold_{fold}.pth"),
                )
                del model
                torch.cuda.empty_cache()

                write_csv("oof_point_predictions.csv", prediction_rows)
                write_csv("tta_window_predictions.csv", window_rows)
                write_csv("seven_metrics_by_fold.csv", fold_rows)
                write_csv("branch_metrics_by_fold.csv", branch_metric_rows)
                write_csv("selected_epochs.csv", epoch_rows)
                write_csv("training_history.csv", history_rows)
                write_csv("fusion_diagnostics.csv", fusion_rows)
                write_csv("module_effects_by_fold.csv", module_effect_rows)
                print(
                    json.dumps(
                        {
                            "status": "fold_complete",
                            "model": model_name,
                            "seed": seed,
                            "fold": fold,
                            "selected_epoch": best_epoch,
                            **fold_metric,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    seed_rows = []
    for model_name in args.models:
        for seed in range(args.seeds):
            subset = [
                row
                for row in prediction_rows
                if row["model"] == model_name and row["seed"] == seed
            ]
            subset.sort(key=lambda row: row["sample"])
            if len(subset) != len(labels):
                raise RuntimeError(f"Incomplete OOF set: {model_name}, seed={seed}")
            probabilities = np.asarray([row["probability"] for row in subset])
            y_true = np.asarray([row["label"] for row in subset])
            seed_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "n_points": len(y_true),
                    "n_groups": len(np.unique(groups)),
                    **metric_values(y_true, probabilities),
                }
            )
    write_csv("seven_metrics_by_seed.csv", seed_rows)
    np.save(os.path.join(OUT, "source_coordinates.npy"), coords)
    np.save(os.path.join(OUT, "source_labels.npy"), labels)
    np.save(os.path.join(OUT, "source_groups.npy"), groups)
    print(
        json.dumps({"status": "complete", "output": OUT}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
