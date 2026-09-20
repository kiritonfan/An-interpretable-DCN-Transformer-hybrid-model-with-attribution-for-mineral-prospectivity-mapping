"""Implement Transformer components and the shared training pipeline."""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedBranchFusion(nn.Module):
    """Learnable convex fusion that cannot silently switch a branch off.

    ``minimum_weight`` reserves a fixed share for every branch.  The remaining
    probability mass is learned with a softmax, so weights stay positive and
    sum to one while every branch continues to receive task gradients.
    """

    def __init__(self, initial_weights, minimum_weight=0.15):
        super().__init__()
        initial = torch.as_tensor(initial_weights, dtype=torch.float32)
        if initial.ndim != 1 or initial.numel() < 2:
            raise ValueError("initial_weights must contain at least two branches")
        if not torch.isclose(initial.sum(), torch.tensor(1.0), atol=1e-6):
            raise ValueError("initial_weights must sum to one")
        if minimum_weight < 0 or minimum_weight * initial.numel() >= 1:
            raise ValueError("minimum_weight leaves no learnable probability mass")
        if torch.any(initial <= minimum_weight):
            raise ValueError("every initial weight must exceed minimum_weight")
        self.minimum_weight = float(minimum_weight)
        self.remaining_weight = 1.0 - self.minimum_weight * initial.numel()
        free = (initial - self.minimum_weight) / self.remaining_weight
        self.fusion_logits = nn.Parameter(free.log())

    def weights(self):
        learned = torch.softmax(self.fusion_logits, dim=0)
        return self.minimum_weight + self.remaining_weight * learned

    def forward(self, *branches):
        if len(branches) != self.fusion_logits.numel():
            raise ValueError(
                f"expected {self.fusion_logits.numel()} branches, got {len(branches)}"
            )
        weights = self.weights().to(device=branches[0].device, dtype=branches[0].dtype)
        return sum(weight * branch for weight, branch in zip(weights, branches))


def exists(val):
    return val is not None


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=None, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim_head if dim_head is not None else dim // num_heads
        if self.head_dim <= 0:
            raise ValueError("dim_head must be positive")
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, self.inner_dim * 3)
        self.proj = nn.Linear(self.inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.attention_weights = None

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        self.attention_weights = attn.detach()

        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.inner_dim)
        x = self.proj(x)
        x = self.dropout(x)

        if return_attention:
            return x, self.attention_weights
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=None, mlp_dim=None, dropout=0.1):
        super().__init__()
        mlp_dim = mlp_dim if mlp_dim is not None else dim * 2
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(
            dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head=64, mlp_dim=None, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim if mlp_dim is not None else dim * 2
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    num_heads=heads,
                    dim_head=self.dim_head,
                    mlp_dim=self.mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, return_attention=False):
        attention_maps = []

        for block in self.layers:
            if return_attention:
                x, attn_weights = block(x, return_attention=True)
                attention_maps.append(attn_weights)
            else:
                x = block(x)

        x = self.norm(x)

        if return_attention:
            return x, attention_maps
        return x


class DTransformer(nn.Module):
    def __init__(
        self, *, image_size, patch_size, attn_layers, num_classes, dropout=0.0
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 81

        self.dim = attn_layers.dim if hasattr(attn_layers, "dim") else 512

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.dim))
        self.dropout = nn.Dropout(dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(self.dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, num_classes)
        )

    def forward(self, img, return_attention=False):
        b, c, h, w = img.shape
        num_patches = h * w

        patches = img.permute(0, 2, 3, 1).reshape(b, num_patches, -1)

        if num_patches != self.num_patches:
            pos_embedding = F.interpolate(
                self.pos_embedding.permute(0, 2, 1), size=num_patches, mode="linear"
            ).permute(0, 2, 1)
        else:
            pos_embedding = self.pos_embedding

        patches = patches + pos_embedding

        patches = self.dropout(patches)

        if return_attention:
            patches, attention_maps = self.attn_layers(patches, return_attention=True)
        else:
            patches = self.attn_layers(patches)

        patches = self.norm(patches)

        pooled_output = patches.mean(dim=1)
        logits = self.mlp_head(pooled_output)

        if return_attention:
            return logits, attention_maps
        return logits


class LegacyCLSDTransformer(nn.Module):
    """CLS-token classifier retained for the historical probability models.

    This class reproduces the Transformer head used to generate the original
    DCN-Transformer and Transformer-only study-area CSV files.  The current
    attribution-guided model continues to use :class:`DTransformer`, which
    performs mean pooling and has no CLS token.
    """

    def __init__(
        self, *, image_size, patch_size, attn_layers, num_classes, dropout=0.0
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = attn_layers.dim if hasattr(attn_layers, "dim") else 512
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(dropout)
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(self.dim)
        hidden_dim = max(128, self.dim // 2)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, img, return_attention=False):
        batch, _, height, width = img.shape
        num_patches = height * width
        patches = img.permute(0, 2, 3, 1).reshape(batch, num_patches, -1)
        if num_patches != self.num_patches:
            position = F.interpolate(
                self.pos_embedding.permute(0, 2, 1),
                size=num_patches,
                mode="linear",
            ).permute(0, 2, 1)
        else:
            position = self.pos_embedding
        patches = patches + position
        cls_token = self.cls_token.expand(batch, -1, -1)
        sequence = self.dropout(torch.cat((cls_token, patches), dim=1))
        if return_attention:
            sequence, attention_maps = self.attn_layers(sequence, return_attention=True)
        else:
            sequence = self.attn_layers(sequence)
            attention_maps = None
        logits = self.mlp_head(self.norm(sequence)[:, 0])
        if return_attention:
            return logits, attention_maps
        return logits


def run_model_pipeline(model_factory, model_name, argv=None, pipeline_defaults=None):
    """Train one model, restore its best checkpoint, and map probabilities.

    The seven public model scripts call this shared runner so that scaling,
    optimization, validation, and study-area inference are identical across
    architectures.  Input samples are expected in NHWC format (N, 9, 9, 42).
    """
    import argparse
    import csv
    import json
    import math
    import random
    from pathlib import Path

    import cv2
    import joblib
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    defaults = {
        "epochs": 100,
        "patience": 15,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_epochs": 5,
        "factor_loss_weight": 0.35,
        "transformer_loss_weight": 0.50,
        "label_smoothing": 0.0,
        "factor_l1_weight": 0.0,
        "factor_l2_weight": 0.0,
    }
    if pipeline_defaults:
        defaults.update(pipeline_defaults)
    parser = argparse.ArgumentParser(
        description=f"Train and predict with {model_name}."
    )
    parser.add_argument("--mode", choices=("all", "train", "predict"), default="all")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-data")
    parser.add_argument("--train-labels")
    parser.add_argument("--validation-data")
    parser.add_argument("--validation-labels")
    parser.add_argument("--study-area-data")
    parser.add_argument("--x-coordinates")
    parser.add_argument("--y-coordinates")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--scaler",
        default=None,
        help="Scaler file used with an external historical checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--patience", type=int, default=defaults["patience"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument(
        "--max-prediction-windows",
        type=int,
        default=None,
        help="Optional inference limit for smoke testing; omit for the full area.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=defaults["learning_rate"]
    )
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--warmup-epochs", type=int, default=defaults["warmup_epochs"])
    parser.add_argument(
        "--factor-loss-weight", type=float, default=defaults["factor_loss_weight"]
    )
    parser.add_argument(
        "--transformer-loss-weight",
        type=float,
        default=defaults["transformer_loss_weight"],
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=defaults["label_smoothing"]
    )
    parser.add_argument(
        "--factor-l1-weight", type=float, default=defaults["factor_l1_weight"]
    )
    parser.add_argument(
        "--factor-l2-weight", type=float, default=defaults["factor_l2_weight"]
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    training_inputs = (
        args.train_data,
        args.train_labels,
        args.validation_data,
        args.validation_labels,
    )
    if args.mode in ("all", "train") and any(
        value is None for value in training_inputs
    ):
        parser.error(
            "training requires --train-data, --train-labels, "
            "--validation-data, and --validation-labels"
        )
    if args.mode in ("all", "predict") and args.study_area_data is None:
        parser.error("prediction requires --study-area-data")

    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.batch_size < 1 or args.prediction_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.max_prediction_windows is not None and args.max_prediction_windows < 1:
        raise ValueError("max-prediction-windows must be positive")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must lie between zero and one")

    root = Path(args.project_root).resolve()

    def resolve_path(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    train_data_path = resolve_path(args.train_data) if args.train_data else None
    train_labels_path = resolve_path(args.train_labels) if args.train_labels else None
    validation_data_path = (
        resolve_path(args.validation_data) if args.validation_data else None
    )
    validation_labels_path = (
        resolve_path(args.validation_labels) if args.validation_labels else None
    )
    study_area_path = (
        resolve_path(args.study_area_data) if args.study_area_data else None
    )
    x_coordinates_path = (
        resolve_path(args.x_coordinates) if args.x_coordinates else None
    )
    y_coordinates_path = (
        resolve_path(args.y_coordinates) if args.y_coordinates else None
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (root / "model_training_and_prediction" / model_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else (output_dir / f"best_{model_name}.pth")
    )
    scaler_path = (
        Path(args.scaler).resolve()
        if args.scaler
        else output_dir / "train_scaler.joblib"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    def load_samples():
        train_x = np.load(train_data_path).astype(np.float32)
        train_y = np.load(train_labels_path).astype(np.int64)
        valid_x = np.load(validation_data_path).astype(np.float32)
        valid_y = np.load(validation_labels_path).astype(np.int64)
        for name, array in (("train_data", train_x), ("verify_data", valid_x)):
            if array.ndim != 4 or array.shape[1:] != (9, 9, 42):
                raise ValueError(
                    f"{name} must have shape (N,9,9,42), got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        if len(train_x) != len(train_y) or len(valid_x) != len(valid_y):
            raise ValueError("sample and label counts do not match")
        return train_x, train_y, valid_x, valid_y

    def to_tensor(array):
        return torch.from_numpy(array).float().permute(0, 3, 1, 2)

    def objective(model, inputs, labels, criterion):
        if hasattr(model, "attribution_network"):
            logits, _, factor_info, branches = model(
                inputs,
                return_attribution=True,
                return_factor_attribution=True,
                return_branch_logits=True,
            )
            factor_targets = (
                labels.float() * (1.0 - args.label_smoothing)
                + 0.5 * args.label_smoothing
            )
            factor_loss = nn.functional.binary_cross_entropy_with_logits(
                factor_info["branch_logit"].squeeze(-1), factor_targets
            )
            transformer_loss = criterion(branches["transformer"], labels)
            loss = (
                criterion(logits, labels)
                + args.factor_loss_weight * factor_loss
                + args.transformer_loss_weight * transformer_loss
            )
            factor_branch = model.attribution_network.factor_branch
            if hasattr(factor_branch, "linear"):
                weight = factor_branch.linear.weight
                loss = loss + (
                    args.factor_l1_weight * weight.abs().sum()
                    + args.factor_l2_weight * weight.square().sum()
                )
            return logits, loss
        if hasattr(model, "logit_fusion"):
            logits, branches = model(inputs, return_branch_logits=True)
            loss = criterion(logits, labels) + (
                args.transformer_loss_weight
                * criterion(branches["transformer"], labels)
            )
            return logits, loss
        logits = model(inputs)
        return logits, criterion(logits, labels)

    def validation(model, loader, criterion):
        model.eval()
        total_loss = 0.0
        labels_all, probability_all = [], []
        with torch.inference_mode():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                total_loss += criterion(logits, labels).item() * len(labels)
                probability_all.extend(
                    torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                )
                labels_all.extend(labels.cpu().tolist())
        labels_array = np.asarray(labels_all, dtype=np.int64)
        probability_array = np.asarray(probability_all, dtype=np.float64)
        predicted = (probability_array >= args.threshold).astype(np.int64)
        metrics = {
            "accuracy": float(accuracy_score(labels_array, predicted)),
            "precision": float(
                precision_score(labels_array, predicted, zero_division=0)
            ),
            "recall": float(recall_score(labels_array, predicted, zero_division=0)),
            "f1": float(f1_score(labels_array, predicted, zero_division=0)),
            "kappa": float(cohen_kappa_score(labels_array, predicted)),
            "mcc": float(matthews_corrcoef(labels_array, predicted)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(labels_array, probability_array))
        except ValueError:
            metrics["roc_auc"] = None
        return total_loss / max(len(labels_array), 1), metrics

    def train_model():
        train_x, train_y, valid_x, valid_y = load_samples()
        scaler = StandardScaler().fit(train_x.reshape(-1, 42))
        train_x = scaler.transform(train_x.reshape(-1, 42)).reshape(train_x.shape)
        valid_x = scaler.transform(valid_x.reshape(-1, 42)).reshape(valid_x.shape)
        train_x = train_x.astype(np.float32)
        valid_x = valid_x.astype(np.float32)
        joblib.dump(scaler, scaler_path)

        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            TensorDataset(to_tensor(train_x), torch.from_numpy(train_y)),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        )
        valid_loader = DataLoader(
            TensorDataset(to_tensor(valid_x), torch.from_numpy(valid_y)),
            batch_size=args.batch_size,
            shuffle=False,
        )
        model = model_factory().to(device)
        if hasattr(model, "attribution_network"):
            reference_loader = DataLoader(
                TensorDataset(to_tensor(train_x), torch.from_numpy(train_y)),
                batch_size=args.batch_size,
                shuffle=False,
            )
            model.attribution_network.update_reference_features(
                reference_loader, device
            )

        counts = np.bincount(train_y, minlength=2)
        class_weights = torch.tensor(
            len(train_y) / (2.0 * np.maximum(counts, 1)),
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=args.label_smoothing
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        best_loss = float("inf")
        best_epoch = 0
        best_metrics = None
        stale = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            if epoch <= args.warmup_epochs:
                ratio = epoch / max(args.warmup_epochs, 1)
            else:
                progress = (epoch - args.warmup_epochs) / max(
                    args.epochs - args.warmup_epochs, 1
                )
                ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_lr = args.learning_rate * ratio
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            model.train()
            running_loss = 0.0
            sample_count = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, loss = objective(model, inputs, labels, criterion)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * len(labels)
                sample_count += len(labels)

            valid_loss, metrics = validation(model, valid_loader, criterion)
            record = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": running_loss / max(sample_count, 1),
                "validation_loss": valid_loss,
                **metrics,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if valid_loss < best_loss - 1e-6:
                best_loss = valid_loss
                best_epoch = epoch
                best_metrics = dict(metrics)
                stale = 0
                torch.save(
                    {
                        "model_name": model_name,
                        "model_state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_loss,
                        "validation_metrics": metrics,
                        "threshold": args.threshold,
                    },
                    checkpoint_path,
                )
            else:
                stale += 1
                if stale >= args.patience:
                    break

        if not checkpoint_path.exists():
            raise RuntimeError("training did not produce a checkpoint")
        with open(
            output_dir / "training_history.csv", "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        with open(
            output_dir / "validation_metrics.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "model": model_name,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "best_validation_metrics": best_metrics,
                    "device": str(device),
                    "checkpoint": str(checkpoint_path),
                    "scaler": str(scaler_path),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        return checkpoint_path

    def coordinate_grid(shape):
        def read(path):
            if path is None or not path.exists():
                return None
            raster = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            return raster if raster is not None and raster.shape == shape else None

        return read(x_coordinates_path), read(y_coordinates_path)

    def predict_study_area():
        if study_area_path is None:
            raise ValueError("--study-area-data is required in prediction mode")
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_name = checkpoint.get("model_name", checkpoint.get("model"))
        if checkpoint_name not in (None, model_name):
            print(
                f"[checkpoint metadata] model={checkpoint_name}; "
                f"loading by strict parameter compatibility as {model_name}",
                flush=True,
            )
        model = model_factory().to(device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state_dict is None:
            raise KeyError("checkpoint contains no model_state_dict or state_dict")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        elif "scaler_mean" in checkpoint and "scaler_scale" in checkpoint:
            scaler = StandardScaler()
            scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
            scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
            scaler.var_ = scaler.scale_**2
            scaler.n_features_in_ = len(scaler.mean_)
            scaler.n_samples_seen_ = 1
        else:
            raise FileNotFoundError(
                f"No scaler at {scaler_path} and checkpoint has no scaler statistics"
            )
        cube = np.load(study_area_path, mmap_mode="r")
        if cube.ndim != 3 or cube.shape[2] != 42:
            raise ValueError(f"study-area data must be HxWx42, got {cube.shape}")

        prediction_mask = np.isfinite(cube).all(axis=2) & ~np.all(cube == 0, axis=2)
        prediction_mask[:4] = False
        prediction_mask[-4:] = False
        prediction_mask[:, :4] = False
        prediction_mask[:, -4:] = False
        centres = np.argwhere(prediction_mask)
        if len(centres) == 0:
            raise RuntimeError("no valid 9x9 prediction windows were found")
        if args.max_prediction_windows is not None:
            centres = centres[: args.max_prediction_windows]
        probabilities = np.empty(len(centres), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(centres), args.prediction_batch_size):
                stop = min(start + args.prediction_batch_size, len(centres))
                windows = np.stack(
                    [
                        cube[row - 4 : row + 5, col - 4 : col + 5, :]
                        for row, col in centres[start:stop]
                    ]
                ).astype(np.float32)
                windows = (
                    scaler.transform(windows.reshape(-1, 42))
                    .reshape(windows.shape)
                    .astype(np.float32)
                )
                inputs = to_tensor(windows).to(device)
                logits = model(inputs)
                probabilities[start:stop] = (
                    torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                )

        probability_map = np.zeros(cube.shape[:2], dtype=np.float32)
        probability_map[centres[:, 0], centres[:, 1]] = probabilities
        saved_mask = np.zeros(cube.shape[:2], dtype=bool)
        saved_mask[centres[:, 0], centres[:, 1]] = True
        np.save(output_dir / "mineralization_probability.npy", probability_map)
        np.save(output_dir / "valid_prediction_mask.npy", saved_mask)
        xx, yy = coordinate_grid(cube.shape[:2])
        csv_path = output_dir / "mineralization_probability.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "Row",
                    "Col",
                    "X",
                    "Y",
                    "Mineralization_Probability",
                    "Predicted_Class",
                    "Valid_Prediction",
                ]
            )
            height, width = cube.shape[:2]
            for row in range(height):
                for col in range(width):
                    probability = float(probability_map[row, col])
                    valid = bool(saved_mask[row, col])
                    writer.writerow(
                        [
                            row,
                            col,
                            "" if xx is None else float(xx[row, col]),
                            "" if yy is None else float(yy[row, col]),
                            probability,
                            int(valid and probability >= args.threshold),
                            valid,
                        ]
                    )
        metadata = {
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "valid_windows": int(len(centres)),
            "threshold": args.threshold,
            "probability_min": float(probabilities.min()),
            "probability_max": float(probabilities.max()),
            "probability_mean": float(probabilities.mean()),
            "csv": str(csv_path),
        }
        with open(
            output_dir / "prediction_metadata.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
        return csv_path

    if args.mode in ("all", "train"):
        train_model()
    if args.mode in ("all", "predict"):
        predict_study_area()
