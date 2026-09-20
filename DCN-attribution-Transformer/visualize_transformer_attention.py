"""Generate study-area maps of received Transformer attention."""


import argparse
import csv
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from attribution_guided_model import create_attribution_guided_model


WINDOW_SIZE = 9
HALF_WINDOW = WINDOW_SIZE // 2
BATCH_SIZE = 512


def load_model(device, model_path):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("model_config", {})
    model = create_attribution_guided_model(
        num_classes=int(config.get("num_classes", 2)),
        feature_dim=int(config.get("feature_dim", 32)),
        transformer_depth=int(config.get("transformer_depth", 1)),
        transformer_heads=int(config.get("transformer_heads", 2)),
        factor_pooling=config.get("factor_pooling", "mean"),
        factor_branch_type=config.get("factor_branch_type", "linear"),
        dropout=float(config.get("dropout", 0.30)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def valid_centres(data):
    finite = np.isfinite(data).all(axis=2)
    nonempty = ~np.all(data == 0, axis=2)
    mask = finite & nonempty
    mask[:HALF_WINDOW] = False
    mask[-HALF_WINDOW:] = False
    mask[:, :HALF_WINDOW] = False
    mask[:, -HALF_WINDOW:] = False
    return np.argwhere(mask)


def extract_windows(data, centres):
    windows = np.stack(
        [
            data[
                row - HALF_WINDOW : row + HALF_WINDOW + 1,
                col - HALF_WINDOW : col + HALF_WINDOW + 1,
                :,
            ]
            for row, col in centres
        ]
    ).astype(np.float32)
    return windows


def received_attention(attention_maps):
    """Return mean received attention with shape (batch, 81)."""
    if not attention_maps:
        raise RuntimeError("The model returned no Transformer attention maps")
    per_layer = []
    for attention in attention_maps:
        if attention.ndim != 4:
            raise ValueError(
                f"Expected attention shape (B,heads,N,N), got {attention.shape}"
            )
        per_layer.append(attention.mean(dim=(1, 2)))
    return torch.stack(per_layer, dim=0).mean(dim=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(os.path.abspath(args.scaler))
    model = load_model(device, os.path.abspath(args.model))
    data = np.load(os.path.abspath(args.data), mmap_mode="r")
    centres = valid_centres(data)
    height, width, channels = data.shape
    regional = np.full((height, width), np.nan, dtype=np.float32)
    central_token = (WINDOW_SIZE * WINDOW_SIZE) // 2
    rows = []

    for start in tqdm(
        range(0, len(centres), BATCH_SIZE),
        desc="Transformer attention",
        ncols=78,
    ):
        batch_centres = centres[start : start + BATCH_SIZE]
        windows = extract_windows(data, batch_centres)
        scaled = scaler.transform(windows.reshape(-1, channels)).reshape(windows.shape)
        inputs = torch.from_numpy(scaled).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            _, maps = model(inputs, return_attention=True)
            scores = received_attention(maps).cpu().numpy()
        centre_scores = scores[:, central_token]
        for (row, col), value in zip(batch_centres, centre_scores):
            regional[row, col] = value
            rows.append((int(row), int(col), float(value)))

    np.save(os.path.join(output_dir, "attention_received_center.npy"), regional)
    with open(
        os.path.join(output_dir, "attention_received_center.csv"),
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "col", "mean_received_attention"])
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(9, 7), facecolor="white")
    image = axis.imshow(regional, cmap="magma", interpolation="nearest")
    axis.set_title("Transformer mean received attention: central token")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    figure.colorbar(image, ax=axis, label="Mean received attention")
    figure.tight_layout()
    figure.savefig(
        os.path.join(output_dir, "attention_received_center.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
