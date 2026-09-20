"""Build spatially separated samples for model development and evaluation."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse.csgraph import connected_components

WINDOW = 9
HALF = WINDOW // 2
GROUP_RADIUS = 12
SEED = 2026


def spatial_groups(coords, radius):
    distance = np.max(np.abs(coords[:, None, :] - coords[None, :, :]), axis=2)
    _, groups = connected_components(
        (distance <= radius).astype(np.uint8), directed=False
    )
    return groups.astype(np.int64)


def choose_test_groups(groups, target_samples, target_groups, rng):
    ids, sizes = np.unique(groups, return_counts=True)
    candidates = []
    for count in range(1, min(len(ids), target_groups + 2) + 1):
        for selected in itertools.combinations(ids.tolist(), count):
            sample_count = sum(
                sizes[np.where(ids == group)[0][0]] for group in selected
            )
            score = abs(sample_count - target_samples) * 10 + abs(count - target_groups)
            candidates.append((score, selected))
    best_score = min(item[0] for item in candidates)
    best = [item[1] for item in candidates if item[0] == best_score]
    return np.asarray(best[int(rng.integers(len(best)))], dtype=np.int64)


def patches(cube, coords):
    return np.stack(
        [cube[r - HALF : r + HALF + 1, c - HALF : c + HALF + 1, :] for r, c in coords]
    ).astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", required=True)
    parser.add_argument("--positive-labels", required=True)
    parser.add_argument("--negative-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cube = np.load(Path(args.cube).resolve(), mmap_mode="r")
    positive_raster = cv2.imread(
        str(Path(args.positive_labels).resolve()), cv2.IMREAD_UNCHANGED
    )
    negative_raster = cv2.imread(
        str(Path(args.negative_labels).resolve()), cv2.IMREAD_UNCHANGED
    )
    if positive_raster is None or negative_raster is None:
        raise FileNotFoundError("Original label rasters were not found")

    positive = np.argwhere(positive_raster == 1).astype(np.int64)
    negative = np.argwhere(negative_raster == 0).astype(np.int64)
    coords = np.concatenate([negative, positive])
    labels = np.concatenate(
        [
            np.zeros(len(negative), dtype=np.int64),
            np.ones(len(positive), dtype=np.int64),
        ]
    )
    if (
        np.any(coords < HALF)
        or np.any(coords[:, 0] >= cube.shape[0] - HALF)
        or np.any(coords[:, 1] >= cube.shape[1] - HALF)
    ):
        raise ValueError("At least one label point is too close to the raster boundary")

    neg_groups = spatial_groups(negative, GROUP_RADIUS)
    pos_groups = spatial_groups(positive, GROUP_RADIUS)
    groups = np.concatenate([neg_groups, pos_groups + neg_groups.max() + 1])
    rng = np.random.default_rng(SEED)
    neg_test_groups = choose_test_groups(
        neg_groups, target_samples=5, target_groups=5, rng=rng
    )
    pos_offset = neg_groups.max() + 1
    pos_test_local = choose_test_groups(
        pos_groups, target_samples=5, target_groups=3, rng=rng
    )
    test_groups = np.concatenate([neg_test_groups, pos_test_local + pos_offset])
    test_mask = np.isin(groups, test_groups)
    dev_mask = ~test_mask

    dev_coords, test_coords = coords[dev_mask], coords[test_mask]
    dev_labels, test_labels = labels[dev_mask], labels[test_mask]
    dev_groups, test_group_values = groups[dev_mask], groups[test_mask]
    distance = np.max(np.abs(test_coords[:, None, :] - dev_coords[None, :, :]), axis=2)
    if distance.min() <= GROUP_RADIUS:
        raise RuntimeError("Spatial group split failed to enforce the requested buffer")

    np.save(output_dir / "development_data.npy", patches(cube, dev_coords))
    np.save(output_dir / "development_labels.npy", dev_labels)
    np.save(output_dir / "development_coordinates.npy", dev_coords)
    np.save(output_dir / "development_groups.npy", dev_groups)
    np.save(output_dir / "independent_test_data.npy", patches(cube, test_coords))
    np.save(output_dir / "independent_test_labels.npy", test_labels)
    np.save(output_dir / "independent_test_coordinates.npy", test_coords)
    np.save(output_dir / "independent_test_groups.npy", test_group_values)

    with open(
        output_dir / "sample_manifest.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "label", "group_id", "row", "col"]
        )
        writer.writeheader()
        for split, ys, gs, cs in [
            ("development", dev_labels, dev_groups, dev_coords),
            ("independent_test", test_labels, test_group_values, test_coords),
        ]:
            for y, g, (r, c) in zip(ys, gs, cs):
                writer.writerow(
                    {
                        "split": split,
                        "label": int(y),
                        "group_id": int(g),
                        "row": int(r),
                        "col": int(c),
                    }
                )

    audit = {
        "seed": SEED,
        "window_size": WINDOW,
        "group_radius_pixels": GROUP_RADIUS,
        "cube_shape": list(cube.shape),
        "label_raster_shape": list(positive_raster.shape),
        "source_samples": int(len(coords)),
        "development_samples": int(dev_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "development_class_counts": np.bincount(dev_labels, minlength=2).tolist(),
        "test_class_counts": np.bincount(test_labels, minlength=2).tolist(),
        "positive_spatial_groups": int(len(np.unique(pos_groups))),
        "negative_spatial_groups": int(len(np.unique(neg_groups))),
        "minimum_test_to_development_chebyshev_distance": int(distance.min()),
        "test_windows_augmented": False,
        "augmentation_policy": "Apply 3x3 center shifts only to each CV training fold.",
    }
    with open(output_dir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
