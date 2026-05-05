"""This script prepares training and validation samples for the mineral prospectivity study."""

import numpy as np
import cv2 as cv
import math
import os
import random
import glob
import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DATA_DIR   = r'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/New_data'
GEOCHEM_DIR     = os.path.join(BASE_DATA_DIR, 'Geochemical_data')
GEOLOGICAL_DIR  = os.path.join(BASE_DATA_DIR, 'Geological_data')
LABEL1_PATH     = os.path.join(BASE_DATA_DIR, 'label/Li_deposits0.tif')
LABEL0_PATH     = os.path.join(BASE_DATA_DIR, 'label/non_deposits.tif')
SAMPLE_DIR      = os.path.join(BASE_DATA_DIR, 'sample')
TRAIN_DIR       = os.path.join(BASE_DATA_DIR, 'train_extend')
VERIFY_DIR      = os.path.join(BASE_DATA_DIR, 'verify_extend')

# ── Hyperparameters ───────────────────────────────────────────────────────────
WINDOW_SIZE    = 9
ALL_CHANNEL    = 42      # 39 geochemical elements + 3 ore-controlling factors
AUGMENT_SIZE   = 3       # spatial augmentation kernel size
VAL_RATIO      = 0.2


def read_geochemical_data():
    """Load all geochemical raster layers and stack them into a single array."""
    files = os.listdir(GEOCHEM_DIR)
    layers = []
    for fname in files:
        arr = cv.imread(os.path.join(GEOCHEM_DIR, fname), 2)
        layers.append(arr.reshape(arr.shape[0], arr.shape[1], 1))
    return np.concatenate(layers, axis=-1)


def read_geological_features():
    """Load all geological feature raster layers and stack them into a single array."""
    files = os.listdir(GEOLOGICAL_DIR)
    layers = []
    for fname in files:
        arr = cv.imread(os.path.join(GEOLOGICAL_DIR, fname), 2)
        layers.append(arr.reshape(arr.shape[0], arr.shape[1], 1))
    return np.concatenate(layers, axis=-1)


def split_train_val(label1_coordinate):
    """Split deposit point indices into train / validation sets (80/20)."""
    orig_count  = len(label1_coordinate[0])
    all_indices = list(range(orig_count))
    random.shuffle(all_indices)
    val_count  = max(1, int(orig_count * VAL_RATIO))
    train_idx  = all_indices[val_count:]
    val_idx    = all_indices[:val_count]

    print(f'Total deposit points: {orig_count}')
    print(f'  Training set: {len(train_idx)} points, indices: {sorted(train_idx)}')
    print(f'  Validation set: {len(val_idx)} points, indices: {sorted(val_idx)}')
    return train_idx, val_idx, orig_count


def build_augmented_labels(label1, label0, label1_coordinate, label0_coordinate,
                            train_idx, val_idx):
    """Apply 3×3 spatial augmentation to each split independently."""
    half = math.floor(AUGMENT_SIZE / 2)

    # positive samples: background=0, augmented region=1
    label1_train_aug  = np.zeros_like(label1)
    label1_verify_aug = np.zeros_like(label1)
    # negative samples: background=1, augmented region=0
    label0_train_aug  = np.ones_like(label0)
    label0_verify_aug = np.ones_like(label0)

    for i in train_idx:
        r1, c1 = label1_coordinate[0][i], label1_coordinate[1][i]
        label1_train_aug[r1 - half: r1 + half + 1, c1 - half: c1 + half + 1] = 1
        r0, c0 = label0_coordinate[0][i], label0_coordinate[1][i]
        label0_train_aug[r0 - half: r0 + half + 1, c0 - half: c0 + half + 1] = 0

    for i in val_idx:
        r1, c1 = label1_coordinate[0][i], label1_coordinate[1][i]
        label1_verify_aug[r1 - half: r1 + half + 1, c1 - half: c1 + half + 1] = 1
        r0, c0 = label0_coordinate[0][i], label0_coordinate[1][i]
        label0_verify_aug[r0 - half: r0 + half + 1, c0 - half: c0 + half + 1] = 0

    index1_train  = np.argwhere(label1_train_aug  == 1)
    index0_train  = np.argwhere(label0_train_aug  == 0)
    index1_verify = np.argwhere(label1_verify_aug == 1)
    index0_verify = np.argwhere(label0_verify_aug == 0)

    print(f'\nAugmented training set:   positive={len(index1_train)}, negative={len(index0_train)}')
    print(f'Augmented validation set: positive={len(index1_verify)}, negative={len(index0_verify)}')
    return index1_train, index0_train, index1_verify, index0_verify


def save_original_samples(combined_array, label1_coordinate, label0_coordinate, orig_count):
    """Save raw (non-augmented) patch samples for archival purposes."""
    hw = math.floor(WINDOW_SIZE / 2)
    for i in range(orig_count):
        np.save(
            os.path.join(SAMPLE_DIR, '1', str(i)),
            combined_array[label1_coordinate[0][i] - hw: label1_coordinate[0][i] + hw + 1,
                           label1_coordinate[1][i] - hw: label1_coordinate[1][i] + hw + 1, :]
        )
        np.save(
            os.path.join(SAMPLE_DIR, '0', str(i)),
            combined_array[label0_coordinate[0][i] - hw: label0_coordinate[0][i] + hw + 1,
                           label0_coordinate[1][i] - hw: label0_coordinate[1][i] + hw + 1, :]
        )


def save_augmented_samples(combined_array, index1, index0, output_dir):
    """Extract and save augmented patch samples for a given split."""
    hw = math.floor(WINDOW_SIZE / 2)
    for num in range(len(index1)):
        np.save(
            os.path.join(output_dir, '1', str(num)),
            combined_array[index1[num][0] - hw: index1[num][0] + hw + 1,
                           index1[num][1] - hw: index1[num][1] + hw + 1, :]
        )
        np.save(
            os.path.join(output_dir, '0', str(num)),
            combined_array[index0[num][0] - hw: index0[num][0] + hw + 1,
                           index0[num][1] - hw: index0[num][1] + hw + 1, :]
        )


def build_dataset_arrays(sample_dir, tag):
    """Load all .npy patches from a directory and return data + label arrays."""
    data_list   = []
    label_list  = []
    npy_paths   = glob.glob(os.path.join(sample_dir, '*/*.npy'))
    for path in npy_paths:
        patch = np.load(path)
        data_list.append(patch.reshape(1, WINDOW_SIZE, WINDOW_SIZE, ALL_CHANNEL))
        label_list.append(path.split('\\')[-2])
    data_array  = np.concatenate(data_list, axis=0)
    label_array = pd.factorize(label_list)[0]
    print(f'{tag} samples: {data_array.shape[0]}')
    return data_array, label_array


def main():
    print('=' * 60)
    print('Sample preparation for mineral prospectivity study')
    print('=' * 60)

    # ── Step 1: Load and merge raster data ───────────────────────────────────
    print('\n=== Loading raster data ===')
    geochemical_array = read_geochemical_data()
    geological_array  = read_geological_features()
    combined_array    = np.concatenate([geochemical_array, geological_array], axis=-1)
    print(f'Combined array shape: {combined_array.shape}')
    np.save('Combined_array.npy', combined_array)

    # ── Step 2: Load label rasters ────────────────────────────────────────────
    print('\n=== Loading label rasters ===')
    label1 = cv.imread(LABEL1_PATH, 2)
    label0 = cv.imread(LABEL0_PATH, 2)
    label1_coordinate = np.where(label1 == 1)
    label0_coordinate = np.where(label0 == 0)

    # ── Step 3: Train / validation split ─────────────────────────────────────
    print('\n=== Splitting train / validation sets ===')
    train_idx, val_idx, orig_count = split_train_val(label1_coordinate)

    # ── Step 4: Spatial augmentation ─────────────────────────────────────────
    print('\n=== Building augmented label maps ===')
    index1_train, index0_train, index1_verify, index0_verify = build_augmented_labels(
        label1, label0, label1_coordinate, label0_coordinate, train_idx, val_idx
    )

    # ── Step 5: Create output directories ────────────────────────────────────
    for d in [
        os.path.join(SAMPLE_DIR, '1'),
        os.path.join(SAMPLE_DIR, '0'),
        os.path.join(TRAIN_DIR,  '1'),
        os.path.join(TRAIN_DIR,  '0'),
        os.path.join(VERIFY_DIR, '1'),
        os.path.join(VERIFY_DIR, '0'),
    ]:
        os.makedirs(d, exist_ok=True)

    # ── Step 6: Save patches ──────────────────────────────────────────────────
    print('\n=== Saving original samples (archive) ===')
    save_original_samples(combined_array, label1_coordinate, label0_coordinate, orig_count)

    print('\n=== Saving augmented training samples ===')
    save_augmented_samples(combined_array, index1_train,  index0_train,  TRAIN_DIR)

    print('\n=== Saving augmented validation samples ===')
    save_augmented_samples(combined_array, index1_verify, index0_verify, VERIFY_DIR)

    # ── Step 7: Build dataset arrays and save ────────────────────────────────
    print('\n=== Building dataset arrays ===')
    train_data_array,  train_label_array  = build_dataset_arrays(TRAIN_DIR,  'Training')
    verify_data_array, verify_label_array = build_dataset_arrays(VERIFY_DIR, 'Validation')

    np.save('train_data.npy',    train_data_array)
    np.save('train_labels.npy',  train_label_array)
    np.save('verify_data.npy',   verify_data_array)
    np.save('verify_labels.npy', verify_label_array)

    print(f'\nOutput directory: {os.getcwd()}')
    print('=' * 60)
    print('Sample preparation complete.')
    print('=' * 60)


if __name__ == '__main__':
    main()
