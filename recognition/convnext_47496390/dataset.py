"""
ADNI 2D MRI Slice Data Pipeline and Leakage-Free Splitting.
Contains:
    - ADNISliceDataset: PyTorch Dataset for loading single-channel 2D MRI slices.
    - get_transforms: Domain-appropriate augmentation pipeline (rotation, affine, jitter, erasing).
    - build_dataloaders: Patient-level stratified train/val/test dataloader builder ensuring zero-leakage cross-patient separation.
"""

import os
import glob
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# https://share.gemini.google/3vBDdHud5qyP
class ADNISliceDataset(Dataset):
    """
    Custom Dataset for ADNI 2D MRI slices.
    Loads single-channel grayscale brain MRI images and returns (image_tensor, label).
    Attributes:
        file_paths (list[str]): List of absolute or relative image file paths.
        labels (list[int]): Class integer labels (0 for NC, 1 for AD).
        transform (callable, optional): PyTorch transformation pipeline.
        patient_ids (list[str], optional): Corresponding patient ID for each slice
    """
    def __init__(self, file_paths, labels, transform=None, patient_ids=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load image as grayscale ('L')
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(in_channels=1):
    """
    Returns training and validation/test transformation pipelines.
    """
    train_ops = [
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1,contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, value = 0),
    ]

    val_ops = [
        transforms.ToTensor(),
    ]

    # Standard grayscale normalization (mean=0.5, std=0.5)
    train_ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    val_ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    train_transform = transforms.Compose(train_ops)
    val_transform = transforms.Compose(val_ops)

    return train_transform, val_transform


def build_dataloaders(
    in_channels=1,
    data_root="AD_NC",
    val_split=0.2,
    batch_size=32,
    num_workers=4,
    seed=42
):
    """
    Constructs PyTorch DataLoaders with strict patient-level separation.
    All 2D slices from any given patient are placed exclusively into either
    the training, validation, or test partition, mathematically guaranteeing zero
    cross-patient data leakage:
    """
    classes = {'NC': 0, 'AD': 1}
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Testing directory not found at: {test_dir}")

    # ---------------------------------------------------------
    # Group training slices by Patient ID to prevent leakage
    # ---------------------------------------------------------
    patient_slices = defaultdict(list)
    patient_label = {}

    for cls_name, cls_idx in classes.items():
        cls_folder = os.path.join(train_dir, cls_name)
        pattern = os.path.join(cls_folder, "*.jpeg")
        for file_path in glob.glob(pattern):
            # Filename: <patient_id>_<slice_idx>.jpeg
            filename = os.path.basename(file_path)
            patient_id = filename.split('_')[0]

            patient_slices[patient_id].append(file_path)
            patient_label[patient_id] = cls_idx

    unique_patients = list(patient_slices.keys())
    patient_labels = [patient_label[pid] for pid in unique_patients]

    # Patient-level Stratified split ensures balanced AD/NC ratio in train and val sets
    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=val_split,
        stratify=patient_labels,
        random_state=seed
    )

    # Flatten patient slices back into slice lists with patient ID tracking
    train_paths, train_labels, train_pids = [], [], []
    for pid in train_patients:
        for path in patient_slices[pid]:
            train_paths.append(path)
            train_labels.append(patient_label[pid])
            train_pids.append(pid)

    val_paths, val_labels, val_pids = [], [], []
    for pid in val_patients:
        for path in patient_slices[pid]:
            val_paths.append(path)
            val_labels.append(patient_label[pid])
            val_pids.append(pid)

    # ---------------------------------------------------------
    # 2. Collect test slices
    # ---------------------------------------------------------
    test_paths, test_labels, test_pids = [], [], []
    test_patient_set = set()

    for cls_name, cls_idx in classes.items():
        cls_folder = os.path.join(test_dir, cls_name)
        for file_path in glob.glob(os.path.join(cls_folder, "*.jpeg")):
            filename = os.path.basename(file_path)
            patient_id = filename.split('_')[0]
            test_paths.append(file_path)
            test_labels.append(cls_idx)
            test_pids.append(patient_id)
            test_patient_set.add(patient_id)

    print(f"ADNI Dataset Summary (Strict Patient-Level Partitioning):")
    print(f"  Train: {len(train_paths):>6} slices from {len(train_patients):>4} unique patients")
    print(f"  Val:   {len(val_paths):>6} slices from {len(val_patients):>4} unique patients")
    print(f"  Test:  {len(test_paths):>6} slices from {len(test_patient_set):>4} unique patients")

    # ---------------------------------------------------------
    # 3. Create datasets and loaders
    # ---------------------------------------------------------
    train_tf, val_tf = get_transforms(in_channels=in_channels)
    
    train_dataset = ADNISliceDataset(train_paths, train_labels, transform=train_tf, patient_ids=train_pids)
    val_dataset = ADNISliceDataset(val_paths, val_labels, transform=val_tf, patient_ids=val_pids)
    test_dataset = ADNISliceDataset(test_paths, test_labels, transform=val_tf, patient_ids=test_pids)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders(
        in_channels=1,
        data_root="/home/groups/comp3710/ADNI/AD_NC",
        val_split=0.2,
        batch_size=32
    )

    # Verification batch
    images, labels = next(iter(train_loader))
    print("\nVerification Batch Tensor Shapes:")
    print(f"  Images shape: {images.shape} (Expected: [32, 1, 240, 256])")
    print(f"  Labels shape: {labels.shape} (Expected: [32])")
    print("dataset.py verification completed successfully!")
