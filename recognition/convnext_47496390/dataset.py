# https://share.gemini.google/3vBDdHud5qyP
import os
import glob
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ADNISliceDataset(Dataset):
    """
    Custom Dataset for ADNI 2D MRI slices.
    Expects a list of file paths and their associated class labels.
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

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


def get_transforms(for_rgb_backbone=True):
    """
    Returns training and validation/test transformation pipelines.
    Set for_rgb_backbone=True if using architectures like ConvNeXt/ResNet
    that expect 3-channel (RGB) inputs.
    """
    train_ops = [
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1,contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
    ]

    val_ops = [
        transforms.ToTensor(),
    ]

    if for_rgb_backbone:
        # Replicate single channel across 3 channels and apply ImageNet normalization
        train_ops.extend([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_ops.extend([
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Standard grayscale normalization (mean=0.5, std=0.5)
        train_ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        val_ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    train_transform = transforms.Compose(train_ops)
    val_transform = transforms.Compose(val_ops)

    return train_transform, val_transform


def build_dataloaders(
    rgb_backbone,
    data_root="AD_NC",
    val_split=0.2,
    batch_size=32,
    num_workers=4,
    seed=42
):
    classes = {'NC': 0, 'AD': 1}
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    # ---------------------------------------------------------
    # 1. Group training slices by Patient ID to prevent leakage
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

    # Stratified split ensures balanced AD/NC proportions among patients
    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=val_split,
        stratify=patient_labels,
        random_state=seed
    )

    # Flatten patient slices back into image lists
    train_paths, train_labels = [], []
    for pid in train_patients:
        for path in patient_slices[pid]:
            train_paths.append(path)
            train_labels.append(patient_label[pid])

    val_paths, val_labels = [], []
    for pid in val_patients:
        for path in patient_slices[pid]:
            val_paths.append(path)
            val_labels.append(patient_label[pid])

    # ---------------------------------------------------------
    # 2. Collect test slices
    # ---------------------------------------------------------
    test_paths, test_labels = [], []
    for cls_name, cls_idx in classes.items():
        cls_folder = os.path.join(test_dir, cls_name)
        for file_path in glob.glob(os.path.join(cls_folder, "*.jpeg")):
            test_paths.append(file_path)
            test_labels.append(cls_idx)

    print(f"Dataset summary:")
    print(f"  Train: {len(train_paths)} slices ({len(train_patients)} patients)")
    print(f"  Val:   {len(val_paths)} slices ({len(val_patients)} patients)")
    print(f"  Test:  {len(test_paths)} slices")

    # ---------------------------------------------------------
    # 3. Create datasets and loaders
    # ---------------------------------------------------------
    train_tf, val_tf = get_transforms(for_rgb_backbone=rgb_backbone)

    train_dataset = ADNISliceDataset(train_paths, train_labels, transform=train_tf)
    val_dataset = ADNISliceDataset(val_paths, val_labels, transform=val_tf)
    test_dataset = ADNISliceDataset(test_paths, test_labels, transform=val_tf)

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
        False,
        data_root="AD_NC",
        val_split=0.2,
        batch_size=32
    )

    # Verification batch
    images, labels = next(iter(train_loader))
    print("\nBatch tensor shapes:")
    print("Images shape:", images.shape)   # [32, 3/1, 256, 240]
    print("Labels shape:", labels.shape)   # [32,]
