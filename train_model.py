import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class InpaintDataset(Dataset):
    """
    Dataset class for inpainting images and binary masks.

    Uses Albumentations for data augmentation.
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = os.path.join(self.masks_dir, os.path.basename(image_path))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # Ensure mask is binary

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask.float()

class AttentionBlock(nn.Module):
    """
    Attention gate block for the Attention U-Net.
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture for binary segmentation.
    """

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        # Encoder
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)

        # Decoder + Attention Gate
        self.up_transpose = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.up_transpose.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.attention.append(
                AttentionBlock(F_g=feature, F_l=feature, F_int=feature // 2)
            )
            self.decoder.append(
                self.double_conv(feature*2, feature)
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through Attention U-Net.
        """
        skip_connections = []
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.up_transpose)):
            x = self.up_transpose[idx](x)
            skip_connection = skip_connections[idx]
            attn_skip = self.attention[idx](g=x, x=skip_connection)
            if x.shape != attn_skip.shape:
                x = nn.functional.interpolate(x, size=attn_skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([attn_skip, x], dim=1)
            x = self.decoder[idx](x)
        return self.final_conv(x)
    
    def double_conv(self, in_channels, out_channels):
        """
        Two-layer Conv-BN-ReLU block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class BCEDiceLoss(nn.Module):
    """
    Combination of BCE loss and Dice loss for segmentation.
    """

    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Compute BCE + Dice loss.
        """
        bce_loss = self.bce(inputs, targets)
        smooth = 1e-5
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50):
    """
    Train the model using provided dataloaders and loss.

    @return: Trained model with best weights saved.
    """
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_attention_unet.pth")
            print("Saved best model!")
    return model

def main_train():
    """
    Main entry point for training the model.
    """
    train_images_dir = "train/images"
    train_masks_dir = "train/masks"

    # Training data augmentation
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    # Validation data augmentation
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Dataset and split
    full_dataset = InpaintDataset(train_images_dir, train_masks_dir, transform=train_transform)
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = InpaintDataset(train_images_dir, train_masks_dir, transform=val_transform)
    val_subset = torch.utils.data.Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=3, out_channels=1)
    model = model.to(device)

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50)

if __name__ == "__main__":
    main_train()
