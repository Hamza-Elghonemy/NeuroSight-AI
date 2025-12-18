import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
from tqdm import tqdm
from skimage.util import random_noise 

# --- Config ---
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 50  # Increased for better convergence
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_unet_enhanced.pth')

print(f"Using device: {DEVICE}")
print(f"Data Root: {DATA_ROOT}")

# --- Dataset ---
class HybridLesionDataset(Dataset):
    def __init__(self, background_dir, img_size=224, num_samples=3000):  # Significantly increased samples
        self.img_size = img_size
        self.num_samples = num_samples
        
        # Look for Normal images
        self.bg_files = glob.glob(os.path.join(background_dir, '**', '*.png'), recursive=True)
        if not self.bg_files:
            self.bg_files = glob.glob(os.path.join(background_dir, '*.png'))
            if not self.bg_files:
                self.bg_files = glob.glob(os.path.join(background_dir, '**', '*.jpg'), recursive=True)
                if not self.bg_files:
                    self.bg_files = glob.glob(os.path.join(background_dir, '*.jpg'))
        
        print(f"Hybrid Generator: Found {len(self.bg_files)} background images in {background_dir}")
        
        # Shared augmentations (same random seed for img and mask)
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, shear=10),
        ])
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _create_random_lesion_mask(self, shape):
        mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
        lesion_type = random.choice(['hemorrhage', 'ischemia', 'tumor'])
        
        num_lesions = np.random.randint(1, 4)  # Multiple lesions possible
        for _ in range(num_lesions):
            center_x = np.random.randint(30, shape[1] - 30)
            center_y = np.random.randint(30, shape[0] - 30)
            
            if lesion_type == 'hemorrhage':
                # Irregular bleed with extensions
                axes_x, axes_y = np.random.randint(15, 50), np.random.randint(15, 50)
                angle = np.random.randint(0, 360)
                cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, 1.0, -1)
                # Add finger-like extensions
                for __ in range(np.random.randint(0, 3)):
                    end_x = np.clip(center_x + np.random.randint(-60, 60), 10, shape[1]-10)
                    end_y = np.clip(center_y + np.random.randint(-60, 60), 10, shape[0]-10)
                    thickness = np.random.randint(5, 15)
                    cv2.line(mask, (center_x, center_y), (end_x, end_y), 1.0, thickness=thickness)
            
            elif lesion_type == 'ischemia':
                # Territorial/wedge shape with soft edges
                radius = np.random.randint(30, 80)
                points = []
                for i in range(6):
                    angle = i * 60 + np.random.randint(-30, 30)
                    dx = int(radius * np.cos(np.deg2rad(angle)))
                    dy = int(radius * np.sin(np.deg2rad(angle)))
                    points.append([center_x + dx, center_y + dy])
                points = np.array([points], dtype=np.int32)
                cv2.fillPoly(mask, points, 1.0)
                mask = cv2.GaussianBlur(mask, (31, 31), 0)  # Very soft edges typical of infarcts
            
            else:  # tumor
                # Rounded mass, sometimes with ring (necrosis)
                axes_x, axes_y = np.random.randint(20, 70), np.random.randint(20, 70)
                angle = np.random.randint(0, 360)
                cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, 1.0, -1)
                if np.random.rand() > 0.6:  # Ring enhancement
                    inner_x, inner_y = int(axes_x * 0.65), int(axes_y * 0.65)
                    cv2.ellipse(mask, (center_x, center_y), (inner_x, inner_y), angle, 0, 360, -1.0, -1)
        
        # Add morphological irregularity
        kernel_size = np.random.randint(3, 11)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if np.random.rand() > 0.5:
            mask = cv2.dilate(mask, kernel, iterations=np.random.randint(1, 3))
        else:
            mask = cv2.erode(mask, kernel, iterations=np.random.randint(1, 2))
        
        # Subtle noise for natural boundaries
        noise = random_noise(np.zeros_like(mask), mode='gaussian', var=0.02)
        mask = mask + noise
        mask = np.clip(mask, 0, 1)
        
        return mask.astype(np.float32), lesion_type

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 1. Load background
        if self.bg_files:
            bg_path = random.choice(self.bg_files)
            img = cv2.imread(bg_path)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (self.img_size, self.img_size))
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Occasionally return normal image (no lesion)
        if np.random.rand() < 0.15:
            blended = img
            mask_np = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        else:
            # 2. Generate lesion
            mask_np, lesion_type = self._create_random_lesion_mask(img.shape[:2])
            
            # 3. Inject lesion with type-specific appearance
            img_float = img.astype(np.float32)
            mask_blurred = cv2.GaussianBlur(mask_np, (15, 15), 0)
            mask_3ch = np.stack([mask_blurred] * 3, axis=2)
            
            if lesion_type == 'hemorrhage':
                intensity = np.random.uniform(60.0, 100.0)
                blended = img_float + (mask_3ch * intensity)
            elif lesion_type == 'ischemia':
                intensity = np.random.uniform(25.0, 60.0)
                blended = img_float - (mask_3ch * intensity)
            else:  # tumor
                base_intensity = np.random.uniform(30.0, 80.0)
                blended = img_float + (mask_3ch * base_intensity)
                # Add heterogeneity
                speckle = random_noise(np.zeros_like(mask_np), mode='gaussian', var=0.1)
                speckle = (speckle - 0.5) * 40
                blended += mask_3ch[..., :1] * speckle[..., np.newaxis]
            
            blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # 4. Augmentations (applied identically to image and mask)
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        img_pil = Image.fromarray(blended)
        img_aug = self.aug(img_pil)
        
        torch.manual_seed(seed)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_aug = self.aug(mask_pil)
        
        img_tensor = self.to_tensor(img_aug)
        img_tensor = self.normalize(img_tensor)
        mask_tensor = self.to_tensor(mask_aug)[:1, ...]  # Keep only 1 channel
        
        return img_tensor, mask_tensor

# --- Model (Deeper UNet) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up3(x)
        
        return self.outc(x)

# --- Loss Functions ---
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# --- Main Training ---
def train():
    bg_folder = os.path.join(DATA_ROOT, 'Brain_Stroke_CT_Dataset', 'Normal')
    if not os.path.exists(bg_folder):
        print(f"Error: Background folder not found at {bg_folder}")
        return

    dataset = HybridLesionDataset(bg_folder, img_size=IMG_SIZE, num_samples=3000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = UNet().to(DEVICE)
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    print("Starting training with improved synthetic data...")
    
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(imgs)
            loss_bce = criterion_bce(pred, masks)
            loss_dice = dice_loss(pred, masks)
            loss = loss_bce + loss_dice
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f} (BCE: {loss_bce.item():.4f}, Dice: {loss_dice.item():.4f})")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  >>> New best model saved!")

    print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()