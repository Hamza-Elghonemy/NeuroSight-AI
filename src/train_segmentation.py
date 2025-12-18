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

# --- Config ---
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 15  # Increased epochs for better convergence with mixed types
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
# Path correction: script is in src/, so data is in ../data
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_unet.pth')

print(f"Using device: {DEVICE}")
print(f"Data Root: {DATA_ROOT}")

# --- Dataset ---
class HybridLesionDataset(Dataset):
    def __init__(self, background_dir, img_size=224, num_samples=600): # Increased samples
        self.img_size = img_size
        self.num_samples = num_samples
        
        # Look for Normal images
        self.bg_files = glob.glob(os.path.join(background_dir, '**', '*.png'), recursive=True)
        if not self.bg_files:
            self.bg_files = glob.glob(os.path.join(background_dir, '*.png'))
            # Try jpg if png not found
            if not self.bg_files:
                self.bg_files = glob.glob(os.path.join(background_dir, '**', '*.jpg'), recursive=True)
        
        print(f"Hybrid Generator: Found {len(self.bg_files)} background images in {background_dir}")
        
    def _create_random_lesion_mask(self, shape):
        mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
        
        # Random lesion parameters
        # Avoid strict edges
        center_x = np.random.randint(40, shape[1] - 40)
        center_y = np.random.randint(40, shape[0] - 40)
        axes_x = np.random.randint(10, 40)
        axes_y = np.random.randint(10, 40)
        angle = np.random.randint(0, 360)
        
        # Draw irregular ellipse
        cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, 1.0, -1)
        
        # Irregularity
        kernel_size = np.random.randint(3, 9)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if np.random.rand() > 0.5:
            mask = cv2.dilate(mask, kernel, iterations=2)
        else:
            mask = cv2.erode(mask, kernel, iterations=1)
            
        return mask

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
                    # BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # 2. Generate Lesion Mask
        mask_np = self._create_random_lesion_mask(img.shape)
        
        # 3. Inject Lesion
        img_float = img.astype(np.float32)
        
        # Blur mask for blending
        mask_blurred = cv2.GaussianBlur(mask_np, (15, 15), 0) # Smooth transition
        mask_3ch = np.stack([mask_blurred]*3, axis=2)
        
        # Decide type: Hemorrhage (Bright) or Ischemia (Dark)
        lesion_type = 'hemorrhage' if np.random.rand() > 0.5 else 'ischemia'
        
        if lesion_type == 'hemorrhage':
            # Add intensity (Hyperdense)
            # Blood is ~60-80HU, Brain ~30-40HU. Visually distinct brighter.
            intensity = np.random.uniform(50.0, 90.0)
            blended = img_float + (mask_3ch * intensity)
        else:
            # Subtract intensity (Hypodense)
            # Ischemia/Infarct is darker (low density).
            intensity = np.random.uniform(20.0, 50.0)
            blended = img_float - (mask_3ch * intensity)
            
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # 4. Transform
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform_img(Image.fromarray(blended))
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0) 
        
        return img_tensor, mask_tensor

# --- Model ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
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
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up2(x)
        
        return self.outc(x)

# --- Main Training ---
def train():
    bg_folder = os.path.join(DATA_ROOT, 'Brain_Stroke_CT_Dataset', 'Normal')
    if not os.path.exists(bg_folder):
        print(f"Error: Background folder not found at {bg_folder}")
        return

    dataset = HybridLesionDataset(bg_folder, img_size=IMG_SIZE, num_samples=600)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = UNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Starting re-training with Hybrid (Hemorrhage + Ischemia) data...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f}")
        
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Updated model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
