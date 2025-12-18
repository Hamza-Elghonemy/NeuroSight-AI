import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
from tqdm import tqdm

# --- Config ---
IMG_SIZE = 224
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_unet.pth')

print(f"Using device: {DEVICE}")

# --- Model Architecture (Must match training) ---
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

# --- Dataset for Real External Test Data ---
class RealTestDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.img_size = img_size
        self.root_dir = root_dir
        # Assuming folder structure: External_Test/PNG/*.png and External_Test/MASKS/*.png
        # We need to match filenames
        
        self.mask_dir = os.path.join(root_dir, 'MASKS')
        self.img_dir = os.path.join(root_dir, 'PNG')
        
        self.mask_files = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
        self.valid_files = []
        
        # Filter to ensure matching image exists
        for m_path in self.mask_files:
            fname = os.path.basename(m_path)
            img_path = os.path.join(self.img_dir, fname)
            if os.path.exists(img_path):
                self.valid_files.append((img_path, m_path))
                
        print(f"Found {len(self.valid_files)} paired images and masks in {root_dir}")

    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_files[idx]
        
        # Load Image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask (lesion values might be 200, 255 etc)
        mask = (mask > 0).astype(np.float32)
        
        # Transform
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform_img(Image.fromarray(img))
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return img_tensor, mask_tensor

# --- Metrics ---
def dice_coefficient(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)

def iou_score(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# --- Evaluation ---
def evaluate():
    test_dir = os.path.join(DATA_ROOT, 'Brain_Stroke_CT_Dataset', 'External_Test')
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    dataset = RealTestDataset(test_dir, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load Model
    model = UNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("Error: No trained model found.")
        return
        
    model.eval()
    
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    print("Starting evaluation on real External Test data...")
    
    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, unit="batch"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            # Predict
            output = model(imgs)
            preds = torch.sigmoid(output)
            preds = (preds > 0.5).float()
            
            # Calculate metrics
            batch_dice = 0.0
            batch_iou = 0.0
            for i in range(imgs.size(0)):
                batch_dice += dice_coefficient(preds[i], masks[i]).item()
                batch_iou += iou_score(preds[i], masks[i]).item()
                
            total_dice += batch_dice / imgs.size(0)
            total_iou += batch_iou / imgs.size(0)
            num_batches += 1
            
    print("-" * 30)
    print(f"Average Dice Score: {total_dice/num_batches:.4f}")
    print(f"Average IoU Score:  {total_iou/num_batches:.4f}")
    print("-" * 30)
    print("Note: These scores reflect performance on REAL labeled lesions.")
    print("If scores are lower than training accuracy, it indicates the 'domain gap'")
    print("between our synthetic training data and real clinical ground truth.")

if __name__ == "__main__":
    evaluate()
