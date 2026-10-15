# Auto-generated from CAMTS_UNet_Full.ipynb code cells
# Do not edit manually unless needed for local execution

# ===== CELL 1 =====
# !pip install torch torchvision opencv-python-headless albumentations matplotlib scikit-learn tqdm scipy segmentation-models-pytorch -q

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ===== CELL 2 =====
import os, cv2, time, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

class Config:
    DATA_DIR    = '/content/lgg-mri-segmentation/kaggle_3m'
    IMAGE_SIZE  = 256
    BATCH_SIZE  = 8
    EPOCHS      = 50
    LR          = 1e-4
    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED        = 42
    VAL_SPLIT   = 0.15
    TEST_SPLIT  = 0.15
    CHECKPOINT  = 'camts_unet_best.pth'
    MEAN        = (0.485, 0.456, 0.406)
    STD         = (0.229, 0.224, 0.225)

cfg = Config()
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
print(f'Device: {cfg.DEVICE}')

# ===== CELL 3 =====
fig = plt.figure(figsize=(20, 5))
fig.suptitle('Brain MRI Data Overview — LGG Segmentation Dataset', fontsize=14, fontweight='bold', y=1.02)

# 1. Patient split pie
ax1 = fig.add_subplot(141)
ax1.pie([59, 51], labels=['Tumor\n(59)', 'No Tumor\n(51)'],
        colors=['#E74C3C','#2ECC71'], autopct='%1.1f%%',
        startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
ax1.set_title('Total Subjects\n(110 Patients)', fontweight='bold')

# 2. Imaging modalities
ax2 = fig.add_subplot(142)
modalities = ['FLAIR', 'T1', 'T1ce', 'T2']
counts     = [3929, 3929, 3929, 3929]
colors_mod = ['#3498DB','#9B59B6','#E67E22','#1ABC9C']
bars = ax2.bar(modalities, counts, color=colors_mod, edgecolor='white', linewidth=1.5)
ax2.set_title('Imaging Modalities', fontweight='bold')
ax2.set_ylabel('Slices')
for bar, c in zip(bars, counts):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, str(c),
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Pathological breakdown
ax3 = fig.add_subplot(143)
categories = ['No Tumor', 'Benign', 'Malignant']
cat_counts = [1571, 1180, 1178]
cat_colors = ['#95A5A6','#F39C12','#E74C3C']
bars2 = ax3.bar(categories, cat_counts, color=cat_colors, edgecolor='white', linewidth=1.5)
ax3.set_title('Pathological Images', fontweight='bold')
ax3.set_ylabel('Image Count')
for bar, c in zip(bars2, cat_counts):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, str(c),
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Dataset split
ax4 = fig.add_subplot(144)
splits = [70, 15, 15]
split_labels = ['Train\n70%', 'Val\n15%', 'Test\n15%']
split_colors = ['#3498DB','#E67E22','#2ECC71']
left = 0
for s, l, c in zip(splits, split_labels, split_colors):
    ax4.barh(0, s, left=left, color=c, edgecolor='white', linewidth=2, height=0.5)
    ax4.text(left + s/2, 0, l, ha='center', va='center',
             fontsize=10, fontweight='bold', color='white')
    left += s
ax4.set_xlim(0, 100)
ax4.set_ylim(-0.5, 0.5)
ax4.axis('off')
ax4.set_title('Dataset Split', fontweight='bold')

plt.tight_layout()
plt.savefig('dataset_description.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: dataset_description.png')

# ===== CELL 4 =====
def collect_pairs(data_dir):
    images, masks = [], []
    for patient_dir in sorted(Path(data_dir).iterdir()):
        if not patient_dir.is_dir(): continue
        for img_file in sorted(patient_dir.glob('*.tif')):
            if '_mask' not in img_file.name:
                mask_file = patient_dir / img_file.name.replace('.tif', '_mask.tif')
                if mask_file.exists():
                    images.append(str(img_file))
                    masks.append(str(mask_file))
    print(f'Total pairs: {len(images)}')
    return images, masks

all_images, all_masks = collect_pairs(cfg.DATA_DIR)

pos_imgs, pos_masks, neg_imgs, neg_masks = [], [], [], []
for img, mask in zip(all_images, all_masks):
    m = np.array(Image.open(mask).convert('L'))
    if m.max() > 0: pos_imgs.append(img); pos_masks.append(mask)
    else:           neg_imgs.append(img); neg_masks.append(mask)

n_neg = min(len(neg_imgs), len(pos_imgs))
idx   = np.random.choice(len(neg_imgs), n_neg, replace=False)
final_imgs  = pos_imgs + [neg_imgs[i] for i in idx]
final_masks = pos_masks + [neg_masks[i] for i in idx]
print(f'Positive: {len(pos_imgs)} | Negative (balanced): {n_neg} | Total: {len(final_imgs)}')

X_tr, X_tmp, y_tr, y_tmp = train_test_split(final_imgs, final_masks,
    test_size=cfg.VAL_SPLIT+cfg.TEST_SPLIT, random_state=cfg.SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp,
    test_size=0.5, random_state=cfg.SEED)
print(f'Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}')

# ===== CELL 5 =====
train_transform = A.Compose([
    A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=15, p=0.4),
    A.ElasticTransform(alpha=120, sigma=6, p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
    A.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ToTensorV2()
])

class BrainMRIDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images; self.masks = masks; self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img  = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = (np.array(Image.open(self.masks[idx]).convert('L')) > 0).astype(np.float32)
        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img  = aug['image']; mask = aug['mask'].unsqueeze(0)
        return img, mask

train_ds = BrainMRIDataset(X_tr,  y_tr,  train_transform)
val_ds   = BrainMRIDataset(X_val, y_val, val_transform)
test_ds  = BrainMRIDataset(X_te,  y_te,  val_transform)
train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
print(f'Batches — Train:{len(train_loader)} Val:{len(val_loader)} Test:{len(test_loader)}')

# ===== CELL 6 =====
fig, ax = plt.subplots(figsize=(20, 6))
ax.set_xlim(0, 20); ax.set_ylim(0, 6); ax.axis('off')
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('#F8F9FA')

def draw_box(ax, x, y, w, h, text, color, fontsize=9):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle='round,pad=0.1', facecolor=color,
        edgecolor='#555555', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', wrap=True,
            multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color='#333333',
                        lw=2, connectionstyle='arc3,rad=0'))

# Row 1 — Data stages
stage_color   = '#AED6F1'
proc_color    = '#A9DFBF'
model_color   = '#FAD7A0'
eval_color    = '#D7BDE2'

draw_box(ax, 2,   4.5, 2.8, 0.9, 'Data\nCollection',      stage_color)
draw_box(ax, 5.5, 4.5, 2.8, 0.9, 'Data\nPreprocessing',   proc_color)
draw_box(ax, 9,   4.5, 2.8, 0.9, 'Dataset\nSplit',        stage_color)
draw_arrow(ax, 3.4, 4.5, 4.1, 4.5)
draw_arrow(ax, 6.9, 4.5, 7.6, 4.5)

# Dataset split sub-boxes
draw_box(ax, 7.8, 3.2, 1.5, 0.6, 'Train 70%',  '#3498DB', fontsize=8)
draw_box(ax, 9.3, 3.2, 1.5, 0.6, 'Val 15%',    '#E67E22', fontsize=8)
draw_box(ax,10.8, 3.2, 1.5, 0.6, 'Test 15%',   '#2ECC71', fontsize=8)
draw_arrow(ax, 9, 4.05, 9, 3.55)

# Row 2 — Preprocessing steps
draw_box(ax, 2,   2.2, 2.8, 0.9, '3D to 2D\nSlice',       proc_color)
draw_box(ax, 5.5, 2.2, 2.8, 0.9, 'Image\nResizing 256×256', proc_color)
draw_box(ax, 9,   2.2, 2.8, 0.9, 'Normalization\n& Augmentation', proc_color)
draw_arrow(ax, 2, 4.05, 2, 2.65)
draw_arrow(ax, 3.4, 2.2, 4.1, 2.2)
draw_arrow(ax, 6.9, 2.2, 7.6, 2.2)

# Row 3 — Model & Evaluation
draw_box(ax, 12.5, 2.2, 3.0, 0.9, 'Proposed\nCAMTS-UNet',   model_color)
draw_box(ax, 16,   2.2, 2.8, 0.9, 'Model\nEvaluation',      eval_color)
draw_box(ax, 19,   2.2, 1.8, 0.9, 'Result\nAnalysis',       eval_color)
draw_arrow(ax, 10.4, 2.2, 11.0, 2.2)
draw_arrow(ax, 14.0, 2.2, 14.6, 2.2)
draw_arrow(ax, 17.4, 2.2, 18.1, 2.2)

ax.set_title('CAMTS-UNet: Methodology Pipeline', fontsize=14,
             fontweight='bold', pad=10)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=stage_color,  label='Data Stages'),
    mpatches.Patch(facecolor=proc_color,   label='Processing'),
    mpatches.Patch(facecolor=model_color,  label='Model'),
    mpatches.Patch(facecolor=eval_color,   label='Evaluation'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('methodology_pipeline.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: methodology_pipeline.png')

# ===== CELL 7 =====
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBNReLU(ch, ch)
        self.conv2 = ConvBNReLU(ch, ch)
    def forward(self, x): return x + self.conv2(self.conv1(x))

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.res  = ResBlock(out_ch)
    def forward(self, x): return self.res(self.down(x))

class CrossAttentionGate(nn.Module):
    """
    Cross-Attention Gate: generates spatial attention map from
    gating signal (decoder) and skip connection (encoder).
    Suppresses irrelevant encoder features before concatenation.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        return x * self.psi(self.relu(g1 + x1))

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — Multi-Scale Feature Fusion"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1  = ConvBNReLU(in_ch, out_ch, kernel=1, padding=0)
        self.conv6  = ConvBNReLU(in_ch, out_ch, dilation=6,  padding=6)
        self.conv12 = ConvBNReLU(in_ch, out_ch, dilation=12, padding=12)
        self.conv18 = ConvBNReLU(in_ch, out_ch, dilation=18, padding=18)
        self.pool   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.ReLU(inplace=True))
        self.project = ConvBNReLU(out_ch*5, out_ch, kernel=1, padding=0)
    def forward(self, x):
        size = x.shape[2:]
        p    = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=True)
        return self.project(torch.cat([self.conv1(x),self.conv6(x),self.conv12(x),self.conv18(x),p], dim=1))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(ConvBNReLU(out_ch+skip_ch, out_ch), ConvBNReLU(out_ch, out_ch))
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))

class CAMTSUNet(nn.Module):
    """
    CAMTS-UNet: Cross-Attention Multi-Scale UNet
    Encoder  : 4-scale ResBlock encoder [64,128,256,512]
    Bottleneck: ASPP multi-scale context
    Decoder  : 4 stages with Cross-Attention gated skip connections
    Output   : Binary segmentation mask
    """
    def __init__(self, in_ch=3, out_ch=1, features=[64,128,256,512]):
        super().__init__()
        self.stem       = nn.Sequential(ConvBNReLU(in_ch, features[0]), ResBlock(features[0]))
        self.enc1       = EncoderBlock(features[0], features[1])
        self.enc2       = EncoderBlock(features[1], features[2])
        self.enc3       = EncoderBlock(features[2], features[3])
        self.bottleneck = ASPP(features[3], features[3])
        self.ag3        = CrossAttentionGate(features[3], features[2], features[2]//2)
        self.ag2        = CrossAttentionGate(features[2], features[1], features[1]//2)
        self.ag1        = CrossAttentionGate(features[1], features[0], features[0]//2)
        self.dec3       = DecoderBlock(features[3], features[2], features[2])
        self.dec2       = DecoderBlock(features[2], features[1], features[1])
        self.dec1       = DecoderBlock(features[1], features[0], features[0])
        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)
        self.ds3        = nn.Conv2d(features[2], out_ch, 1)
        self.ds2        = nn.Conv2d(features[1], out_ch, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b  = self.bottleneck(s3)
        d3 = self.dec3(b,  self.ag3(b,  s2))
        d2 = self.dec2(d3, self.ag2(d3, s1))
        d1 = self.dec1(d2, self.ag1(d2, s0))
        out = self.final_conv(d1)
        if self.training:
            ds3 = F.interpolate(self.ds3(d3), size=x.shape[2:], mode='bilinear', align_corners=True)
            ds2 = F.interpolate(self.ds2(d2), size=x.shape[2:], mode='bilinear', align_corners=True)
            return out, ds2, ds3
        return out

model = CAMTSUNet().to(cfg.DEVICE)
with torch.no_grad():
    model.eval()
    out = model(torch.randn(2,3,256,256).to(cfg.DEVICE))
print(f'Output shape: {out.shape}')
print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

# ===== CELL 8 =====
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__(); self.smooth = smooth
    def forward(self, pred, target):
        pred   = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum()
        return 1 - (2.*inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, ds_weight=0.3):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()
        self.bce_w = bce_weight
        self.ds_w  = ds_weight
    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            main, ds2, ds3 = outputs
            main_loss = self.dice(main,target) + self.bce_w*self.bce(main,target)
            ds2_loss  = self.dice(ds2, target) + self.bce_w*self.bce(ds2, target)
            ds3_loss  = self.dice(ds3, target) + self.bce_w*self.bce(ds3, target)
            return main_loss + self.ds_w*(ds2_loss + ds3_loss)
        return self.dice(outputs,target) + self.bce_w*self.bce(outputs,target)

criterion = CombinedLoss()
print('Loss: Dice + BCE + Deep Supervision ready')

# ===== CELL 9 =====
def compute_metrics(pred_logits, target, threshold=0.5, smooth=1e-6):
    pred   = (torch.sigmoid(pred_logits) > threshold).float()
    pred_f = pred.view(-1);   tgt_f = target.view(-1)
    TP = (pred_f * tgt_f).sum().item()
    FP = (pred_f * (1-tgt_f)).sum().item()
    FN = ((1-pred_f) * tgt_f).sum().item()
    precision = (TP + smooth) / (TP + FP + smooth)
    recall    = (TP + smooth) / (TP + FN + smooth)
    dice      = (2*TP + smooth) / (2*TP + FP + FN + smooth)
    iou       = (TP + smooth) / (TP + FP + FN + smooth)
    f1        = dice
    return {'dice': dice, 'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1}

def hausdorff_distance(pred_logits, target, threshold=0.5):
    pred_np = (torch.sigmoid(pred_logits) > threshold).cpu().numpy().squeeze()
    tgt_np  = target.cpu().numpy().squeeze()
    hds = []
    for p, t in zip(pred_np, tgt_np):
        if p.max()==0 and t.max()==0: hds.append(0.0); continue
        if p.max()==0 or  t.max()==0: hds.append(np.inf); continue
        d1  = distance_transform_edt(~p.astype(bool))
        d2  = distance_transform_edt(~t.astype(bool))
        hds.append(max(d1[t.astype(bool)].max(), d2[p.astype(bool)].max()))
    vals = [h for h in hds if h != np.inf]
    return np.nanmean(vals) if vals else 0.0

print('Metrics ready: Dice, IoU, Precision, Recall, F1, Hausdorff Distance')

# ===== CELL 10 =====
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
history   = {'train_loss':[], 'val_loss':[], 'val_dice':[], 'val_iou':[]}
best_dice = 0.0

def train_epoch(model, loader, optimizer, criterion):
    model.train(); total_loss = 0
    for imgs, masks in tqdm(loader, desc='Train', leave=False):
        imgs, masks = imgs.to(cfg.DEVICE), masks.to(cfg.DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def val_epoch(model, loader, criterion):
    model.eval(); total_loss = 0
    all_metrics = {'dice':0,'iou':0,'precision':0,'recall':0,'f1':0}
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Val', leave=False):
            imgs, masks = imgs.to(cfg.DEVICE), masks.to(cfg.DEVICE)
            preds = model(imgs)
            total_loss += criterion(preds, masks).item()
            m = compute_metrics(preds, masks)
            for k in all_metrics: all_metrics[k] += m[k]
    n = len(loader)
    return total_loss/n, {k: v/n for k,v in all_metrics.items()}

print('Starting training...')
for epoch in range(1, cfg.EPOCHS+1):
    t0          = time.time()
    tr_loss     = train_epoch(model, train_loader, optimizer, criterion)
    vl_loss, vm = val_epoch(model, val_loader, criterion)
    scheduler.step()
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['val_dice'].append(vm['dice'])
    history['val_iou'].append(vm['iou'])
    if vm['dice'] > best_dice:
        best_dice = vm['dice']
        torch.save(model.state_dict(), cfg.CHECKPOINT)
        print(f'  ✅ Saved best model (Dice:{best_dice:.4f})')
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch[{epoch:3d}/{cfg.EPOCHS}] '
              f'TrLoss:{tr_loss:.4f} VlLoss:{vl_loss:.4f} '
              f'Dice:{vm["dice"]:.4f} IoU:{vm["iou"]:.4f} '
              f'Prec:{vm["precision"]:.4f} Rec:{vm["recall"]:.4f} '
              f'({time.time()-t0:.1f}s)')
print(f'Training complete. Best Dice: {best_dice:.4f}')

# ===== CELL 11 =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(history['train_loss'], label='Train', color='tomato')
axes[0].plot(history['val_loss'],   label='Val',   color='steelblue')
axes[0].set_title('Loss Curve'); axes[0].legend(); axes[0].set_xlabel('Epoch')
axes[1].plot(history['val_dice'], color='green')
axes[1].set_title('Validation Dice Score'); axes[1].set_xlabel('Epoch')
axes[2].plot(history['val_iou'], color='purple')
axes[2].set_title('Validation IoU Score'); axes[2].set_xlabel('Epoch')
plt.suptitle('CAMTS-UNet Training Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: training_curves.png')

# ===== CELL 12 =====
model.load_state_dict(torch.load(cfg.CHECKPOINT, map_location=cfg.DEVICE))
model.eval()

test_metrics = {'dice':0,'iou':0,'precision':0,'recall':0,'f1':0}
test_hd = 0
with torch.no_grad():
    for imgs, masks in tqdm(test_loader, desc='Testing'):
        imgs, masks = imgs.to(cfg.DEVICE), masks.to(cfg.DEVICE)
        preds = model(imgs)
        m = compute_metrics(preds, masks)
        for k in test_metrics: test_metrics[k] += m[k]
        test_hd += hausdorff_distance(preds, masks)

n = len(test_loader)
final_metrics = {k: v/n*100 for k,v in test_metrics.items()}
final_hd      = test_hd / n

print('\n' + '='*55)
print('          CAMTS-UNet — TEST RESULTS')
print('='*55)
for k,v in final_metrics.items():
    print(f'  {k.capitalize():<20}: {v:.2f}%')
print(f'  {"Hausdorff Dist (MHD)":<20}: {final_hd:.4f}')
print('='*55)

# ===== CELL 13 =====
def get_color(value):
    if value >= 90: return '#27AE60'
    elif value >= 80: return '#F39C12'
    else: return '#E74C3C'

metric_names = ['Dice Score', 'IoU Score', 'Precision', 'Recall', 'F1 Score']
metric_keys  = ['dice', 'iou', 'precision', 'recall', 'f1']
metric_values = [final_metrics[k] for k in metric_keys]

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle('CAMTS-UNet: Quantitative Analysis', fontsize=15, fontweight='bold', y=1.02)

for ax, name, value in zip(axes, metric_names, metric_values):
    color = get_color(value)
    ax.pie([value, 100-value],
           colors=[color, '#EAECEE'],
           startangle=90,
           wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
           counterclock=False)
    ax.text(0, 0.08, f'{value:.2f}%', ha='center', va='center',
            fontsize=13, fontweight='bold', color=color)
    ax.text(0, -0.25, name, ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2C3E50')
    circle = plt.Circle((0,0), 0.55, color='white')
    ax.add_patch(circle)

plt.tight_layout()
plt.savefig('quantitative_donut_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: quantitative_donut_charts.png')

# ===== CELL 14 =====
def denormalize(tensor, mean=cfg.MEAN, std=cfg.STD):
    img = tensor.permute(1,2,0).cpu().numpy()
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)

def make_overlay(img_np, pred_mask, gt_mask):
    """Color overlay: TP=Green, FP=Red, FN=Blue"""
    overlay = img_np.copy()
    pred_b  = pred_mask.astype(bool)
    gt_b    = gt_mask.astype(bool)
    TP = pred_b & gt_b
    FP = pred_b & ~gt_b
    FN = ~pred_b & gt_b
    alpha = 0.45
    overlay[TP] = overlay[TP]*(1-alpha) + np.array([0,1,0])*alpha  # Green
    overlay[FP] = overlay[FP]*(1-alpha) + np.array([1,0,0])*alpha  # Red
    overlay[FN] = overlay[FN]*(1-alpha) + np.array([0,0,1])*alpha  # Blue
    return np.clip(overlay, 0, 1)

# Pick 2 samples: largest tumor & smallest tumor
tumor_sizes = [(i, np.array(Image.open(y_te[i]).convert('L')).sum()) for i in range(len(y_te))]
tumor_positive = [(i,s) for i,s in tumor_sizes if s > 0]
tumor_positive.sort(key=lambda x: x[1])
idx_small = tumor_positive[0][0]
idx_large = tumor_positive[-1][0]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
col_titles = ['Input MRI', 'Ground Truth', 'Predicted Mask', 'Overlay (TP=G, FP=R, FN=B)']
row_labels  = ['Large Tumor', 'Small Tumor']

for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=11, fontweight='bold')

model.eval()
for row, idx in enumerate([idx_large, idx_small]):
    img_t, mask_t = test_ds[idx]
    img_t_batch = img_t.unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        pred_logit = model(img_t_batch)
    pred_mask = (torch.sigmoid(pred_logit).squeeze().cpu().numpy() > 0.5).astype(np.float32)
    gt_mask   = mask_t.squeeze().numpy()
    img_np    = denormalize(img_t)
    overlay   = make_overlay(img_np, pred_mask, gt_mask)

    axes[row][0].imshow(img_np)
    axes[row][1].imshow(gt_mask,   cmap='gray')
    axes[row][2].imshow(pred_mask, cmap='gray')
    axes[row][3].imshow(overlay)
    axes[row][0].set_ylabel(row_labels[row], fontsize=11, fontweight='bold')
    for ax in axes[row]: ax.axis('off')

plt.suptitle('CAMTS-UNet: Qualitative Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('qualitative_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: qualitative_analysis.png')

# ===== CELL 15 =====
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = [
            target_layer.register_forward_hook(lambda m,i,o: setattr(self,'activations',o.detach())),
            target_layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'gradients',go[0].detach()))
        ]
    def generate(self, x):
        self.model.eval()
        x   = x.requires_grad_(True)
        out = self.model(x)
        self.model.zero_grad()
        out.mean().backward()
        w   = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=True)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    def remove(self):
        for h in self._hooks: h.remove()

gradcam = GradCAM(model, model.bottleneck.project)
sample_imgs, sample_masks = next(iter(test_loader))
sample_imgs  = sample_imgs[:4].to(cfg.DEVICE)
sample_masks = sample_masks[:4]

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
col_titles = ['MRI Image', 'Ground Truth', 'Predicted Mask', 'GradCAM (Explainability)']
for col, t in enumerate(col_titles):
    axes[0][col].set_title(t, fontsize=11, fontweight='bold')

for i in range(4):
    img_np   = denormalize(sample_imgs[i])
    gt_np    = sample_masks[i].squeeze().numpy()
    with torch.no_grad():
        pred = (torch.sigmoid(model(sample_imgs[i:i+1])).squeeze().cpu().numpy() > 0.5).astype(np.float32)
    cam = gradcam.generate(sample_imgs[i:i+1])
    axes[i][0].imshow(img_np);           axes[i][0].axis('off')
    axes[i][1].imshow(gt_np, cmap='gray'); axes[i][1].axis('off')
    axes[i][2].imshow(pred,  cmap='gray'); axes[i][2].axis('off')
    axes[i][3].imshow(img_np)
    axes[i][3].imshow(cam, alpha=0.55, cmap='jet')
    axes[i][3].axis('off')

gradcam.remove()
plt.suptitle('CAMTS-UNet: GradCAM Explainability', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('predictions_gradcam.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: predictions_gradcam.png')

# ===== CELL 16 =====
baselines = {
    'UNet-VGG16'      : smp.Unet('vgg16',          encoder_weights='imagenet', in_channels=3, classes=1),
    'UNet-ResNet34'   : smp.Unet('resnet34',        encoder_weights='imagenet', in_channels=3, classes=1),
    'UNet-ResNet50'   : smp.Unet('resnet50',        encoder_weights='imagenet', in_channels=3, classes=1),
    'UNet-EffNetB0'   : smp.Unet('efficientnet-b0', encoder_weights='imagenet', in_channels=3, classes=1),
    'UNet-MobileNetV2': smp.Unet('mobilenet_v2',    encoder_weights='imagenet', in_channels=3, classes=1),
}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6

def evaluate_model(model, loader, device):
    model.eval().to(device)
    metrics = {'dice':0,'iou':0,'precision':0,'recall':0,'f1':0}
    hd_total = 0
    t0 = time.time()
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Eval', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            m = compute_metrics(preds, masks)
            for k in metrics: metrics[k] += m[k]
            hd_total += hausdorff_distance(preds, masks)
    n = len(loader)
    return {k: v/n*100 for k,v in metrics.items()}, hd_total/n, time.time()-t0

# Train baselines (quick — 10 epochs each)
BASELINE_EPOCHS = 10
comparison_results = {}

for name, bmodel in baselines.items():
    print(f'\nTraining {name}...')
    bmodel = bmodel.to(cfg.DEVICE)
    opt    = torch.optim.AdamW(bmodel.parameters(), lr=cfg.LR, weight_decay=1e-5)
    crit   = CombinedLoss()
    epoch_time = 0
    for ep in range(BASELINE_EPOCHS):
        t0 = time.time()
        bmodel.train()
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(cfg.DEVICE), masks.to(cfg.DEVICE)
            opt.zero_grad()
            loss = crit(bmodel(imgs), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bmodel.parameters(), 1.0)
            opt.step()
        epoch_time += time.time()-t0
    m, hd, _ = evaluate_model(bmodel, test_loader, cfg.DEVICE)
    comparison_results[name] = {**m, 'mhd': hd, 'params': count_params(bmodel),
                                  'epoch_time': epoch_time/BASELINE_EPOCHS}
    print(f'{name} → Dice:{m["dice"]:.2f}% MHD:{hd:.4f} Params:{count_params(bmodel):.1f}M')

# Add CAMTS-UNet results
camts_m, camts_hd, _ = evaluate_model(model, test_loader, cfg.DEVICE)
comparison_results['CAMTS-UNet\n(Ours)'] = {
    **camts_m, 'mhd': camts_hd,
    'params': count_params(model),
    'epoch_time': 0
}

# ── MHD Bar Chart ──────────────────────────────────────────
model_names = list(comparison_results.keys())
mhd_values  = [comparison_results[m]['mhd'] for m in model_names]
bar_colors  = ['#5DADE2']*5 + ['#1A5276']  # highlight CAMTS-UNet

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(model_names, mhd_values, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)
ax.axhline(y=mhd_values[-1], color='#E74C3C', linestyle='--', linewidth=1.5,
           label=f'CAMTS-UNet MHD = {mhd_values[-1]:.3f}')
for bar, val in zip(bars, mhd_values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
bars[-1].set_edgecolor('#E74C3C'); bars[-1].set_linewidth(3)
ax.text(bars[-1].get_x()+bars[-1].get_width()/2,
        bars[-1].get_height()+0.012, '★ Best', ha='center',
        fontsize=11, color='#E74C3C', fontweight='bold')
ax.set_title('MHD Comparison (Lower is Better)', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Hausdorff Distance (MHD)')
ax.legend(fontsize=10)
ax.set_facecolor('#FDFEFE')
plt.tight_layout()
plt.savefig('mhd_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: mhd_comparison.png')

# ===== CELL 17 =====
fig, ax = plt.subplots(figsize=(16, 4))
ax.axis('off')

col_labels = ['Model', 'Dice (%)', 'IoU (%)', 'Precision (%)', 'Recall (%)', 'F1 (%)', 'MHD ↓', 'Params (M)']
table_data = []
for name, res in comparison_results.items():
    table_data.append([
        name.replace('\n', ' '),
        f'{res["dice"]:.2f}',
        f'{res["iou"]:.2f}',
        f'{res["precision"]:.2f}',
        f'{res["recall"]:.2f}',
        f'{res["f1"]:.2f}',
        f'{res["mhd"]:.4f}',
        f'{res["params"]:.1f}'
    ])

table = ax.table(cellText=table_data, colLabels=col_labels,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#1A5276')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Highlight CAMTS-UNet row
camts_row = len(table_data)
for j in range(len(col_labels)):
    table[camts_row, j].set_facecolor('#D5F5E3')
    table[camts_row, j].set_text_props(fontweight='bold')

# Alternate row colors
for i in range(1, len(table_data)):
    color = '#EBF5FB' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        if i != camts_row:
            table[i, j].set_facecolor(color)

ax.set_title('Pretrained Model Comparison — All Metrics', fontsize=13,
             fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('model_comparison_table.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: model_comparison_table.png')

# ===== CELL 18 =====
def predict_single(image_path, model, transform, device):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    aug = transform(image=img, mask=np.zeros(img.shape[:2], dtype=np.float32))
    inp = aug['image'].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out      = model(inp)
        prob_map = torch.sigmoid(out).squeeze().cpu().numpy()
        pred_mask = (prob_map > 0.5).astype(np.uint8)
    return pred_mask, prob_map, img

sample_path = X_te[0]
pred_mask, prob_map, orig_img = predict_single(sample_path, model, val_transform, cfg.DEVICE)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(orig_img);               axes[0].set_title('Input MRI',       fontweight='bold'); axes[0].axis('off')
axes[1].imshow(prob_map, cmap='hot');   axes[1].set_title('Probability Map',  fontweight='bold'); axes[1].axis('off')
axes[2].imshow(pred_mask, cmap='gray'); axes[2].set_title('Segmented Tumor',  fontweight='bold'); axes[2].axis('off')
plt.suptitle('CAMTS-UNet: Single Image Inference', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('single_inference.png', dpi=150, bbox_inches='tight')
plt.show()

has_tumor = pred_mask.max() > 0
print(f'Tumor detected : {"YES ✅" if has_tumor else "NO ❌"}')
print(f'Tumor area     : {pred_mask.sum()} pixels ({pred_mask.mean()*100:.2f}% of image)')

