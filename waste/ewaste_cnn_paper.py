"""
ewaste_cnn_paper.py -- E-Waste Classification (Final v9)
Based on: IOP Conf. Ser.: Earth Environ. Sci. 1529 (2025) 012032

Full dataset (5967 imgs) + WeightedRandomSampler for balanced batches
+ ResNet50 V2 + MixUp (Phase 2) + Calibrated Augmentation + TTA
"""

import os
os.environ['TORCH_HOME'] = 'D:\\torch_cache'
import sys, json, random, copy, math
import numpy as np
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

# =================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_full")
if not os.path.isdir(DATASET_DIR):
    DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")

IMG_SIZE = 224          # Standard ImageNet size — keeps compute fast on 6GB GPU
BATCH_SIZE = 32         # 16 -> 32 for stable gradients with MixUp
NUM_CLASSES = 10
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = False   # Allow non-deterministic for speed
        torch.backends.cudnn.benchmark = True         # Auto-tune conv algorithms

# =================================================================
# MODEL -- ResNet50 V2 with 2-layer classifier head
# =================================================================

class EWasteCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        # Split backbone for progressive unfreezing
        self.layer0 = nn.Sequential(*list(backbone.children())[:4])   # conv1, bn1, relu, maxpool
        self.layer1 = backbone.layer1  # 64->256
        self.layer2 = backbone.layer2  # 256->512
        self.layer3 = backbone.layer3  # 512->1024
        self.layer4 = backbone.layer4  # 1024->2048
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def freeze_backbone(self):
        """Freeze entire backbone."""
        for layer in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in layer.parameters(): p.requires_grad = False

    def unfreeze_top_layers(self):
        """Unfreeze only layer3 + layer4 (deep features). Keep early layers frozen."""
        for layer in [self.layer0, self.layer1, self.layer2]:
            for p in layer.parameters(): p.requires_grad = False
        for layer in [self.layer3, self.layer4]:
            for p in layer.parameters(): p.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all backbone layers."""
        for layer in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in layer.parameters(): p.requires_grad = True

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.classifier(x)

# =================================================================
# MIXUP (Phase 2 only)
# =================================================================

def mixup_data(x, y, alpha=0.2):
    """MixUp: blend pairs of images and labels to reduce overfitting."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss: weighted combination of losses for both label sets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =================================================================
# DATA WITH BALANCED SAMPLING + CALIBRATED AUGMENTATION
# =================================================================

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
    ])

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_tta_transforms():
    """7 augmentations for Test-Time Augmentation."""
    return [
        transforms.Compose([  # 1. Original
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([  # 2. Horizontal flip
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([  # 3. Center crop
            transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([  # 4. Top-left crop
            transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
            transforms.FiveCrop(IMG_SIZE),
            transforms.Lambda(lambda crops: crops[0]),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([  # 5. Bottom-right crop
            transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
            transforms.FiveCrop(IMG_SIZE),
            transforms.Lambda(lambda crops: crops[3]),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([  # 6. Slight scale-down
            transforms.Resize((IMG_SIZE+64, IMG_SIZE+64)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([  # 7. Horizontal flip + center crop
            transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    ]

def make_balanced_sampler(dataset):
    """
    WeightedRandomSampler: oversample minority classes so each batch 
    sees roughly equal representation. This is better than class weights
    in the loss function because it doesn't distort gradients.
    """
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    total = len(targets)
    
    # Weight for each sample = 1 / (count of its class)
    weights = [1.0 / class_counts[t] for t in targets]
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=total,     # Sample same total as dataset
        replacement=True       # Must be True for oversampling
    )
    
    print("\n  Balanced sampler created:")
    for cls_idx in sorted(class_counts.keys()):
        cls_name = dataset.classes[cls_idx]
        print(f"    {cls_name:20s}: {class_counts[cls_idx]:5d} imgs -> weight {1.0/class_counts[cls_idx]:.6f}")
    
    return sampler

def load_data():
    train_t = get_train_transform()
    eval_t = get_eval_transform()
    
    splits = {}
    for name, t in [('train', train_t), ('valid', eval_t), ('test', eval_t)]:
        path = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(path):
            print(f"ERROR: {path} not found"); sys.exit(1)
        splits[name] = datasets.ImageFolder(path, transform=t)
        print(f"  {name:5s}: {len(splits[name]):5d} images")

    # Create balanced sampler for training
    sampler = make_balanced_sampler(splits['train'])
    
    loaders = {
        'train': DataLoader(splits['train'], batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=0, pin_memory=True),
        'valid': DataLoader(splits['valid'], batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True),
        'test':  DataLoader(splits['test'],  batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True),
    }
    return loaders, splits['train'].classes

# =================================================================
# TRAINING
# =================================================================

# AMP scaler for mixed precision training (FP16 on RTX 3050)
amp_scaler = torch.amp.GradScaler('cuda')

def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.2):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        if use_mixup:
            mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(mixed_imgs)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            loss_sum += loss.item() * imgs.size(0)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += (lam * pred.eq(y_a).sum().item() + (1 - lam) * pred.eq(y_b).sum().item())
        else:
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = criterion(out, labels)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            loss_sum += loss.item() * imgs.size(0)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return loss_sum / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = criterion(out, labels)
            loss_sum += loss.item() * imgs.size(0)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return loss_sum / total, correct / total

def evaluate_tta(model, device, class_names):
    """Test-Time Augmentation: average predictions across 7 views."""
    from PIL import Image
    model.eval()
    tta_transforms = get_tta_transforms()
    test_dir = os.path.join(DATASET_DIR, 'test')
    all_preds, all_labels = [], []

    for true_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for img_file in sorted(os.listdir(class_dir)):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            img = Image.open(os.path.join(class_dir, img_file)).convert('RGB')
            avg_probs = torch.zeros(NUM_CLASSES).to(device)
            with torch.no_grad():
                for t in tta_transforms:
                    inp = t(img).unsqueeze(0).to(device)
                    avg_probs += torch.softmax(model(inp), dim=1).squeeze()
            all_preds.append((avg_probs / len(tta_transforms)).argmax().item())
            all_labels.append(true_idx)
    return np.array(all_labels), np.array(all_preds)

# =================================================================
# LEARNING RATE WARMUP SCHEDULER
# =================================================================

class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            factor = self.current_epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * factor
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))

# =================================================================
# MAIN
# =================================================================

def main():
    print("=" * 60)
    print("E-WASTE CNN v9 (ResNet50 V2 + MixUp + Calibrated Aug + TTA)")
    print("=" * 60)

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}\n")

    loaders, class_names = load_data()
    model = EWasteCNN(pretrained=True).to(device)
    # NO class weights - balanced sampling handles it
    # Light label smoothing since no MixUp
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_val_acc, best_wts = 0.0, None

    def save_best(ep, va):
        nonlocal best_val_acc, best_wts
        if va > best_val_acc:
            best_val_acc = va
            best_wts = copy.deepcopy(model.state_dict())
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'val_accuracy': va, 'class_names': class_names},
                       os.path.join(OUTPUT_DIR, 'best_ewaste_model.pth'))
            return True
        return False

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # =============== PHASE 1: Head only ===============
    P1 = 25
    print(f"\n{'='*60}")
    print(f"PHASE 1: Classifier Only ({P1} epochs)")
    print(f"{'='*60}\n")
    model.freeze_backbone()
    
    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable (head only): {trainable_p1:,}\n")

    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=1e-3, weight_decay=1e-2)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=P1, eta_min=1e-6)

    for ep in range(1, P1+1):
        tl, ta = train_one_epoch(model, loaders['train'], criterion, opt1, device,
                                 use_mixup=False)
        vl, va = evaluate(model, loaders['valid'], criterion, device)
        sched1.step()
        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl); history['val_acc'].append(va)
        m = " *BEST*" if save_best(ep, va) else ""
        print(f"  Epoch {ep:2d}/{P1} | T: {tl:.4f}/{ta:.4f} | V: {vl:.4f}/{va:.4f}{m}")

    print(f"\n  Phase 1 best: {best_val_acc:.4f}")

    # =============== PHASE 2: Fine-tune top layers (layer3 + layer4) ===============
    P2 = 40
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-Tune Top Layers ({P2} epochs, AdamW)")
    print(f"{'='*60}\n")

    if best_wts: model.load_state_dict(best_wts)
    model.unfreeze_top_layers()  # Only layer3 + layer4
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable (layer3+layer4+head): {t:,}\n")

    opt2 = optim.AdamW([
        {'params': model.layer3.parameters(), 'lr': 5e-5},
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 5e-4},
    ], weight_decay=1e-2)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=P2, eta_min=1e-7)

    patience, pctr = 15, 0
    for ep in range(1, P2+1):
        gep = P1 + ep
        tl, ta = train_one_epoch(model, loaders['train'], criterion, opt2, device,
                                 use_mixup=False)  # No MixUp — clean fine-tuning
        vl, va = evaluate(model, loaders['valid'], criterion, device)
        sched2.step()
        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl); history['val_acc'].append(va)
        is_best = save_best(gep, va)
        m = " *BEST*" if is_best else ""
        cur_lr = opt2.param_groups[1]['lr']  # layer4 LR
        print(f"  Epoch {gep:2d}/{P1+P2} | T: {tl:.4f}/{ta:.4f} | V: {vl:.4f}/{va:.4f} | lr: {cur_lr:.6f}{m}")
        pctr = 0 if is_best else pctr + 1
        if pctr >= patience:
            print(f"\n  EarlyStopping at epoch {gep}."); break

    # =============== EVALUATE ===============
    model.load_state_dict(best_wts)

    test_loss, test_acc = evaluate(model, loaders['test'], criterion, device)
    print(f"\n  Standard Test: {test_acc:.4f} ({test_acc*100:.1f}%)")

    print("  Running TTA (7 views)...")
    y_true, y_pred = evaluate_tta(model, device, class_names)
    tta_acc = np.mean(y_true == y_pred)
    print(f"  TTA Test:     {tta_acc:.4f} ({tta_acc*100:.1f}%)")

    # ---- Plots ----
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    er = range(1, len(history['train_acc'])+1)
    fig, (a1,a2) = plt.subplots(1,2,figsize=(14,5))
    a1.plot(er, history['train_acc'], label='Train', lw=2, color='#2196F3')
    a1.plot(er, history['val_acc'], label='Val', lw=2, color='#FF9800')
    a1.axvline(x=P1, color='red', ls='--', alpha=0.5, label='Phase 1→2')
    a1.set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=[0,1.05])
    a1.legend(); a1.grid(True, alpha=0.3)
    a2.plot(er, history['train_loss'], label='Train', lw=2, color='#2196F3')
    a2.plot(er, history['val_loss'], label='Val', lw=2, color='#FF9800')
    a2.axvline(x=P1, color='red', ls='--', alpha=0.5, label='Phase 1→2')
    a2.set(title='Loss', xlabel='Epoch', ylabel='Loss')
    a2.legend(); a2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12,10))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix (TTA)', ylabel='True', xlabel='Predicted')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "="*60 + "\nCLASSIFICATION REPORT (TTA)\n" + "="*60)
    print(report)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'class_names.json'), 'w') as f:
        json.dump({'class_names': class_names}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL | Val: {best_val_acc*100:.1f}% | Test: {test_acc*100:.1f}% | TTA: {tta_acc*100:.1f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
