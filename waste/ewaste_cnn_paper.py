"""
ewaste_cnn_paper.py -- E-Waste Classification (v6 - Maximum Accuracy)
Based on: IOP Conf. Ser.: Earth Environ. Sci. 1529 (2025) 012032

Full dataset + ResNet18 + MixUp + 2-Phase + 80 total epochs
"""

import os
os.environ['TORCH_HOME'] = 'D:\\torch_cache'
import sys, json, random, copy
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# =================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_full")
if not os.path.isdir(DATASET_DIR):
    DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")

IMG_SIZE = 224          # Standard ResNet input size (was 180)
BATCH_SIZE = 16
NUM_CLASSES = 10
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

# =================================================================
# MODEL
# =================================================================

class EWasteCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def freeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = False
        print("  [Backbone FROZEN]")

    def unfreeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = True
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [Backbone UNFROZEN] Trainable: {t:,}")

    def forward(self, x):
        return self.classifier(self.features(x))

# =================================================================
# DATA
# =================================================================

def get_transforms():
    train_t = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_t, eval_t

def load_data():
    train_t, eval_t = get_transforms()
    splits = {}
    for name, t in [('train', train_t), ('valid', eval_t), ('test', eval_t)]:
        path = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(path):
            print(f"ERROR: {path} not found"); sys.exit(1)
        splits[name] = datasets.ImageFolder(path, transform=t)
        print(f"  {name:5s}: {len(splits[name]):5d} images")

    # Print per-class counts
    print("\n  Per-class (train):")
    class_counts = defaultdict(int)
    for _, label in splits['train'].samples:
        class_counts[splits['train'].classes[label]] += 1
    for cls in sorted(class_counts.keys()):
        print(f"    {cls:20s}: {class_counts[cls]:5d}")

    loaders = {
        'train': DataLoader(splits['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True),
        'valid': DataLoader(splits['valid'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True),
        'test':  DataLoader(splits['test'],  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True),
    }
    return loaders, splits['train'].classes

def compute_class_weights(train_dir, device):
    """Compute balanced class weights for imbalanced dataset."""
    class_dirs = sorted([d for d in os.listdir(train_dir)
                        if os.path.isdir(os.path.join(train_dir, d))])
    counts = []
    for cls in class_dirs:
        cls_path = os.path.join(train_dir, cls)
        count = len([f for f in os.listdir(cls_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        counts.append(count)

    counts = np.array(counts, dtype=np.float32)
    # Smooth weights: sqrt to avoid extreme weighting
    weights = np.sqrt(counts.sum() / (len(counts) * counts))

    print("\n  Class weights (smoothed):")
    for i, cls in enumerate(class_dirs):
        print(f"    {cls:20s}: {weights[i]:.4f} ({int(counts[i])} imgs)")

    return torch.tensor(weights, dtype=torch.float32).to(device)

# =================================================================
# MIXUP
# =================================================================

def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation: blends pairs of images and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =================================================================
# TRAINING
# =================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if use_mixup and random.random() < 0.5:  # 50% chance of MixUp
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)
            out = model(imgs)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            _, pred = out.max(1)
            correct += (lam * pred.eq(y_a).sum().item() + (1-lam) * pred.eq(y_b).sum().item())
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        total += labels.size(0)
    return loss_sum / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            loss_sum += loss.item() * imgs.size(0)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return loss_sum / total, correct / total

# =================================================================
# MAIN
# =================================================================

def main():
    print("=" * 60)
    print("E-WASTE CNN v6 (ResNet18 + MixUp + Full Dataset)")
    print("=" * 60)

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Dataset: {DATASET_DIR}\n")

    loaders, class_names = load_data()

    # Class weights for imbalanced data
    class_weights = compute_class_weights(os.path.join(DATASET_DIR, 'train'), device)

    model = EWasteCNN(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_val_acc, best_wts = 0.0, None

    def save_best(epoch, val_acc):
        nonlocal best_val_acc, best_wts
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_accuracy': val_acc, 'class_names': class_names},
                       os.path.join(OUTPUT_DIR, 'best_ewaste_model.pth'))
            return True
        return False

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # =============== PHASE 1: Head only (20 epochs) ===============
    P1 = 20
    print(f"\n{'='*60}")
    print(f"PHASE 1: Classifier Only ({P1} epochs)")
    print(f"{'='*60}\n")

    model.freeze_backbone()
    opt1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=P1, eta_min=1e-5)

    for ep in range(1, P1 + 1):
        tl, ta = train_one_epoch(model, loaders['train'], criterion, opt1, device, use_mixup=False)
        vl, va = evaluate(model, loaders['valid'], criterion, device)
        sched1.step()
        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl); history['val_acc'].append(va)
        m = " -> BEST" if save_best(ep, va) else ""
        print(f"  Epoch {ep:2d}/{P1} | T: {tl:.4f}/{ta:.4f} | V: {vl:.4f}/{va:.4f} | LR: {opt1.param_groups[0]['lr']:.5f}{m}")

    print(f"\n  Phase 1 best: {best_val_acc:.4f}")

    # =============== PHASE 2: Full fine-tune (60 epochs) ===============
    P2 = 60
    print(f"\n{'='*60}")
    print(f"PHASE 2: Full Fine-Tuning + MixUp ({P2} epochs)")
    print(f"{'='*60}\n")

    if best_wts: model.load_state_dict(best_wts)
    model.unfreeze_backbone()

    opt2 = optim.Adam([
        {'params': model.features.parameters(), 'lr': 5e-6},     # Very careful backbone
        {'params': model.classifier.parameters(), 'lr': 1e-4},   # Classifier
    ], weight_decay=5e-5)
    sched2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=15, T_mult=2, eta_min=1e-7)

    patience, pctr = 18, 0
    for ep in range(1, P2 + 1):
        gep = P1 + ep
        tl, ta = train_one_epoch(model, loaders['train'], criterion, opt2, device, use_mixup=True)
        vl, va = evaluate(model, loaders['valid'], criterion, device)
        sched2.step()
        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl); history['val_acc'].append(va)

        is_best = save_best(gep, va)
        m = " -> BEST" if is_best else ""
        total = P1 + P2
        print(f"  Epoch {gep:2d}/{total} | T: {tl:.4f}/{ta:.4f} | V: {vl:.4f}/{va:.4f} | LR: {opt2.param_groups[1]['lr']:.6f}{m}")

        pctr = 0 if is_best else pctr + 1
        if pctr >= patience:
            print(f"\n  EarlyStopping at epoch {gep}."); break

    # =============== EVALUATE ===============
    model.load_state_dict(best_wts)
    print(f"\n  Best val: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")

    test_loss, test_acc = evaluate(model, loaders['test'], criterion, device)
    print(f"  Test:    {test_acc:.4f} ({test_acc*100:.1f}%)")

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

    # Confusion matrix
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for imgs, lbl in loaders['test']:
            out = model(imgs.to(device))
            preds_list.extend(out.argmax(1).cpu().numpy())
            labels_list.extend(lbl.numpy())
    y_true, y_pred = np.array(labels_list), np.array(preds_list)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(12,10))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix', ylabel='True', xlabel='Predicted')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "="*60 + "\nCLASSIFICATION REPORT\n" + "="*60)
    print(report)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'class_names.json'), 'w') as f:
        json.dump({'class_names': class_names}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE | Val: {best_val_acc*100:.1f}% | Test: {test_acc*100:.1f}%")
    print(f"Results: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
