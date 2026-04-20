"""
contribution_gradcam.py -- Multi-Architecture Comparison & Explainability Study
Contribution to: IOP Conf. Ser.: Earth Environ. Sci. 1529 (2025) 012032

CONTRIBUTION:
The base paper tested only a single CNN architecture. This script provides
a systematic multi-architecture comparison study, evaluating 4 backbone
networks (ResNet18, ResNet50, MobileNetV3, EfficientNet-B0) on the same
e-waste dataset. For each model, we analyze:
  1. Classification accuracy (test set)
  2. Model efficiency (parameters, inference speed)
  3. Grad-CAM attention patterns (where each model looks)
  4. t-SNE feature space visualization (how well classes separate)

This enables evidence-based architecture selection for real-world e-waste
sorting systems with different hardware constraints.
"""

import os
os.environ['TORCH_HOME'] = 'D:\\torch_cache'
import sys, json, time, copy, random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image

# =================================================================
# CONFIGURATION
# =================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_full")
if not os.path.isdir(DATASET_DIR):
    DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "contribution")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training config for quick comparison (head-only training)
HEAD_EPOCHS = 15   # Fast head-only training for fair comparison
FINETUNE_EPOCHS = 20  # Short fine-tune phase

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# =================================================================
# MULTI-ARCHITECTURE MODEL DEFINITIONS
# =================================================================

class MultiArchModel(nn.Module):
    """Unified wrapper for multiple backbone architectures."""
    
    def __init__(self, arch_name, num_classes=NUM_CLASSES):
        super().__init__()
        self.arch_name = arch_name
        
        if arch_name == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            # For Grad-CAM: last conv block
            self.gradcam_layer = list(backbone.layer4.children())[-1]
            
        elif arch_name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = 2048
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.gradcam_layer = list(backbone.layer4.children())[-1]
            
        elif arch_name == 'mobilenetv3':
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            feat_dim = 576
            self.features = nn.Sequential(backbone.features, backbone.avgpool)
            self.gradcam_layer = list(backbone.features.children())[-1]
            
        elif arch_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feat_dim = 1280
            self.features = nn.Sequential(backbone.features, backbone.avgpool)
            self.gradcam_layer = list(backbone.features.children())[-1]
        else:
            raise ValueError(f"Unknown architecture: {arch_name}")
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.feat_dim = feat_dim
    
    def freeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = False
    
    def unfreeze_backbone(self):
        for p in self.features.parameters(): p.requires_grad = True
    
    def forward(self, x):
        feats = self.features(x)
        return self.classifier(feats)
    
    def extract_features(self, x):
        """Extract penultimate features for t-SNE."""
        with torch.no_grad():
            feats = self.features(x)
            feats = feats.view(feats.size(0), -1)
        return feats

# =================================================================
# GRAD-CAM
# =================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = F.softmax(output, dim=1)[0, target_class].item()
        self.model.zero_grad()
        output[0, target_class].backward()
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, target_class, confidence

# =================================================================
# DATA LOADING
# =================================================================

def get_transforms():
    train_t = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
            print(f"  ERROR: {path} not found"); sys.exit(1)
        splits[name] = datasets.ImageFolder(path, transform=t)
    
    # Balanced sampler
    targets = [s[1] for s in splits['train'].samples]
    class_counts = Counter(targets)
    weights = [1.0 / class_counts[t] for t in targets]
    sampler = WeightedRandomSampler(weights, len(targets), replacement=True)
    
    loaders = {
        'train': DataLoader(splits['train'], batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=0, pin_memory=True),
        'valid': DataLoader(splits['valid'], batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True),
        'test':  DataLoader(splits['test'],  batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True),
    }
    return loaders, splits['train'].classes, splits['test']

# =================================================================
# TRAINING & EVALUATION
# =================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(imgs)
            loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return loss_sum / total, correct / total

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return correct / total

def measure_inference_speed(model, device, n_runs=100):
    """Measure average inference time per image."""
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs * 1000  # ms
    return elapsed

def train_model(arch_name, loaders, device):
    """Train a model with 2-phase strategy and return best model + metrics."""
    print(f"\n  {'='*50}")
    print(f"  Training: {arch_name.upper()}")
    print(f"  {'='*50}")
    
    set_seed()
    model = MultiArchModel(arch_name).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    best_val_acc, best_wts = 0.0, None
    
    # --- Phase 1: Head only ---
    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Phase 1 ({HEAD_EPOCHS} ep, head only, {trainable:,} trainable)")
    
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=1e-3, weight_decay=1e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HEAD_EPOCHS, eta_min=1e-6)
    
    for ep in range(1, HEAD_EPOCHS + 1):
        tl, ta = train_one_epoch(model, loaders['train'], criterion, opt, device)
        va = evaluate(model, loaders['valid'], device)
        sched.step()
        marker = ""
        if va > best_val_acc:
            best_val_acc = va
            best_wts = copy.deepcopy(model.state_dict())
            marker = " *"
        if ep % 5 == 0 or ep == 1:
            print(f"    Ep {ep:2d}/{HEAD_EPOCHS} | T: {ta:.4f} | V: {va:.4f}{marker}")
    
    # --- Phase 2: Fine-tune ---
    if best_wts: model.load_state_dict(best_wts)
    model.unfreeze_backbone()
    print(f"  Phase 2 ({FINETUNE_EPOCHS} ep, full fine-tune)")
    
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS, eta_min=1e-7)
    
    patience, pctr = 10, 0
    for ep in range(1, FINETUNE_EPOCHS + 1):
        tl, ta = train_one_epoch(model, loaders['train'], criterion, opt, device)
        va = evaluate(model, loaders['valid'], device)
        sched.step()
        improved = va > best_val_acc
        if improved:
            best_val_acc = va
            best_wts = copy.deepcopy(model.state_dict())
            pctr = 0
        else:
            pctr += 1
        if ep % 5 == 0 or ep == 1 or improved:
            marker = " *" if improved else ""
            print(f"    Ep {ep:2d}/{FINETUNE_EPOCHS} | T: {ta:.4f} | V: {va:.4f}{marker}")
        if pctr >= patience:
            print(f"    EarlyStopping at ep {ep}")
            break
    
    # Load best and evaluate test
    model.load_state_dict(best_wts)
    test_acc = evaluate(model, loaders['test'], device)
    inf_time = measure_inference_speed(model, device)
    
    print(f"  RESULT: Val={best_val_acc:.4f} | Test={test_acc:.4f} | {inf_time:.1f}ms/img")
    
    return model, {
        'arch': arch_name,
        'params': total_params,
        'val_acc': round(best_val_acc, 4),
        'test_acc': round(test_acc, 4),
        'inference_ms': round(inf_time, 2),
    }

# =================================================================
# ANALYSIS 1: t-SNE FEATURE SPACE VISUALIZATION
# =================================================================

def generate_tsne(model, test_dataset, device, arch_name, class_names):
    """Extract features and plot t-SNE for a trained model."""
    from sklearn.manifold import TSNE
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    
    print(f"    Extracting features for {arch_name}...")
    model.eval()
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = model.extract_features(imgs)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
    
    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"    Running t-SNE on {all_feats.shape[0]} samples ({all_feats.shape[1]} dims)...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(all_feats)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, cls_name in enumerate(class_names):
        mask = all_labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], 
                  label=cls_name, s=20, alpha=0.7)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_title(f't-SNE Feature Space: {arch_name.upper()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f'tsne_{arch_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    [OK] Saved: {save_path}")

# =================================================================
# ANALYSIS 2: CROSS-ARCHITECTURE GRAD-CAM COMPARISON
# =================================================================

def generate_cross_gradcam(trained_models, device, class_names):
    """Generate side-by-side Grad-CAM heatmaps across architectures."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import matplotlib.cm as colormap
    
    print("\n  Generating cross-architecture Grad-CAM comparison...")
    
    test_dir = os.path.join(DATASET_DIR, 'test')
    n_classes = min(5, len(class_names))  # Show 5 classes for clarity
    n_models = len(trained_models)
    
    fig, axes = plt.subplots(n_classes, n_models + 1, figsize=(4 * (n_models + 1), 4 * n_classes))
    fig.suptitle('Cross-Architecture Grad-CAM Comparison\n'
                 'How different models focus on the same e-waste images',
                 fontsize=14, fontweight='bold', y=1.02)
    
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    for row, class_name in enumerate(class_names[:n_classes]):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        img_files = sorted([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not img_files:
            continue
        
        # Load image
        img_path = os.path.join(class_dir, img_files[0])
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
        
        # Column 0: Original image
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f'{class_name}', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Columns 1+: Grad-CAM for each model
        for col, (arch_name, model) in enumerate(trained_models.items(), 1):
            try:
                grad_cam = GradCAM(model, model.gradcam_layer)
                inp = eval_transform(img_pil).unsqueeze(0).to(device)
                inp.requires_grad_(True)
                heatmap, pred_cls, conf = grad_cam.generate(inp)
                
                # Overlay
                hm_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
                hm_resized = np.array(hm_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS), dtype=np.float32) / 255.0
                hm_colored = (colormap.cm.jet(hm_resized)[:, :, :3] * 255).astype(np.uint8)
                overlay = (hm_colored * 0.45 + img_np * 0.55).astype(np.uint8)
                
                pred_name = class_names[pred_cls]
                correct = "✓" if pred_name == class_name else "✗"
                color = 'green' if pred_name == class_name else 'red'
                
                axes[row, col].imshow(overlay)
                axes[row, col].set_title(f'{arch_name}\n{correct} {pred_name} ({conf:.0%})', 
                                        fontsize=9, color=color)
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error:\n{str(e)[:30]}', 
                                   ha='center', va='center', fontsize=8)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'cross_architecture_gradcam.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_path}")

# =================================================================
# ANALYSIS 3: COMPARISON TABLE & CHARTS
# =================================================================

def generate_comparison_report(all_metrics, class_names):
    """Generate comparison table and radar/bar charts."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    
    print("\n  Generating comparison charts...")
    
    archs = [m['arch'] for m in all_metrics]
    test_accs = [m['test_acc'] * 100 for m in all_metrics]
    params_m = [m['params'] / 1e6 for m in all_metrics]
    inf_times = [m['inference_ms'] for m in all_metrics]
    
    # --- Chart 1: Accuracy vs Parameters vs Speed ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    
    # Accuracy bar chart
    bars = axes[0].bar(archs, test_accs, color=colors[:len(archs)], alpha=0.85)
    axes[0].set_title('Test Accuracy (%)', fontsize=13, fontweight='bold')
    axes[0].set_ylim([min(test_accs) - 5, 100])
    axes[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, test_accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # Parameters bar chart
    bars = axes[1].bar(archs, params_m, color=colors[:len(archs)], alpha=0.85)
    axes[1].set_title('Model Size (M params)', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, params_m):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}M', ha='center', fontsize=10)
    
    # Inference speed bar chart
    bars = axes[2].bar(archs, inf_times, color=colors[:len(archs)], alpha=0.85)
    axes[2].set_title('Inference Time (ms/image)', fontsize=13, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, inf_times):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}ms', ha='center', fontsize=10)
    
    plt.suptitle('Multi-Architecture Comparison for E-Waste Classification',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'architecture_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_path}")
    
    # --- Chart 2: Efficiency scatter (Accuracy vs Speed) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, m in enumerate(all_metrics):
        ax.scatter(m['inference_ms'], m['test_acc'] * 100, 
                  s=m['params'] / 1e4,  # Size = model size
                  c=colors[i], alpha=0.8, edgecolors='black', linewidth=1)
        ax.annotate(m['arch'], (m['inference_ms'], m['test_acc'] * 100),
                   textcoords="offset points", xytext=(10, 5), fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Inference Time (ms) →', fontsize=12)
    ax.set_ylabel('Test Accuracy (%) ↑', fontsize=12)
    ax.set_title('Accuracy vs Speed Tradeoff\n(bubble size = model parameters)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'accuracy_vs_speed.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_path}")
    
    # --- Save JSON report ---
    report = {
        'comparison': all_metrics,
        'recommendation': {
            'best_accuracy': max(all_metrics, key=lambda x: x['test_acc'])['arch'],
            'best_speed': min(all_metrics, key=lambda x: x['inference_ms'])['arch'],
            'best_efficiency': min(all_metrics, key=lambda x: x['params'])['arch'],
        }
    }
    report_path = os.path.join(OUTPUT_DIR, 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  [OK] Saved: {report_path}")

# =================================================================
# MAIN
# =================================================================

def main():
    print("=" * 60)
    print("MULTI-ARCHITECTURE COMPARISON & EXPLAINABILITY STUDY")
    print("Contribution to IOP Conf. Ser.: Earth Environ. Sci. 1529")
    print("=" * 60)
    
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n  [!] Running on CPU (will be slow)")
    
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Output:  {OUTPUT_DIR}\n")
    
    # Load data once
    loaders, class_names, test_dataset = load_data()
    print(f"  Classes: {class_names}")
    print(f"  Train: {len(loaders['train'].dataset)}, "
          f"Valid: {len(loaders['valid'].dataset)}, "
          f"Test: {len(loaders['test'].dataset)}")
    
    # =============================
    # TRAIN ALL 4 ARCHITECTURES
    # =============================
    architectures = ['resnet18', 'resnet50', 'mobilenetv3', 'efficientnet_b0']
    trained_models = {}
    all_metrics = []
    
    for arch in architectures:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, metrics = train_model(arch, loaders, device)
        trained_models[arch] = model
        all_metrics.append(metrics)
    
    # =============================
    # PRINT COMPARISON TABLE
    # =============================
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Architecture':<18} {'Params':>10} {'Val Acc':>10} {'Test Acc':>10} {'Speed':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for m in all_metrics:
        print(f"  {m['arch']:<18} {m['params']/1e6:>8.1f}M {m['val_acc']*100:>9.1f}% "
              f"{m['test_acc']*100:>9.1f}% {m['inference_ms']:>8.1f}ms")
    print("=" * 70)
    
    # =============================
    # ANALYSIS 1: t-SNE for each model
    # =============================
    print("\n-- Analysis 1: t-SNE Feature Space Visualization --")
    for arch, model in trained_models.items():
        generate_tsne(model, test_dataset, device, arch, class_names)
    
    # =============================
    # ANALYSIS 2: Cross-architecture Grad-CAM
    # =============================
    print("\n-- Analysis 2: Cross-Architecture Grad-CAM --")
    generate_cross_gradcam(trained_models, device, class_names)
    
    # =============================
    # ANALYSIS 3: Comparison charts
    # =============================
    print("\n-- Analysis 3: Comparison Charts & Report --")
    generate_comparison_report(all_metrics, class_names)
    
    # =============================
    # FINAL SUMMARY
    # =============================
    best = max(all_metrics, key=lambda x: x['test_acc'])
    fastest = min(all_metrics, key=lambda x: x['inference_ms'])
    smallest = min(all_metrics, key=lambda x: x['params'])
    
    print(f"\n{'='*60}")
    print(f"STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"  Best Accuracy:  {best['arch']} ({best['test_acc']*100:.1f}%)")
    print(f"  Fastest:        {fastest['arch']} ({fastest['inference_ms']:.1f}ms)")
    print(f"  Smallest:       {smallest['arch']} ({smallest['params']/1e6:.1f}M params)")
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")
    
    if os.path.isdir(OUTPUT_DIR):
        print("\n  Generated files:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f"    {f:45s} ({size_kb:.1f} KB)")

if __name__ == '__main__':
    main()
