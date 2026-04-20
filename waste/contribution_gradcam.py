"""
contribution_gradcam.py -- Grad-CAM Explainability Analysis (PyTorch)
Contribution to: IOP Conf. Ser.: Earth Environ. Sci. 1529 (2025) 012032

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) on the
ResNet18 backbone to provide visual explainability for model predictions.

Outputs:
1. Per-class Grad-CAM heatmap overlays (correct predictions)
2. Misclassification analysis with heatmaps
3. Average attention maps per class
4. Quantitative attention metrics
"""

import os
os.environ['TORCH_HOME'] = 'D:\\torch_cache'
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# =================================================================
# CONFIGURATION
# =================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset_full")
if not os.path.isdir(DATASET_DIR):
    DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")

MODEL_PATH = os.path.join(RESULTS_DIR, "best_ewaste_model.pth")
IMG_SIZE = 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    'Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse',
    'PCB', 'Player', 'Printer', 'Television', 'WashingMachine'
]

# Import the model class
sys.path.insert(0, SCRIPT_DIR)
from ewaste_cnn_paper import EWasteCNN


# =================================================================
# GRAD-CAM IMPLEMENTATION
# =================================================================

class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) for PyTorch models."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
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
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, target_class, confidence


def get_target_layer(model):
    """Get the appropriate target layer for Grad-CAM based on model architecture.
    
    For ResNet18: the last conv layer in layer4[1].conv2
    For custom CNN: the last Conv2d in features Sequential
    """
    # Try ResNet-style: look for layer4 inside features
    features = model.features
    
    # features is nn.Sequential wrapping ResNet layers:
    # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
    children = list(features.children())
    
    # Look for the last layer that contains conv layers (layer4)
    for child in reversed(children):
        if hasattr(child, 'children'):
            sub_children = list(child.children())
            if len(sub_children) > 0:
                # Found a compound module (like layer4)
                for sub in reversed(sub_children):
                    if hasattr(sub, 'conv2'):
                        # ResNet BasicBlock -- target conv2
                        return sub.conv2
                    elif hasattr(sub, 'conv3'):
                        # ResNet Bottleneck -- target conv3
                        return sub.conv3
        # Direct Conv2d layer
        if isinstance(child, torch.nn.Conv2d):
            return child
    
    # Fallback: last Conv2d anywhere in features
    last_conv = None
    for module in features.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def overlay_heatmap(img_np, heatmap, alpha=0.4):
    """Overlay a Grad-CAM heatmap on the original image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_pil = heatmap_pil.resize((img_np.shape[1], img_np.shape[0]), Image.LANCZOS)
    heatmap_resized = np.array(heatmap_pil, dtype=np.float32) / 255.0
    
    cmap = cm.get_cmap('jet')
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    superimposed = (heatmap_colored * alpha + img_np * (1 - alpha)).astype(np.uint8)
    return superimposed


# =================================================================
# SETUP
# =================================================================

def setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  [!] Using CPU")
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run ewaste_cnn_paper.py first!")
        sys.exit(1)
    
    model = EWasteCNN(pretrained=False).to(device)  # Don't re-download weights
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    class_names = checkpoint.get('class_names', CLASS_NAMES)
    
    print(f"  [OK] Model loaded from epoch {checkpoint['epoch']}")
    print(f"       Val accuracy: {checkpoint['val_accuracy']:.4f}")
    print(f"  Classes: {class_names}")
    
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    
    return model, device, class_names


def load_image(img_path, device):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    img_tensor = transform(img).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)
    
    return img_tensor, img_np


# =================================================================
# ANALYSIS FUNCTIONS
# =================================================================

def generate_per_class_gradcam(model, device, class_names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("\n  Generating per-class Grad-CAM visualizations...")
    
    target_layer = get_target_layer(model)
    print(f"  Target layer: {target_layer}")
    grad_cam = GradCAM(model, target_layer)
    
    test_dir = os.path.join(DATASET_DIR, 'test')
    n_classes = len(class_names)
    
    fig, axes = plt.subplots(n_classes, 3, figsize=(12, 4 * n_classes))
    fig.suptitle('Grad-CAM Analysis: Per-Class Attention Maps\n'
                 '(Where the model focuses when classifying each e-waste type)',
                 fontsize=16, fontweight='bold', y=1.01)
    
    for row, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            for col in range(3):
                axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[row, col].axis('off')
            continue
        
        image_files = sorted([f for f in os.listdir(class_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not image_files:
            continue
        
        img_path = os.path.join(class_dir, image_files[0])
        img_tensor, img_np = load_image(img_path, device)
        
        heatmap, pred_class, confidence = grad_cam.generate(img_tensor)
        pred_name = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"
        
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f'Original\n({class_name})', fontsize=10)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(heatmap, cmap='jet', aspect='auto')
        axes[row, 1].set_title('Grad-CAM Heatmap', fontsize=10)
        axes[row, 1].axis('off')
        
        superimposed = overlay_heatmap(img_np, heatmap)
        correct = "[OK]" if pred_name == class_name else "[X]"
        color = 'green' if pred_name == class_name else 'red'
        axes[row, 2].imshow(superimposed)
        axes[row, 2].set_title(
            f'{correct} Pred: {pred_name}\nConf: {confidence:.2%}',
            fontsize=10, color=color)
        axes[row, 2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(GRADCAM_DIR, 'per_class_gradcam.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Per-class Grad-CAM saved: {save_path}")


def generate_misclassification_analysis(model, device, class_names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("\n  Generating misclassification analysis...")
    
    target_layer = get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)
    
    test_dir = os.path.join(DATASET_DIR, 'test')
    misclassified = []
    total_correct = 0
    total_wrong = 0
    
    for true_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(class_dir, img_file)
            try:
                img_tensor, img_np = load_image(img_path, device)
                heatmap, pred_class, confidence = grad_cam.generate(img_tensor)
                
                if pred_class != true_idx:
                    total_wrong += 1
                    misclassified.append({
                        'img_np': img_np,
                        'heatmap': heatmap,
                        'true_class': class_name,
                        'true_idx': true_idx,
                        'pred_class': class_names[pred_class],
                        'pred_idx': pred_class,
                        'confidence': confidence,
                        'filename': img_file
                    })
                else:
                    total_correct += 1
            except Exception:
                continue
    
    print(f"  Total correct: {total_correct}, Total wrong: {total_wrong}")
    
    if not misclassified:
        print("  No misclassifications found!")
        return
    
    misclassified.sort(key=lambda x: x['confidence'], reverse=True)
    n_show = min(10, len(misclassified))
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
    fig.suptitle('Grad-CAM Misclassification Analysis\n'
                 '(Most confident mistakes first)',
                 fontsize=14, fontweight='bold', y=1.01)
    
    if n_show == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_show):
        mc = misclassified[i]
        axes[i, 0].imshow(mc['img_np'])
        axes[i, 0].set_title(f"True: {mc['true_class']}", fontsize=10, color='green')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mc['heatmap'], cmap='jet', aspect='auto')
        axes[i, 1].set_title('Grad-CAM Attention', fontsize=10)
        axes[i, 1].axis('off')
        
        superimposed = overlay_heatmap(mc['img_np'], mc['heatmap'])
        axes[i, 2].imshow(superimposed)
        axes[i, 2].set_title(
            f"[X] Pred: {mc['pred_class']}\nConf: {mc['confidence']:.2%}",
            fontsize=10, color='red')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRADCAM_DIR, 'misclassification_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Misclassification analysis saved")
    
    summary = [{'filename': mc['filename'], 'true_class': mc['true_class'],
                'pred_class': mc['pred_class'], 'confidence': round(mc['confidence'], 4)}
               for mc in misclassified]
    with open(os.path.join(GRADCAM_DIR, 'misclassifications.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def generate_average_attention_maps(model, device, class_names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("\n  Generating average attention maps...")
    
    target_layer = get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)
    
    test_dir = os.path.join(DATASET_DIR, 'test')
    avg_heatmaps = {}
    
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        heatmaps = []
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            try:
                img_tensor, _ = load_image(os.path.join(class_dir, img_file), device)
                heatmap, pred_class, _ = grad_cam.generate(img_tensor)
                
                if pred_class == class_names.index(class_name):
                    hm_resized = np.array(
                        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                            (IMG_SIZE, IMG_SIZE), Image.LANCZOS),
                        dtype=np.float32) / 255.0
                    heatmaps.append(hm_resized)
            except Exception:
                continue
        
        if heatmaps:
            avg_heatmaps[class_name] = np.mean(heatmaps, axis=0)
            print(f"    {class_name}: averaged {len(heatmaps)} heatmaps")
    
    n = len(avg_heatmaps)
    cols = 5
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle('Average Grad-CAM Attention Maps per Class\n'
                 '(Where the model typically looks for each e-waste type)',
                 fontsize=14, fontweight='bold')
    
    axes_flat = axes.flatten() if n > 1 else [axes]
    
    for i, (cls, avg_hm) in enumerate(sorted(avg_heatmaps.items())):
        axes_flat[i].imshow(avg_hm, cmap='jet', aspect='equal')
        axes_flat[i].set_title(cls, fontsize=12, fontweight='bold')
        axes_flat[i].axis('off')
    
    for i in range(len(avg_heatmaps), len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRADCAM_DIR, 'average_attention_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Average attention maps saved")


def compute_attention_metrics(model, device, class_names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("\n  Computing attention metrics...")
    
    target_layer = get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)
    
    test_dir = os.path.join(DATASET_DIR, 'test')
    metrics = {}
    
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        focus_scores = []
        coverage_scores = []
        peak_intensities = []
        
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            try:
                img_tensor, _ = load_image(os.path.join(class_dir, img_file), device)
                heatmap, _, _ = grad_cam.generate(img_tensor)
                
                hm_flat = heatmap.flatten()
                hm_flat = hm_flat / (hm_flat.sum() + 1e-8)
                entropy = -np.sum(hm_flat * np.log(hm_flat + 1e-8))
                max_entropy = np.log(len(hm_flat))
                focus_score = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
                focus_scores.append(focus_score)
                
                hm_full = np.array(
                    Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                        (IMG_SIZE, IMG_SIZE), Image.LANCZOS),
                    dtype=np.float32) / 255.0
                coverage = np.mean(hm_full > 0.5)
                coverage_scores.append(coverage)
                
                peak_intensities.append(float(np.max(heatmap)))
            except Exception:
                continue
        
        if focus_scores:
            metrics[class_name] = {
                'focus_score': round(float(np.mean(focus_scores)), 4),
                'coverage': round(float(np.mean(coverage_scores)), 4),
                'peak_intensity': round(float(np.mean(peak_intensities)), 4),
                'n_samples': len(focus_scores)
            }
    
    print("\n  +---------------------+----------+----------+-----------+")
    print("  | Class               | Focus    | Coverage | Peak Int. |")
    print("  +---------------------+----------+----------+-----------+")
    for cls in sorted(metrics.keys()):
        m = metrics[cls]
        print(f"  | {cls:19s} | {m['focus_score']:.4f}   | {m['coverage']:.4f}   | {m['peak_intensity']:.4f}    |")
    print("  +---------------------+----------+----------+-----------+")
    
    with open(os.path.join(GRADCAM_DIR, 'attention_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  [OK] Attention metrics saved")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    classes_sorted = sorted(metrics.keys())
    x = np.arange(len(classes_sorted))
    
    for ax, key, title, color in [
        (axes[0], 'focus_score', 'Attention Focus Score\n(Higher = More Focused)', 'steelblue'),
        (axes[1], 'coverage', 'Attention Coverage\n(Fraction with high attention)', 'coral'),
        (axes[2], 'peak_intensity', 'Peak Attention Intensity\n(Maximum activation strength)', 'seagreen'),
    ]:
        vals = [metrics[c][key] for c in classes_sorted]
        ax.bar(x, vals, color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(classes_sorted, rotation=45, ha='right')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim([0, max(1, max(vals) * 1.1)])
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Grad-CAM Quantitative Attention Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(GRADCAM_DIR, 'attention_metrics_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Attention metrics chart saved")
    
    return metrics


# =================================================================
# MAIN
# =================================================================

def main():
    print("=" * 60)
    print("GRAD-CAM EXPLAINABILITY ANALYSIS (PyTorch)")
    print("E-Waste CNN -- Contribution Module")
    print("=" * 60)
    
    model, device, class_names = setup()
    
    print("\n-- Analysis 1: Per-Class Grad-CAM --")
    generate_per_class_gradcam(model, device, class_names)
    
    print("\n-- Analysis 2: Misclassification Analysis --")
    generate_misclassification_analysis(model, device, class_names)
    
    print("\n-- Analysis 3: Average Attention Maps --")
    generate_average_attention_maps(model, device, class_names)
    
    print("\n-- Analysis 4: Quantitative Attention Metrics --")
    compute_attention_metrics(model, device, class_names)
    
    print("\n" + "=" * 60)
    print("GRAD-CAM ANALYSIS COMPLETE")
    print(f"All outputs saved to: {GRADCAM_DIR}")
    print("=" * 60)
    
    if os.path.isdir(GRADCAM_DIR):
        print("\nGenerated files:")
        for f in sorted(os.listdir(GRADCAM_DIR)):
            fpath = os.path.join(GRADCAM_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f:40s} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
