"""
Training Analysis and Visualization Script

This script loads training history and creates comprehensive visualizations
of the training process including loss curves, PSNR, Dice scores, and learning rate.

Usage:
    python analyze_training.py
    python analyze_training.py --history_path ./checkpoints/training_history.json
    python analyze_training.py --output_path ./training_report.png
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_training_history(history_path):
    """Load training history from JSON file."""
    print(f"📂 Loading training history from: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Print summary statistics
    print(f"\n📊 Training Summary:")
    print(f"   Total epochs: {len(history['train_loss'])}")
    print(f"   Best validation Dice: {max(history['val_dice']):.4f}")
    print(f"   Best validation PSNR: {max(history['val_psnr']):.2f} dB")
    print(f"   Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final validation loss: {history['val_loss'][-1]:.4f}")
    
    return history


def plot_training_curves(history, output_path=None):
    """Create comprehensive training visualization."""
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Training and Validation Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2, color='#1f77b4')
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Find best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch}')
    ax1.legend(fontsize=10)
    
    # Plot 2: PSNR
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_psnr'], label='Training PSNR', linewidth=2, color='#2ca02c')
    ax2.plot(epochs, history['val_psnr'], label='Validation PSNR', linewidth=2, color='#d62728')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('Peak Signal-to-Noise Ratio', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Find best PSNR epoch
    best_psnr_epoch = np.argmax(history['val_psnr']) + 1
    ax2.axvline(x=best_psnr_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_psnr_epoch}')
    ax2.legend(fontsize=10)
    
    # Plot 3: Dice Score
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, history['val_dice'], label='Validation Dice', linewidth=2, color='#9467bd')
    ax3.axhline(y=0.974, color='red', linestyle='--', linewidth=2, label='Target (0.974)', alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    ax3.set_title('Dice Coefficient (Overlap Metric)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Find best Dice epoch
    best_dice_epoch = np.argmax(history['val_dice']) + 1
    best_dice_score = max(history['val_dice'])
    ax3.axvline(x=best_dice_epoch, color='green', linestyle='--', alpha=0.5)
    ax3.text(best_dice_epoch, best_dice_score, f' Best: {best_dice_score:.4f}', 
             fontsize=10, va='bottom', ha='left', color='green', fontweight='bold')
    
    # Plot 4: Learning Rate
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, history['learning_rates'], label='Learning Rate', linewidth=2, color='#8c564b')
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Loss Comparison (detailed view)
    ax5 = fig.add_subplot(gs[2, 0])
    train_val_ratio = [t / v if v != 0 else 1 for t, v in zip(history['train_loss'], history['val_loss'])]
    ax5.plot(epochs, train_val_ratio, label='Train/Val Loss Ratio', linewidth=2, color='#e377c2')
    ax5.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Ratio', fontsize=12, fontweight='bold')
    ax5.set_title('Overfitting Indicator (Train/Val Loss Ratio)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.fill_between(epochs, train_val_ratio, 1, where=[r < 1 for r in train_val_ratio], 
                      alpha=0.3, color='green', label='Val > Train (Good)')
    ax5.fill_between(epochs, 1, train_val_ratio, where=[r > 1 for r in train_val_ratio], 
                      alpha=0.3, color='red', label='Train > Val (Overfitting)')
    ax5.legend(fontsize=9)
    
    # Plot 6: Statistics Summary (text)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # Calculate statistics
    stats_text = f"""
    📊 TRAINING STATISTICS
    {'=' * 40}
    
    Training Epochs: {len(epochs)}
    
    LOSS METRICS:
    • Best Val Loss: {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})
    • Final Train Loss: {history['train_loss'][-1]:.4f}
    • Final Val Loss: {history['val_loss'][-1]:.4f}
    • Loss Reduction: {((history['val_loss'][0] - history['val_loss'][-1]) / history['val_loss'][0] * 100):.1f}%
    
    PSNR METRICS:
    • Best Val PSNR: {max(history['val_psnr']):.2f} dB (Epoch {np.argmax(history['val_psnr']) + 1})
    • Final Train PSNR: {history['train_psnr'][-1]:.2f} dB
    • Final Val PSNR: {history['val_psnr'][-1]:.2f} dB
    • PSNR Improvement: {(history['val_psnr'][-1] - history['val_psnr'][0]):.2f} dB
    
    DICE COEFFICIENT:
    • Best Dice: {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})
    • Final Dice: {history['val_dice'][-1]:.4f}
    • Target Dice: 0.9740
    • Target Achieved: {'✅ YES' if max(history['val_dice']) >= 0.974 else '❌ NO'}
    
    LEARNING RATE:
    • Initial LR: {history['learning_rates'][0]:.6f}
    • Final LR: {history['learning_rates'][-1]:.6f}
    • LR Reductions: {sum(1 for i in range(1, len(history['learning_rates'])) if history['learning_rates'][i] < history['learning_rates'][i-1])}
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    fig.suptitle('🧠 UNet Skull Image Inpainting - Training Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Training analysis saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_simple_curves(history, output_path=None):
    """Create simple 2x2 plot for quick visualization."""
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR
    axes[0, 1].plot(epochs, history['train_psnr'], label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_psnr'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('PSNR')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dice
    axes[1, 0].plot(epochs, history['val_dice'], linewidth=2, color='green')
    axes[1, 0].axhline(y=0.974, color='red', linestyle='--', linewidth=2, label='Target')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].set_title('Dice Coefficient')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['learning_rates'], linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"💾 Simple plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize training results')
    parser.add_argument('--history_path', type=str, default='./checkpoints/training_history.json',
                       help='Path to training history JSON file')
    parser.add_argument('--output_path', type=str, default='./training_analysis.png',
                       help='Path to save analysis plot')
    parser.add_argument('--simple', action='store_true',
                       help='Create simple 2x2 plot instead of detailed analysis')
    
    args = parser.parse_args()
    
    # Check if history file exists
    if not Path(args.history_path).exists():
        print(f"❌ Error: Training history file not found: {args.history_path}")
        print(f"   Please train a model first using: python train_model.py")
        return
    
    # Load history
    try:
        history = load_training_history(args.history_path)
    except Exception as e:
        print(f"❌ Error loading training history: {e}")
        return
    
    # Create visualization
    try:
        if args.simple:
            plot_simple_curves(history, args.output_path)
        else:
            plot_training_curves(history, args.output_path)
        
        print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
