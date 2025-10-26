"""
Training script for UNet Image Inpainting Model

This script trains a UNet model to reconstruct masked regions in skull images.
It includes checkpointing, early stopping, and comprehensive logging.

Usage:
    python train_model.py --epochs 100 --save_dir ./checkpoints
    python train_model.py --resume ./checkpoints/best_model.pth --epochs 50
"""

import argparse
import os
import sys
import time
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
import lpips

import config as cf
from Unet_Architecture.Image_Painting.Prepare_Dataset import data
from Unet_Architecture.Image_Painting.Train_Val import (
    train_epoch, 
    valid_epoch, 
    generate_images_from_batch
)
from Unet_Architecture.Image_Painting.UnetSkipConnection import Unet


def save_checkpoint(model, optimizer, epoch, val_dice, val_loss, save_path, is_best=False):
    """
    Save model checkpoint with training state.
    
    Args:
        model: The neural network model
        optimizer: The optimizer state
        epoch: Current epoch number
        val_dice: Validation Dice score
        val_loss: Validation loss
        save_path: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
        'val_loss': val_loss,
        'config': {
            'IMAGE_WIDTH': cf.IMAGE_WIDTH,
            'IMAGE_HEIGHT': cf.IMAGE_HEIGHT,
            'LEARNING_RATE': cf.LEARNING_RATE,
            'BATCH_SIZE': cf.BATCH_SIZE,
            'perceptual_loss_rate': cf.perceptual_loss_rate
        }
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save best model separately
    if is_best:
        best_path = os.path.join(save_path, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"🏆 Best model saved: {best_path} (Dice: {val_dice:.4f})")
    
    # Also save model-only for easier inference loading
    model_only_path = os.path.join(save_path, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_only_path)
    
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint and resume training state.
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        start_epoch: Epoch to resume from
        best_dice: Best Dice score from checkpoint
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_dice = checkpoint.get('val_dice', 0.0)
    
    print(f"✅ Resumed from epoch {checkpoint['epoch']}")
    print(f"   Previous best Dice: {best_dice:.4f}")
    
    return start_epoch, best_dice


def train(args):
    """Main training function."""
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Checkpoints will be saved to: {save_dir}")
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    print(f"   Train: {cf.PATH_TRAIN}")
    print(f"   Val: {cf.PATH_VAL}")
    
    train_dataset, val_dataset, train_loader, val_loader = data(
        cf.PATH_TRAIN, 
        cf.PATH_VAL, 
        cf.IMAGE_WIDTH, 
        cf.IMAGE_HEIGHT, 
        cf.BATCH_SIZE
    )
    
    print(f"✅ Dataset loaded:")
    print(f"   Train images: {len(train_dataset)}")
    print(f"   Val images: {len(val_dataset)}")
    print(f"   Batch size: {cf.BATCH_SIZE}")
    
    # Initialize model
    print(f"\n🏗️  Building model...")
    in_channels = 1
    num_classes = 1
    model = Unet(in_channels, num_classes)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Setup loss functions and optimizer
    criterion_bce = nn.BCEWithLogitsLoss().to(device)
    criterion_lpips = lpips.LPIPS(net='vgg').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.5, 0.999])
    
    # Setup learning rate scheduler
    # Some torch builds may not support 'verbose' kwarg; omit for compatibility
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    print(f"\n⚙️  Training configuration:")
    print(f"   Learning rate: {args.lr}")
    print(f"   Perceptual loss weight: {cf.perceptual_loss_rate}")
    print(f"   Max epochs: {args.epochs}")
    print(f"   Early stopping patience: {args.patience}")
    print(f"   Target Dice score: {cf.TARGET_DICE_SCORE}")
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_dice = 0.0
    patience_counter = 0
    
    if args.resume:
        start_epoch, best_dice = load_checkpoint(model, optimizer, args.resume, device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_psnr': [],
        'val_loss': [],
        'val_psnr': [],
        'val_dice': [],
        'learning_rates': []
    }
    
    # Training loop
    print(f"\n🚀 Starting training...\n")
    print("=" * 80)
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_psnr, train_loss = train_epoch(
            model, optimizer, criterion_bce, criterion_lpips,
            train_loader, device, epoch,
            cf.perceptual_loss_rate, log_interval=50
        )
        
        # Validate
        val_psnr, val_loss, val_dice = valid_epoch(
            model, criterion_bce, criterion_lpips,
            val_loader, device, cf.perceptual_loss_rate
        )
        
        # Update learning rate based on validation Dice
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(float(train_loss))
        history['train_psnr'].append(float(train_psnr.cpu() if torch.is_tensor(train_psnr) else train_psnr))
        history['val_loss'].append(float(val_loss))
        history['val_psnr'].append(float(val_psnr.cpu() if torch.is_tensor(val_psnr) else val_psnr))
        history['val_dice'].append(float(val_dice))
        history['learning_rates'].append(current_lr)
        
        # Check if best model
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        if epoch % args.save_every == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, val_dice, val_loss,
                save_dir, is_best=is_best
            )
        
        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        print("=" * 80)
        print(f"📊 Epoch {epoch}/{start_epoch + args.epochs - 1} Summary:")
        print(f"   Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        print(f"   Train - Loss: {train_loss:.4f} | PSNR: {train_psnr:.2f} dB")
        print(f"   Val   - Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | Dice: {val_dice:.4f}")
        if is_best:
            print(f"   🌟 New best model! (Previous: {best_dice - val_dice:.4f})")
        print(f"   Patience: {patience_counter}/{args.patience}")
        print("=" * 80 + "\n")
        
        # Save training history
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"⚠️  Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        # Check if target Dice reached
        if val_dice >= cf.TARGET_DICE_SCORE:
            print(f"🎯 Target Dice score {cf.TARGET_DICE_SCORE} reached! Dice: {val_dice:.4f}")
            save_checkpoint(model, optimizer, epoch, val_dice, val_loss, save_dir, is_best=True)
            break
        
        # Generate sample predictions every N epochs
        if epoch % args.visualize_every == 0:
            print(f"🎨 Generating sample predictions...")
            model.eval()
            with torch.no_grad():
                inputs, labels, mask_learn = next(iter(val_loader))
                inputs = inputs.to(device)
                labels = labels.to(device)
                mask_learn = mask_learn.to(device)
                predictions = torch.sigmoid(model(inputs))
                
                # Save to save_dir/samples/
                samples_dir = save_dir / 'samples'
                samples_dir.mkdir(exist_ok=True)
                
                import matplotlib.pyplot as plt
                for i in range(min(3, inputs.size(0))):
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    
                    inp = ((inputs[i].cpu() + 1) / 2).squeeze().numpy()
                    pred = predictions[i].cpu().squeeze().numpy()
                    gt = ((labels[i].cpu() + 1) / 2).squeeze().numpy()
                    mask = mask_learn[i].cpu().squeeze().numpy()
                    
                    axes[0].imshow(inp, cmap='gray')
                    axes[0].set_title('Input (Masked)')
                    axes[0].axis('off')
                    
                    axes[1].imshow(pred, cmap='gray')
                    axes[1].set_title('Prediction')
                    axes[1].axis('off')
                    
                    axes[2].imshow(gt, cmap='gray')
                    axes[2].set_title('Ground Truth')
                    axes[2].axis('off')
                    
                    axes[3].imshow(mask, cmap='gray')
                    axes[3].set_title('Mask Region')
                    axes[3].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(samples_dir / f'epoch_{epoch}_sample_{i}.png', dpi=150, bbox_inches='tight')
                    plt.close()
            model.train()
    
    # Training complete
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print(f"   Best Dice score: {best_dice:.4f}")
    print(f"   Best model saved at: {save_dir / 'best_model.pth'}")
    print(f"   Training history saved at: {save_dir / 'training_history.json'}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train UNet for skull image inpainting')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=cf.MAX_EPOCHS_PER_ROUND,
                        help=f'Number of training epochs (default: {cf.MAX_EPOCHS_PER_ROUND})')
    parser.add_argument('--lr', type=float, default=cf.LEARNING_RATE,
                        help=f'Learning rate (default: {cf.LEARNING_RATE})')
    parser.add_argument('--batch_size', type=int, default=cf.BATCH_SIZE,
                        help=f'Batch size (default: {cf.BATCH_SIZE})')
    
    # Checkpoint parameters
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    
    # Visualization
    parser.add_argument('--visualize_every', type=int, default=10,
                        help='Generate sample predictions every N epochs (default: 10)')
    
    args = parser.parse_args()
    
    # Update config if batch_size is provided
    if args.batch_size != cf.BATCH_SIZE:
        cf.BATCH_SIZE = args.batch_size
    
    try:
        train(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
