"""This script trains and evaluates the attribution-guided model for the study."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from attribution_guided_model import create_attribution_guided_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              cohen_kappa_score, matthews_corrcoef,
                              confusion_matrix)
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
import json

      
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def train(model, train_loader, val_loader, num_epochs=200, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    learning_rate = 0.00005

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

                                            
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(20, num_epochs // 4),
        T_mult=2,
        eta_min=1e-6
    )

                                
                        
                                                       
                                               
                                                     
                                                                
                                          
    best_ema_loss = float('inf')                  
    min_delta     = 1e-4                     
    best_val_acc  = 0.0                              
    no_improve_epochs = 0
    max_no_improve    = 50

                                             
                               
    ema_val_loss = None
    ema_alpha    = 0.3

                                   
    prev_lr = learning_rate

                                         
    converge_epoch   = None                                            
    converge_metrics = None                         

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_kappa': [], 'val_mcc': [],
        'ema_val_loss': [], 'composite': [],
        'learning_rate': []
    }

    print(f"\n[Warning] Small dataset detected ({len(train_loader.dataset)} samples)")
    print(f"Automatically switched to a small-dataset optimized configuration:")
    print(f"  - learning rate: {learning_rate} (reduced to 5e-5)")
    print(f"  - : 0.01")
    print(f"  - Dropout: 0.15")
    print(f"  - Early stopping: {max_no_improve} epochs (dual-metric composite)")
    print(f"  - Warmup epochs: 5")
    print(f"  - Gradient clipping: 1.0")
    print(f"  - LR scheduler: CosineAnnealingWarmRestarts (T_0={max(20, num_epochs//4)}, T_mult=2)\n")

    warmup_epochs = 5
    base_lr = learning_rate

    for epoch in range(num_epochs):
                                              
                                                       
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * warmup_factor
            print(f"Warmup stage ({epoch+1}/{warmup_epochs}), learning rate: {base_lr * warmup_factor:.2e}")

                                  
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"\nEpoch {epoch+1}/{num_epochs} starting training:")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            output, attribution_scores = model(data, return_attribution=True)

            loss = criterion(output, target)

                                       
                                           
                                              
            scores_flat = attribution_scores.view(data.size(0), -1)            
            attribution_contrast = -torch.var(scores_flat, dim=1).mean()
            loss = loss + 0.001 * attribution_contrast

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

            if batch_idx % 5 == 0:
                print(f'Batch: {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*train_correct/train_total:.2f}%, '
                      f'Attr Contrast(var): {-attribution_contrast.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

                                                         
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)

        print(f'Epoch {epoch+1} training complete - mean loss: {avg_train_loss:.4f}, '
              f'Accuracy: {train_acc:.2f}%, learning rate: {current_lr:.2e}')

                                  
        model.eval()
        val_loss = 0.0
        all_preds  = []                       
        all_labels = []                

        print(f"\nEpoch {epoch+1} starting validation:")
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data, return_attribution=False)

                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

                if batch_idx % 5 == 0:
                    cur_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100
                    print(f'Validation Batch: {batch_idx+1}/{len(val_loader)}, '
                          f'Accumulated Acc: {cur_acc:.2f}%')

        avg_val_loss = val_loss / len(val_loader)

                                             
                               
        val_acc       = 100. * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall    = recall_score(all_labels, all_preds, zero_division=0)
        val_f1        = f1_score(all_labels, all_preds, zero_division=0)
        val_kappa     = cohen_kappa_score(all_labels, all_preds) if len(set(all_preds)) > 1 else 0.0
        val_mcc       = matthews_corrcoef(all_labels, all_preds)

                                                           
        if train_acc >= 100.0 and converge_epoch is None:
            converge_epoch = epoch + 1
            converge_metrics = {
                'epoch':     converge_epoch,
                'acc':       float(val_acc),
                'precision': float(val_precision),
                'recall':    float(val_recall),
                'f1':        float(val_f1),
                'kappa':     float(val_kappa),
                'mcc':       float(val_mcc),
                'val_loss':  float(avg_val_loss),
            }
            print(f'\n{"*"*60}')
            print(f'[] Training SetAccuracy 100%（Epoch {converge_epoch}）')
            print(f'         Validation Set six metrics at this point:')
            print(f'  ACC:       {val_acc:.2f}%')
            print(f'  Precision: {val_precision:.4f}')
            print(f'  Recall:    {val_recall:.4f}')
            print(f'  F1:        {val_f1:.4f}')
            print(f'  Kappa:     {val_kappa:.4f}')
            print(f'  MCC:       {val_mcc:.4f}')
            print(f'{"*"*60}')

                                         
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            cm = confusion_matrix(all_labels, all_preds)
            print(f'  Confusion matrix (rows=true, cols=predicted):')
            print(f'    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}')
            print(f'    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}')

        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_kappa'].append(val_kappa)
        history['val_mcc'].append(val_mcc)

                                               
        if ema_val_loss is None:
            ema_val_loss = avg_val_loss
        else:
            ema_val_loss = ema_alpha * avg_val_loss + (1.0 - ema_alpha) * ema_val_loss
        history['ema_val_loss'].append(ema_val_loss)

                                   
        composite = val_f1 * 100 - 0.1 * ema_val_loss
        history['composite'].append(composite)

        print(f'\nEpoch {epoch+1} validation metrics:')
        print(f'  Loss: {avg_val_loss:.4f} (EMA: {ema_val_loss:.4f})')
        print(f'  ACC:       {val_acc:.2f}%')
        print(f'  Precision: {val_precision:.4f}')
        print(f'  Recall:    {val_recall:.4f}')
        print(f'  F1:        {val_f1:.4f}')
        print(f'  Kappa:     {val_kappa:.4f}')
        print(f'  MCC:       {val_mcc:.4f}')

                           
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

                                                 
                                                                     
                                
        if epoch >= warmup_epochs and current_lr > prev_lr * 1.5:
            print(f'[LR Restart] Learning rate jumped {prev_lr:.2e} → {current_lr:.2e}，'
                  f'resetting early-stopping counter (previous value: {no_improve_epochs}）')
            no_improve_epochs = 0
        prev_lr = current_lr

                                                  
                                               
        if ema_val_loss < best_ema_loss - min_delta:
            best_ema_loss = ema_val_loss
            best_val_acc  = val_acc
            no_improve_epochs = 0

                                                                            
                                                                   
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch':         int(epoch),
                'val_acc':       float(val_acc),
                'val_loss':      float(avg_val_loss),
                'ema_val_loss':  float(ema_val_loss),
                'val_precision': float(val_precision),
                'val_recall':    float(val_recall),
                'val_f1':        float(val_f1),
                'val_kappa':     float(val_kappa),
                'val_mcc':       float(val_mcc),
            }, 'best_attribution_guided_model.pth')

            print(f'[Save] EMA loss improved -> {ema_val_loss:.6f} '
                  f'(Δ={best_ema_loss - ema_val_loss + min_delta:.6f}), '
                  f'Acc: {val_acc:.2f}%, F1: {val_f1:.4f}, '
                  f'Kappa: {val_kappa:.4f}, MCC: {val_mcc:.4f}')
        else:
            no_improve_epochs += 1
            print(f'EMA loss did not improve sufficiently for {no_improve_epochs}/{max_no_improve} epoch '
                  f'(best ema_loss: {best_ema_loss:.6f}, current: {ema_val_loss:.6f})')

            if no_improve_epochs >= max_no_improve:
                print(f'{max_no_improve}  epoch EMA Validation Loss，')
                break

                                                   
    best_epoch_idx = int(np.argmin(history['ema_val_loss']))
    print(f"\n{'='*60}")
    print(f"Training completed! Best epoch: {best_epoch_idx + 1}")
    print(f"{'='*60}")
    print(f"  ACC:       {history['val_acc'][best_epoch_idx]:.2f}%")
    print(f"  Precision: {history['val_precision'][best_epoch_idx]:.4f}")
    print(f"  Recall:    {history['val_recall'][best_epoch_idx]:.4f}")
    print(f"  F1:        {history['val_f1'][best_epoch_idx]:.4f}")
    print(f"  Kappa:     {history['val_kappa'][best_epoch_idx]:.4f}")
    print(f"  MCC:       {history['val_mcc'][best_epoch_idx]:.4f}")
    print(f"  Val Loss:  {history['val_loss'][best_epoch_idx]:.4f}")
    print(f"{'='*60}")

                                     
    try:
        plt.figure(figsize=(15, 12))

                                           
        cv_x = (converge_epoch - 1) if converge_epoch is not None else None

                               
        plt.subplot(2, 3, 1)
        plt.plot(history['train_loss'], label='Training Loss', linewidth=2, color='blue')
        plt.plot(history['val_loss'], label='Validation Loss()', linewidth=1.5,
                 color='red', linestyle='--', alpha=0.5)
        plt.plot(history['ema_val_loss'], label='Validation Loss(EMA)', linewidth=2,
                 color='darkred', linestyle='-')
        if cv_x is not None:
            plt.axvline(cv_x, color='gold', linewidth=1.5, linestyle='--',
                        label=f'Convergence point E{converge_epoch}')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=9)
        plt.title('Loss Curve', fontsize=14, fontproperties='SimHei')
        plt.grid(True, linestyle='--', alpha=0.7)

                         
        plt.subplot(2, 3, 2)
        plt.plot(history['train_acc'], label='Training ACC', linewidth=2, color='green')
        plt.plot(history['val_acc'], label='Validation ACC', linewidth=2,
                 color='limegreen', linestyle='--')
        plt.plot([v * 100 for v in history['val_f1']], label='Validation F1×100',
                 linewidth=2, color='darkorange', linestyle='-.')
        if cv_x is not None:
            plt.axvline(cv_x, color='gold', linewidth=1.5, linestyle='--',
                        label=f'Convergence point E{converge_epoch}')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Value (%)', fontsize=12)
        plt.legend(fontsize=9)
        plt.title('Accuracy & F1 ', fontsize=14, fontproperties='SimHei')
        plt.grid(True, linestyle='--', alpha=0.7)

                                              
        plt.subplot(2, 3, 3)
        plt.plot(history['val_precision'], label='Precision', linewidth=2, color='steelblue')
        plt.plot(history['val_recall'],    label='Recall',    linewidth=2, color='tomato')
        plt.plot(history['val_kappa'],     label='Kappa',     linewidth=1.5,
                 color='purple', linestyle='--')
        plt.plot(history['val_mcc'],       label='MCC',       linewidth=1.5,
                 color='sienna', linestyle=':')
        if cv_x is not None:
            plt.axvline(cv_x, color='gold', linewidth=1.5, linestyle='--',
                        label=f'Convergence point E{converge_epoch}')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('', fontsize=12)
        plt.ylim(-0.1, 1.1)
        plt.legend(fontsize=9)
        plt.title('Precision / Recall / Kappa / MCC', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

                   
        plt.subplot(2, 3, 4)
        plt.plot(history['learning_rate'], label='', linewidth=2, color='purple')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=10)
        plt.title('Learning Rate Schedule', fontsize=14, fontproperties='SimHei')
        plt.grid(True, linestyle='--', alpha=0.7)

                             
        plt.subplot(2, 3, 5)
        gap = [t - v for t, v in zip(history['train_loss'], history['ema_val_loss'])]
        bar_colors = ['red' if g > 0.05 else 'steelblue' for g in gap]
        plt.bar(range(len(gap)), gap, color=bar_colors, alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.axhline(0.05, color='red', linewidth=1, linestyle='--', label='')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Train Loss − EMA Val Loss', fontsize=10)
        plt.legend(fontsize=9)
        plt.title('', fontsize=14, fontproperties='SimHei')
        plt.grid(True, linestyle='--', alpha=0.7)

                                                   
        plt.subplot(2, 3, 6)
        plt.plot(history['composite'], label='Composite()', linewidth=2, color='teal')
                                           
        best_idx = int(np.argmin(history['ema_val_loss']))
        plt.scatter(best_idx, history['composite'][best_idx],
                    color='red', zorder=5, s=60, label=f'Loss Epoch {best_idx+1}')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('F1×100 − 0.1×EMA_Loss', fontsize=10)
        plt.legend(fontsize=9)
        plt.title('（Composite）', fontsize=14, fontproperties='SimHei')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.suptitle('ModelTraining History', fontsize=16, fontproperties='SimHei')
        plt.tight_layout()
        plt.savefig('attribution_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Training History 'attribution_training_history.png'")
    except Exception as e:
        print(f"Training History: {str(e)}")

                                     
    if converge_metrics is not None:
        print(f'\n{"="*60}')
        print(f'[Training SetConvergence point Epoch {converge_metrics["epoch"]}] Validation Set')
        print(f'{"="*60}')
        print(f'  ACC:       {converge_metrics["acc"]:.2f}%')
        print(f'  Precision: {converge_metrics["precision"]:.4f}')
        print(f'  Recall:    {converge_metrics["recall"]:.4f}')
        print(f'  F1:        {converge_metrics["f1"]:.4f}')
        print(f'  Kappa:     {converge_metrics["kappa"]:.4f}')
        print(f'  MCC:       {converge_metrics["mcc"]:.4f}')
        print(f'  Val Loss:  {converge_metrics["val_loss"]:.4f}')
        print(f'{"="*60}')
    else:
        print('\n[] Training SetAccuracy 100%，。')

                                                      
    try:
        save_obj = {'history': history}
        if converge_metrics is not None:
            save_obj['converge_metrics'] = converge_metrics
        with open('training_history.json', 'w', encoding='utf-8') as f:
            json.dump(save_obj, f, indent=2, ensure_ascii=False)
        print("Training History 'training_history.json'")
    except Exception as e:
        print(f"Training History JSON : {str(e)}")

    best_metrics = {
        'acc':             history['val_acc'][best_epoch_idx],
        'precision':       history['val_precision'][best_epoch_idx],
        'recall':          history['val_recall'][best_epoch_idx],
        'f1':              history['val_f1'][best_epoch_idx],
        'kappa':           history['val_kappa'][best_epoch_idx],
        'mcc':             history['val_mcc'][best_epoch_idx],
        'converge_metrics': converge_metrics,
        'val_loss':  history['val_loss'][best_epoch_idx],
    }
    return best_metrics


def main():
    print("="*60)
    print("Attribution-guided DCN-Transformer mineral prediction model")
    print("="*60)

                                
    train_data = np.load('D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/train_data.npy')
    train_labels = np.load('D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/train_labels.npy')
    val_data = np.load('D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/verify_data.npy')
    val_labels = np.load('D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/verify_labels.npy')

    if len(train_data.shape) != 4 or train_data.shape[1] != 9 or train_data.shape[2] != 9 or train_data.shape[3] != 42:
        raise ValueError(f"Training data，(n, 9, 9, 42)，{train_data.shape}")
    if len(val_data.shape) != 4 or val_data.shape[1] != 9 or val_data.shape[2] != 9 or val_data.shape[3] != 42:
        raise ValueError(f"Validation data，(n, 9, 9, 42)，{val_data.shape}")

    print("\n=== Dataset Information ===")
    print(f"Training data: {len(train_data)}")
    print(f"Validation data: {len(val_data)}")
    print(f"Data shape: {train_data.shape}")
    print(f": 39 + 3Ore-controlling factors = 42")

    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    print("\nTraining SetLabel Distribution:")
    for label, count in zip(train_unique, train_counts):
        print(f"{'Mineralized' if label == 1 else 'Non-mineralized'}: {count}  ({count/len(train_labels)*100:.2f}%)")

    val_unique, val_counts = np.unique(val_labels, return_counts=True)
    print("\nValidation SetLabel Distribution:")
    for label, count in zip(val_unique, val_counts):
        print(f"{'Mineralized' if label == 1 else 'Non-mineralized'}: {count}  ({count/len(val_labels)*100:.2f}%)")

                                
    num_classes = 2
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = torch.FloatTensor(len(train_data) / (num_classes * class_counts)).to(device)
    print("\nClass weights:")
    for i, weight in enumerate(class_weights):
        print(f"{'Mineralized' if i == 1 else 'Non-mineralized'} : {weight:.4f}")

                                 
    print("\nStandardizing data...")
    scaler = StandardScaler()
    n_samples = train_data.shape[0]
    n_channels = train_data.shape[3]

    train_data_reshaped = train_data.reshape(-1, n_channels)
    val_data_reshaped = val_data.reshape(-1, n_channels)

    train_data_reshaped = scaler.fit_transform(train_data_reshaped)
    val_data_reshaped = scaler.transform(val_data_reshaped)

    train_data = train_data_reshaped.reshape(n_samples, 9, 9, n_channels)
    val_data = val_data_reshaped.reshape(-1, 9, 9, n_channels)                 

                                                                
    joblib.dump(scaler, 'train_scaler.joblib')
    print("Scaler saved as 'train_scaler.joblib'")

                                                      
    train_data = torch.FloatTensor(train_data).permute(0, 3, 1, 2).to(device)
    train_labels = torch.LongTensor(train_labels).to(device)
    val_data = torch.FloatTensor(val_data).permute(0, 3, 1, 2).to(device)
    val_labels = torch.LongTensor(val_labels).to(device)

    print(f"\nData shape:")
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

                                 
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    train_size = len(train_dataset)
    if train_size < 500:
        batch_size = 16
        print(f"[Warning] ({train_size})，batch size{batch_size}")
    elif train_size < 2000:
        batch_size = 32
        print(f"[Warning] ({train_size})，batch size{batch_size}")
    else:
        batch_size = 96

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                                
    print("\n=== Creating attribution-guided model ===")

                                                         
    manual_config = "ultra_light"

    if manual_config == "ultra_light":
        feature_dim = 32
        transformer_depth = 1
        transformer_heads = 2
        config_name = "Ultra-light configuration (optimized v2.0)"
        expected_params = "200K-300K"
        print(f"[]  (: ~{expected_params})")
        print(f"  d_model=32, depth=1, heads=2")
        print(f"  DCNv2=3, attribution network=3, Dropout=0.15")

    elif manual_config == "mini":
        feature_dim = 16
        transformer_depth = 1
        transformer_heads = 2
        config_name = ""
        expected_params = "8K-12K"
        print(f"[]  (: ~{expected_params})")

    elif manual_config == "medium":
        feature_dim = 64
        transformer_depth = 2
        transformer_heads = 4
        config_name = "（ v2.0）"
        expected_params = "400K-600K"
        print(f"[]  (: ~{expected_params})")
        print(f"  d_model=64, depth=2, heads=4")
        print(f"  DCNv2=3, attribution network=3, Dropout=0.15")

    elif manual_config == "nano":
        feature_dim = 8
        transformer_depth = 1
        transformer_heads = 1
        config_name = ""
        expected_params = "3K-5K"
        print(f"[]  (: ~{expected_params})")

    else:        
        if train_size < 500:
            feature_dim = 16
            transformer_depth = 1
            transformer_heads = 2
            config_name = "（）"
            expected_params = "8K-12K"
            print(f"[Warning] ({train_size})，Model")
        elif train_size < 1000:
            feature_dim = 32
            transformer_depth = 1
            transformer_heads = 2
            config_name = "（）"
            expected_params = "20K-30K"
            print(f"[Warning] ({train_size})，Model")
        elif train_size < 5000:
            feature_dim = 64
            transformer_depth = 2
            transformer_heads = 4
            config_name = "（）"
            expected_params = "300K-500K"
            print(f"[Warning] ({train_size})，Model")
        else:
            feature_dim = 128
            transformer_depth = 4
            transformer_heads = 8
            config_name = "（）"
            expected_params = "5M+"
            print(f"[OK] ({train_size} samples), using the high-capacity model")

    model = create_attribution_guided_model(
        num_classes=2,
        feature_dim=feature_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads
    ).to(device)

    print(f"\nModel: {config_name}")
    print(f"  - Feature Dim: {feature_dim}")
    print(f"  - Transformer Depth: {transformer_depth}")
    print(f"  - Transformer Heads: {transformer_heads}")
    print(f"  - : {expected_params}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel:")
    print(f": {total_params:,}")
    print(f": {trainable_params:,}")

                                           
    print("\n=== attribution network（Training Set）===")
    model.attribution_network.update_reference_features(train_loader, device)

                              
    print("\n=== Starting training ===")
    best_metrics = train(
        model,
        train_loader,
        val_loader,
        num_epochs=200,
        class_weights=class_weights
    )

    print(f"\n！Model:")
    print(f"  ACC:       {best_metrics['acc']:.2f}%")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  F1:        {best_metrics['f1']:.4f}")
    print(f"  Kappa:     {best_metrics['kappa']:.4f}")
    print(f"  MCC:       {best_metrics['mcc']:.4f}")

                                    
    print("\n=== Loading best model ===")
                                                       
                                               
    checkpoint = torch.load('best_attribution_guided_model.pth',
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

                                
    _run_integrated_prediction(
        model, device, scaler,
        feature_dim=feature_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads
    )


def _run_integrated_prediction(model, device, train_scaler,
                                feature_dim, transformer_depth, transformer_heads):
    PREDICT_DATA_PATH = 'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/Combined_array.npy'
    XX_PATH = 'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/New_data/coordinate/XX.tif'
    YY_PATH = 'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/New_data/coordinate/YY.tif'
    OUTPUT_DIR = 'prediction_results'
    BATCH_SIZE = 96
    CONFIDENCE_THRESHOLD = 0.5
    WINDOW_SIZE = 9

    print("\n" + "="*70)
    print("Integrated prediction pipeline: run mineral prospectivity prediction on the full region")
    print("="*70)

                                 
    if not os.path.exists(PREDICT_DATA_PATH):
        print(f"[] : {PREDICT_DATA_PATH}")
        print("      ， predict_attribution_model.py")
        return

    print(f"\n: {PREDICT_DATA_PATH}")
    try:
        data = np.load(PREDICT_DATA_PATH)
    except Exception as e:
        print(f"[] : {e}")
        return

    print(f"  Data shape: {data.shape}  |  : {data.dtype}")
    print(f"  : [{data.min():.2f}, {data.max():.2f}]")

                                          
    print("\nTraining Set scaler ...")
    h, w, c = data.shape
    data_flat = data.reshape(-1, c)
    data_flat = train_scaler.transform(data_flat)
    data = data_flat.reshape(h, w, c)
    print("  Standardization complete")

                                 
    coords = [(i, j) for i in range(h - WINDOW_SIZE + 1)
                      for j in range(w - WINDOW_SIZE + 1)]
    num_patches = len(coords)
    probabilities = np.zeros((h, w))

    print(f"\n（ {WINDOW_SIZE}×{WINDOW_SIZE}， {num_patches} ）...")
    model.eval()
    with torch.no_grad():
        for k in tqdm(range(0, num_patches, BATCH_SIZE), desc="Prediction progress", unit="batch"):
            batch_coords = coords[k: k + BATCH_SIZE]
            patches = np.array([data[i:i+WINDOW_SIZE, j:j+WINDOW_SIZE, :]
                                 for i, j in batch_coords])
            batch_tensor = torch.FloatTensor(patches).permute(0, 3, 1, 2).to(device)

            output = model(batch_tensor)
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()

            for idx, (i, j) in enumerate(batch_coords):
                probabilities[i + WINDOW_SIZE // 2, j + WINDOW_SIZE // 2] = probs[idx]

    print("  Prediction complete")

                               
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prob_npy = os.path.join(OUTPUT_DIR, 'probabilities.npy')
    np.save(prob_npy, probabilities)
    print(f"\n[]  → {prob_npy}")

              
    try:
        XX = np.array(Image.open(XX_PATH)) if os.path.exists(XX_PATH) else None
        YY = np.array(Image.open(YY_PATH)) if os.path.exists(YY_PATH) else None
    except Exception:
        XX, YY = None, None

    if XX is not None and YY is not None and XX.shape == (h, w):
        df = pd.DataFrame({'X': XX.flatten(), 'Y': YY.flatten(),
                           'Probability': probabilities.flatten()})
    else:
        df = pd.DataFrame({'Row': np.repeat(np.arange(h), w),
                           'Col': np.tile(np.arange(w), h),
                           'Probability': probabilities.flatten()})
    csv_path = os.path.join(OUTPUT_DIR, 'prediction_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[]  CSV  → {csv_path}")

                             
    total_params = sum(p.numel() for p in model.parameters())
    ref_init = model.attribution_network.reference_initialized.item()
    config_info = {
        'model_architecture': 'Attribution Guided DCN-Transformer v3.0',
        'feature_dim': feature_dim,
        'transformer_depth': transformer_depth,
        'transformer_heads': transformer_heads,
        'dcnv2_layers': 3,
        'attribution_network_layers': 3,
        'dropout': 0.15,
        'total_parameters': total_params,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'window_size': WINDOW_SIZE,
        'architecture_fixes': {
            'encoder_redundant_ffn': '',
            'lr_scheduler': 'CosineAnnealingWarmRestarts',
            'attribution_baseline': 'Training Set' if ref_init else '（）'
        }
    }
    json_path = os.path.join(OUTPUT_DIR, 'model_config.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"[] Model → {json_path}")

                              
    print("\n...")

    plt.figure(figsize=(12, 10))
    im = plt.imshow(probabilities, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, label='Prediction Probability（Mineralized）')
    plt.title('', fontsize=16)
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, 'probability_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  []  → {heatmap_path}")

    plt.figure(figsize=(10, 6))
    plt.hist(probabilities.flatten(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f' = {CONFIDENCE_THRESHOLD}')
    plt.xlabel('Prediction Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Probability', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    dist_path = os.path.join(OUTPUT_DIR, 'probability_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  []  → {dist_path}")

    plt.figure(figsize=(12, 10))
    binary = (probabilities > CONFIDENCE_THRESHOLD).astype(int)
    plt.imshow(binary, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='Predicted Class (0=non-mineral, 1=mineral)', ticks=[0, 1])
    pos = binary.sum()
    pct = pos / binary.size * 100
    plt.text(0.02, 0.98,
             f'Mineral: {pos} ({pct:.2f}%)\nNon-mineral: {binary.size - pos} ({100-pct:.2f}%)',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.title(f'（Threshold={CONFIDENCE_THRESHOLD}）', fontsize=16)
    plt.tight_layout()
    bin_path = os.path.join(OUTPUT_DIR, 'binary_prediction.png')
    plt.savefig(bin_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  []  → {bin_path}")

    plt.figure(figsize=(14, 12))
    plt.imshow(probabilities, cmap='gray')
    for mask, color, label, size in [
        (probabilities > 0.9,                              'darkred', 'Very high prospectivity (>0.9)',     20),
        ((probabilities > 0.7) & (probabilities <= 0.9),  'red',     'High prospectivity (0.7-0.9)',   15),
        ((probabilities > 0.5) & (probabilities <= 0.7),  'yellow',  'Moderate prospectivity (0.5-0.7)',  8),
    ]:
        ys, xs = np.where(mask)
        if len(ys):
            plt.scatter(xs, ys, c=color, s=size, alpha=0.7, label=label)
    plt.colorbar(label='Prediction Probability')
    plt.title('', fontsize=16)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    zone_path = os.path.join(OUTPUT_DIR, 'potential_zones.png')
    plt.savefig(zone_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  []  → {zone_path}")

                               
    total = probabilities.size
    print(f"\n{'='*60}")
    print("")
    print(f"{'='*60}")
    print(f"  : {total}")
    print(f"  Mean probability: {probabilities.mean():.4f}  |  Standard deviation: {probabilities.std():.4f}")
    for thresh, label in [(0.9, ''), (0.7, ''), (0.5, ''), (0.3, '')]:
        cnt = int((probabilities > thresh).sum())
        print(f"  >{thresh} {label}: {cnt:8d} ({cnt/total*100:6.2f}%)")
    print(f"{'='*60}")
    print(f"\n: {OUTPUT_DIR}/")
    print("="*70)


if __name__ == '__main__':
    main()
