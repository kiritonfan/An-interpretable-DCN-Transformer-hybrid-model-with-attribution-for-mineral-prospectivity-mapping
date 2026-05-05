"""This script runs the Transformer-only ablation experiment for the study."""

import os
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, cohen_kappa_score, matthews_corrcoef)
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

                                              
from Dense_transformers import DTransformer, Encoder              

                          
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

                                                              
                        
                                                              
FEATURE_DIM      = 32                               
TRANSFORMER_DEPTH= 1
TRANSFORMER_HEADS= 2
NUM_CLASSES      = 2
NUM_EPOCHS       = 200
LEARNING_RATE    = 5e-5
WEIGHT_DECAY     = 0.01
WARMUP_EPOCHS    = 5
MAX_NO_IMPROVE   = 50
GRAD_CLIP        = 1.0
DROPOUT          = 0.15                                  
MODEL_SAVE_PATH  = 'best_transformer_only_model.pth'
OUTPUT_DIR       = 'prediction_results_transformer_only'

      
TRAIN_DATA_PATH  = 'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/train_data.npy'
TRAIN_LABEL_PATH = 'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/train_labels.npy'
VAL_DATA_PATH    = 'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/verify_data.npy'
VAL_LABEL_PATH   = 'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/verify_labels.npy'
PREDICT_DATA_PATH= 'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/Combined_array.npy'
XX_PATH          = 'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/New_data/coordinate/XX.tif'
YY_PATH          = 'D:/shiyan/To-Hao Yunfei/HSI_transformer_demo/nanling/New_data/coordinate/YY.tif'


                                                              
                      
                                                              
class TransformerOnlyModel(nn.Module):
    def __init__(self, num_classes=2, feature_dim=FEATURE_DIM,
                 transformer_depth=TRANSFORMER_DEPTH,
                 transformer_heads=TRANSFORMER_HEADS):
        super(TransformerOnlyModel, self).__init__()

                                                   
                                                
        self.input_projection = nn.Sequential(
            nn.Conv2d(42, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

                                             
        encoder = Encoder(
            dim=feature_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=feature_dim // transformer_heads,
            mlp_dim=feature_dim * 2,                                  
            dropout=DROPOUT
        )

                                          
        self.transformer = DTransformer(
            image_size=9,
            patch_size=1,
            attn_layers=encoder,
            num_classes=num_classes,
            dropout=DROPOUT
        )

        self._initialize_weights()

        print(f"[OK] Transformer-only model created")
        print(f"  Input projection: Conv2d(42, {feature_dim}, 3x3) + BN + ReLU")
        print(f"  Encoder: dim={feature_dim}, depth={transformer_depth}, heads={transformer_heads}")
        print(f"  mlp_dim={feature_dim*2}, dropout={DROPOUT}")
        print(f"  DTransformer: image_size=9, patch_size=1, cls_token + pos_embedding")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.input_projection(x)                           
        output   = self.transformer(features)                   
        return output

    def forward_with_attention(self, x):
        features = self.input_projection(x)
        output, attention_maps = self.transformer(features, return_attention=True)
        return output, attention_maps


                                                              
       
                                                              
def train(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * 2, eta_min=1e-7
    )

    best_val_acc  = 0.0
    best_val_loss = float('inf')
    no_improve    = 0
    history       = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    base_lr       = LEARNING_RATE

    print(f"\n[Transformer-only] Starting training for {num_epochs} epochs")
    print(f"  Learning rate: {LEARNING_RATE}  |  Weight decay: {WEIGHT_DECAY}  |  Early stopping patience: {MAX_NO_IMPROVE}")

    for epoch in range(num_epochs):
                
        if epoch < WARMUP_EPOCHS:
            factor = (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * factor

                        
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss   = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            t_loss   += loss.item()
            _, pred   = output.max(1)
            t_total  += target.size(0)
            t_correct += pred.eq(target).sum().item()

        avg_t_loss = t_loss / len(train_loader)
        t_acc      = 100. * t_correct / t_total

                        
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                v_loss   += criterion(output, target).item()
                _, pred   = output.max(1)
                v_total  += target.size(0)
                v_correct += pred.eq(target).sum().item()

        avg_v_loss = v_loss / len(val_loader)
        v_acc      = 100. * v_correct / v_total
        cur_lr     = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(avg_v_loss)
        history['val_acc'].append(v_acc)
        history['lr'].append(cur_lr)

        print(f"Epoch [{epoch+1:3d}/{num_epochs}]  "
              f"TrainLoss: {avg_t_loss:.4f}  TrainAcc: {t_acc:.2f}%  "
              f"ValLoss: {avg_v_loss:.4f}  ValAcc: {v_acc:.2f}%  "
              f"LR: {cur_lr:.2e}")

        scheduler.step()

        if v_acc > best_val_acc:
            best_val_acc  = v_acc
            best_val_loss = avg_v_loss
            no_improve    = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': v_acc,
                'val_loss': avg_v_loss,
                'history': history
            }, MODEL_SAVE_PATH)
            print(f"  >> Saved best model  Val Acc: {v_acc:.2f}%  Val Loss: {avg_v_loss:.4f}")
        else:
            no_improve += 1
            if no_improve >= MAX_NO_IMPROVE:
                print(f"  >> {MAX_NO_IMPROVE} epochs without improvement, stopping early")
                break

    _plot_history(history, save_path='transformer_only_training_history.png',
                  title='Transformer-only Model Training History')

    print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%  Best Val Loss: {best_val_loss:.4f}")
    return best_val_acc, best_val_loss


def _plot_history(history, save_path, title='Training History'):
    try:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Training Loss', color='blue')
        plt.plot(history['val_loss'],   label='Validation Loss', color='red', linestyle='--')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(); plt.grid(True, alpha=0.4)
        plt.title('Loss Curve')

        plt.subplot(1, 3, 2)
        plt.plot(history['train_acc'], label='Training Accuracy', color='green')
        plt.plot(history['val_acc'],   label='Validation Accuracy', color='orange', linestyle='--')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
        plt.legend(); plt.grid(True, alpha=0.4)
        plt.title('Accuracy Curve')

        plt.subplot(1, 3, 3)
        plt.plot(history['lr'], color='purple')
        plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
        plt.yscale('log'); plt.grid(True, alpha=0.4)
        plt.title('Learning Rate Schedule')

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curve saved: {save_path}")
    except Exception as e:
        print(f"Failed to plot training curve: {e}")


                                                              
            
                                                              
def compute_val_metrics(model, val_loader, model_name='Model'):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = output.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc       = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0) * 100
    recall    = recall_score(y_true, y_pred, average='binary', zero_division=0) * 100
    f1        = f1_score(y_true, y_pred, average='binary', zero_division=0) * 100
    kappa     = cohen_kappa_score(y_true, y_pred)
    mcc       = matthews_corrcoef(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f"Validation Metrics - {model_name}")
    print(f"{'='*55}")
    print(f"  ACC       (Accuracy):         {acc:.4f}%")
    print(f"  Precision (Precision):         {precision:.4f}%")
    print(f"  Recall    (Recall):         {recall:.4f}%")
    print(f"  F1 Score:                   {f1:.4f}%")
    print(f"  Kappa     (Kappa coefficient):      {kappa:.6f}")
    print(f"  MCC       (Matthews correlation coefficient): {mcc:.6f}")
    print(f"{'='*55}\n")

    return {'ACC': acc, 'Precision': precision, 'Recall': recall,
            'F1': f1, 'Kappa': kappa, 'MCC': mcc}


                                                              
      
                                                              
class TransformerOnlyPredictor:

    def __init__(self, model_path, feature_dim=FEATURE_DIM,
                 transformer_depth=TRANSFORMER_DEPTH,
                 transformer_heads=TRANSFORMER_HEADS,
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device in Use: {self.device}")

        print("Loading the Transformer-only model...")
        self.model, self.checkpoint = self._load_model(
            model_path, feature_dim, transformer_depth, transformer_heads
        )
        self.model.eval()
        print(f"  Training epoch: Epoch {self.checkpoint.get('epoch', 'N/A')}")
        print(f"  Validation accuracy: {self.checkpoint.get('val_acc', 0):.2f}%")
        print(f"  Validation Loss: {self.checkpoint.get('val_loss', 0):.4f}")
        print("Model loaded successfully\n")

    def _load_model(self, model_path, feature_dim, transformer_depth, transformer_heads):
        model = TransformerOnlyModel(
            num_classes=NUM_CLASSES,
            feature_dim=feature_dim,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads
        ).to(self.device)

        total = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total:,}")

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint

    def predict_batch(self, data, batch_size=96, show_progress=True):
        window_size = 9
        h, w, c = data.shape

        scaler = StandardScaler()
        data   = scaler.fit_transform(data.reshape(-1, c)).reshape(h, w, c)

        probabilities = np.zeros((h, w))
        coords = [(i, j) for i in range(h - window_size + 1)
                         for j in range(w - window_size + 1)]

        print(f"Total windows to predict: {len(coords)} windows")

        with torch.no_grad():
            itr = range(0, len(coords), batch_size)
            if show_progress:
                itr = tqdm(itr, desc="Transformer-Only Prediction progress", unit="batch")

            for k in itr:
                bc = coords[k:k + batch_size]
                patches = np.array([data[i:i+window_size, j:j+window_size, :]
                                    for i, j in bc])
                tensor  = torch.FloatTensor(patches).permute(0, 3, 1, 2).to(self.device)
                output  = self.model(tensor)
                probs   = torch.softmax(output, dim=1)[:, 1].cpu().numpy()

                for idx, (i, j) in enumerate(bc):
                    probabilities[i + window_size // 2, j + window_size // 2] = probs[idx]

        print("Prediction completed\n")
        return probabilities

    def predict_and_save(self, data, output_dir=OUTPUT_DIR,
                         xx_path=None, yy_path=None,
                         batch_size=96, confidence_threshold=0.5):
        os.makedirs(output_dir, exist_ok=True)

        total = sum(p.numel() for p in self.model.parameters())
        config = {
            'model_architecture': 'Transformer-Only (Ablation)',
            'feature_dim': FEATURE_DIM,
            'transformer_depth': TRANSFORMER_DEPTH,
            'transformer_heads': TRANSFORMER_HEADS,
            'mlp_dim': FEATURE_DIM * 2,
            'dropout': DROPOUT,
            'total_parameters': total,
            'device': str(self.device),
            'batch_size': batch_size,
            'confidence_threshold': confidence_threshold,
            'window_size': 9,
            'input_channels': data.shape[2],
            'note': 'The input projection layer replaces DCNv2, while the Transformer structure is identical to the hybrid model.'
        }
        with open(os.path.join(output_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        probabilities = self.predict_batch(data, batch_size=batch_size)
        h, w = probabilities.shape

        np.save(os.path.join(output_dir, 'probabilities.npy'), probabilities)

        if xx_path and yy_path:
            try:
                XX = np.array(Image.open(xx_path))
                YY = np.array(Image.open(yy_path))
                if XX.shape == (h, w):
                    df = pd.DataFrame({'X': XX.flatten(), 'Y': YY.flatten(),
                                       'Probability': probabilities.flatten()})
                else:
                    raise ValueError("Coordinate dimensions do not match")
            except Exception as e:
                print(f"Failed to load coordinates ({e}), using row/column indices")
                df = pd.DataFrame({'Row': np.repeat(np.arange(h), w),
                                   'Col': np.tile(np.arange(w), h),
                                   'Probability': probabilities.flatten()})
        else:
            df = pd.DataFrame({'Row': np.repeat(np.arange(h), w),
                               'Col': np.tile(np.arange(w), h),
                               'Probability': probabilities.flatten()})

        df.to_csv(os.path.join(output_dir, 'prediction_results.csv'), index=False)
        print(f"CSV saved: {output_dir}/prediction_results.csv")

        _print_statistics(probabilities, confidence_threshold)
        _visualize_results(probabilities, output_dir, confidence_threshold,
                           title_suffix='(Transformer-only)')

        return probabilities, df


                                                              
                
                                                              
def _print_statistics(probabilities, threshold=0.5):
    total = probabilities.size
    print("\n" + "="*55)
    print("Prediction Statistics - Transformer-only Model")
    print("="*55)
    print(f"Total pixels: {total}")
    print(f"Mean probability: {probabilities.mean():.4f}  |  Max: {probabilities.max():.4f}  |  Min: {probabilities.min():.4f}")
    print(f"Standard deviation: {probabilities.std():.4f}  |  Median: {np.median(probabilities):.4f}")
    print("\nStatistics at Different Thresholds:")
    for t in [0.3, 0.5, 0.7, 0.9]:
        cnt  = np.sum(probabilities > t)
        mark = " <-- Current threshold" if t == threshold else ""
        print(f"  > {t}: {cnt:8d} ({cnt/total*100:6.2f}%){mark}")
    print("\nMineral Prospectivity Classification:")
    print(f"  Very high (>0.9):      {np.sum(probabilities > 0.9):8d} ({np.sum(probabilities > 0.9)/total*100:.2f}%)")
    print(f"  High (0.7-0.9):     {np.sum((probabilities>0.7)&(probabilities<=0.9)):8d}")
    print(f"  Medium (0.5-0.7):     {np.sum((probabilities>0.5)&(probabilities<=0.7)):8d}")
    print(f"  Low (0.3-0.5):     {np.sum((probabilities>0.3)&(probabilities<=0.5)):8d}")
    print(f"  Very low (<=0.3):     {np.sum(probabilities<=0.3):8d}")
    print("="*55 + "\n")


def _visualize_results(probabilities, output_dir, threshold=0.5, title_suffix=''):
    print("Generating visualizations...")

    plt.figure(figsize=(12, 10))
    im = plt.imshow(probabilities, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, label='Prediction Probability')
    plt.title(f'Mineral Prospectivity Heatmap {title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(probabilities.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    plt.xlabel('Prediction Probability'); plt.ylabel('Frequency')
    plt.title(f'Prediction Probability Distribution {title_suffix}')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 10))
    binary  = (probabilities > threshold).astype(int)
    plt.imshow(binary, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='Predicted Class (0=non-mineral, 1=mineral)', ticks=[0, 1])
    pos_cnt = np.sum(binary)
    plt.text(0.02, 0.98, f'Mineral: {pos_cnt} ({pos_cnt/binary.size*100:.2f}%)\nNon-mineral: {binary.size-pos_cnt} ({(1-pos_cnt/binary.size)*100:.2f}%)',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.title(f'Binarized Prediction Result (threshold={threshold}) {title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binary_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(14, 12))
    plt.imshow(probabilities, cmap='gray')
    for mask, color, label in [
        (probabilities > 0.9, 'darkred', 'Very high prospectivity (>0.9)'),
        ((probabilities > 0.7) & (probabilities <= 0.9), 'red', 'High prospectivity (0.7-0.9)'),
        ((probabilities > 0.5) & (probabilities <= 0.7), 'yellow', 'Moderate prospectivity (0.5-0.7)')
    ]:
        idx = np.where(mask)
        if len(idx[0]) > 0:
            plt.scatter(idx[1], idx[0], c=color, s=15, alpha=0.7, label=label)
    plt.colorbar(label='Prediction Probability')
    plt.title(f'Mineral Prospectivity Zones {title_suffix}', fontsize=16)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'potential_zones.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All visualizations saved to: {output_dir}\n")


                                                              
      
                                                              
def main():
    print("=" * 60)
    print("Ablation Study - Transformer-only Model Mineral Prediction")
    print("=" * 60)

                                     
    print("\nLoading data...")
    train_data   = np.load(TRAIN_DATA_PATH)
    train_labels = np.load(TRAIN_LABEL_PATH)
    val_data     = np.load(VAL_DATA_PATH)
    val_labels   = np.load(VAL_LABEL_PATH)

    for name, arr in [('Training data', train_data), ('Validation data', val_data)]:
        if arr.ndim != 4 or arr.shape[1:] != (9, 9, 42):
            raise ValueError(f"{name}Invalid shape, expected (n,9,9,42), got {arr.shape}")

    print(f"Training samples: {len(train_data)}  Validation samples: {len(val_data)}")

    for split, labels in [('Training Set', train_labels), ('Validation Set', val_labels)]:
        print(f"\n{split}Label Distribution:")
        for lbl, cnt in zip(*np.unique(labels, return_counts=True)):
            name = 'Mineralized' if lbl == 1 else 'Non-mineralized'
            print(f"  {name}: {cnt} ({cnt/len(labels)*100:.2f}%)")

    class_counts  = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = torch.FloatTensor(len(train_data) / (NUM_CLASSES * class_counts)).to(device)
    print(f"\nClass weights: Non-mineral={class_weights[0]:.4f} , Mineral={class_weights[1]:.4f}")

                                 
    scaler    = StandardScaler()
    n, c      = train_data.shape[0], train_data.shape[3]
    train_data = scaler.fit_transform(train_data.reshape(-1, c)).reshape(n, 9, 9, c)
    nv        = val_data.shape[0]
    val_data  = scaler.transform(val_data.reshape(-1, c)).reshape(nv, 9, 9, c)

                                                  
    train_data   = torch.FloatTensor(train_data).permute(0, 3, 1, 2).to(device)
    train_labels = torch.LongTensor(train_labels).to(device)
    val_data     = torch.FloatTensor(val_data).permute(0, 3, 1, 2).to(device)
    val_labels   = torch.LongTensor(val_labels).to(device)

    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

                                      
    train_size = len(train_data)
    if train_size < 500:
        batch_size = 16
    elif train_size < 2000:
        batch_size = 32
    else:
        batch_size = 96

    print(f"Batch size: {batch_size}")

    train_loader = DataLoader(TensorDataset(train_data, train_labels),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_data, val_labels),
                              batch_size=batch_size, shuffle=False)

                                
    print("\n=== Creating Transformer-only model ===")
    model = TransformerOnlyModel(
        num_classes=NUM_CLASSES,
        feature_dim=FEATURE_DIM,
        transformer_depth=TRANSFORMER_DEPTH,
        transformer_heads=TRANSFORMER_HEADS
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

                              
    print("\n=== Starting training ===")
    best_acc, best_loss = train(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        class_weights=class_weights
    )

                                           
    print("\n=== Validation SetEvaluation ===")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    metrics = compute_val_metrics(model, val_loader, model_name='Transformer-onlyModel')

                                
    print("\n=== Starting prediction ===")
    print("Loading prediction data...")
    try:
        predict_data = np.load(PREDICT_DATA_PATH)
        print(f"Prediction data shape: {predict_data.shape}")
    except Exception as e:
        print(f"Failed to load prediction data: {e}")
        return

    predictor = TransformerOnlyPredictor(
        model_path=MODEL_SAVE_PATH,
        feature_dim=FEATURE_DIM,
        transformer_depth=TRANSFORMER_DEPTH,
        transformer_heads=TRANSFORMER_HEADS,
        device=device
    )

    probabilities, results_df = predictor.predict_and_save(
        predict_data,
        output_dir=OUTPUT_DIR,
        xx_path=XX_PATH,
        yy_path=YY_PATH,
        batch_size=96
    )

    print("=" * 60)
    print("Ablation Study - Transformer-only Model completed!")
    print("=" * 60)
    print(f"\nTraining Results: Best Val Acc = {best_acc:.2f}%  Val Loss = {best_loss:.4f}")
    print(f"Validation Metrics: ACC={metrics['ACC']:.4f}%  Precision={metrics['Precision']:.4f}%  "
          f"Recall={metrics['Recall']:.4f}%  F1={metrics['F1']:.4f}%  "
          f"Kappa={metrics['Kappa']:.6f}  MCC={metrics['MCC']:.6f}")
    print(f"Prediction results saved to: {OUTPUT_DIR}/")
    print(f"  - model_config.json")
    print(f"  - probabilities.npy")
    print(f"  - prediction_results.csv")
    print(f"  - probability_heatmap.png")
    print(f"  - probability_distribution.png")
    print(f"  - binary_prediction.png")
    print(f"  - potential_zones.png")
    print(f"Training curve: transformer_only_training_history.png")
    print(f"Model weights: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
