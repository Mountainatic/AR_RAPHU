"""Stage1Trainer: training loop with proximal group lasso and early stopping."""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np, os, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from .losses import total_loss
from .proximal import apply_group_proximal_step

def calculate_metrics(pred, target):
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse); mae = torch.mean(torch.abs(pred - target))
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return rmse.item(), mae.item(), r2.item()

class Stage1Trainer:
    def __init__(self, model, train_loader, val_loader, lr=0.001, weight_decay=1e-5,
                 lambda_group=0.01, lambda_smooth=0.0, max_epochs=100, patience=15,
                 device=None, run_dir=None, seed=42):
        self.model = model; self.train_loader = train_loader; self.val_loader = val_loader
        self.lambda_group = lambda_group; self.lambda_smooth = lambda_smooth
        self.max_epochs = max_epochs; self.patience = patience
        self.device = device or torch.device('cpu')
        self.run_dir = run_dir or 'results_stage1/default'; self.seed = seed
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        os.makedirs(self.run_dir, exist_ok=True)

    def train(self):
        model = self.model; device = self.device
        history = {'train_loss': [], 'val_rmse': [], 'val_mae': [], 'val_r2': []}
        metrics_log = []; best_val_rmse = float('inf'); best_epoch_info = {}; patience_counter = 0
        for epoch in range(self.max_epochs):
            model.train(); batch_losses = []
            for bx, by in self.train_loader:
                bx, by = bx.to(device), by.to(device); self.optimizer.zero_grad()
                pred, aux = model(bx, return_aux=True)
                loss, mse, smooth = total_loss(pred, by, model, self.lambda_smooth)
                loss.backward(); self.optimizer.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.lambda_group > 0:
                    apply_group_proximal_step(model.response_branches, current_lr, self.lambda_group)
                batch_losses.append(loss.item())
            model.eval(); val_preds, val_targets = [], []
            with torch.no_grad():
                for vx, vy in self.val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    pred, _ = model(vx, return_aux=True)
                    val_preds.append(pred); val_targets.append(vy)
            val_preds = torch.cat(val_preds, dim=0); val_targets = torch.cat(val_targets, dim=0)
            rmse, mae, r2 = calculate_metrics(val_preds, val_targets)
            avg_train_loss = np.mean(batch_losses)
            history['train_loss'].append(avg_train_loss); history['val_rmse'].append(rmse)
            history['val_mae'].append(mae); history['val_r2'].append(r2)
            metrics_log.append({'Epoch': epoch+1, 'Train_Loss': avg_train_loss,
                'Val_RMSE': rmse, 'Val_MAE': mae, 'Val_R2': r2})
            if rmse < best_val_rmse:
                best_val_rmse = rmse; best_epoch_info = metrics_log[-1]; patience_counter = 0
                torch.save(model.state_dict(), os.path.join(self.run_dir, 'best_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= self.patience: break
            if (epoch + 1) % 10 == 0:
                active = model.get_active_variables().sum().item()
                print(f'Epoch {epoch+1}/{self.max_epochs} | Loss: {avg_train_loss:.4f} | Val RMSE: {rmse:.4f} | R2: {r2:.4f} | Active: {active}/{model.num_variables}')
        torch.save(model.state_dict(), os.path.join(self.run_dir, 'last_model.pt'))
        pd.DataFrame(metrics_log).to_csv(os.path.join(self.run_dir, 'fold_metrics.csv'), index=False)
        best_epoch = best_epoch_info.get('Epoch', 0)
        print(f'Best at epoch {best_epoch}: RMSE={best_epoch_info.get("Val_RMSE",0):.4f}, R2={best_epoch_info.get("Val_R2",0):.4f}')
        return history, best_epoch_info, metrics_log
