"""Analysis and visualization for Stage1TargetDelayKAN (Section 7 of prompt1.md)."""
import torch, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def _to_numpy(x):
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.array(x)

class Stage1Analyzer:
    def __init__(self, model, var_names=None, save_dir='results_stage1/default'):
        self.model = model; self.var_names = var_names
        self.save_dir = save_dir; os.makedirs(save_dir, exist_ok=True)

    def plot_all(self, aux, preds, targets):
        self.plot_prior_posterior_delay(aux)
        self.plot_branch_norms(aux)
        self.plot_response_curves()
        self.plot_contribution_heatmap(aux)
        self.plot_predictions(preds, targets)

    def plot_prior_posterior_delay(self, aux):
        pi = _to_numpy(aux['pi']); q = _to_numpy(aux['q'])
        if pi.ndim == 3: pi = pi.mean(axis=0)
        if q.ndim == 3: q_mean = q.mean(axis=0)
        else: q_mean = q
        N, L = pi.shape
        n_cols = min(4, N); n_rows = (N + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows * n_cols == 1: axes = [axes]
        else: axes = axes.flatten()
        for j in range(N):
            ax = axes[j]
            ax.plot(range(L), pi[j], 'b-', lw=2, label='Prior pi')
            ax.plot(range(L), q_mean[j], 'r--', lw=1.5, label='Mean post q')
            name = self.var_names[j] if self.var_names else f'Var{j}'
            ax.set_title(name); ax.set_xlabel('Lag tau'); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        for j in range(N, len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, 'delay_prior_posterior.png')); plt.close()

    def plot_branch_norms(self, aux):
        norms = _to_numpy(aux['branch_norm']).flatten()
        N = len(norms); names = self.var_names if self.var_names else [f'Var{j}' for j in range(N)]
        fig, ax = plt.subplots(figsize=(max(8,N*0.6), 5))
        colors = ['green' if n > 1e-6 else 'gray' for n in norms]
        ax.bar(range(N), norms, color=colors)
        ax.axhline(y=1e-6, color='r', linestyle='--', label='Threshold')
        ax.set_xticks(range(N)); ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_title('Variable Branch Norms'); ax.set_yscale('log'); ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, 'branch_norms.png')); plt.close()

    def plot_response_curves(self):
        """Sample KAN response curves by evaluating branches on a grid."""
        N = self.model.num_variables
        n_cols = min(4, N); n_rows = (N + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows * n_cols == 1: axes = [axes]
        else: axes = axes.flatten()
        device = next(self.model.parameters()).device
        x_grid = torch.linspace(-3, 3, 200).view(-1, 1).to(device)
        for j in range(N):
            with torch.no_grad():
                y = self.model.response_branches.branches[j](x_grid).cpu().numpy().flatten()
            name = self.var_names[j] if self.var_names else f'Var{j}'
            axes[j].plot(x_grid.cpu().numpy().flatten(), y, lw=1.5)
            axes[j].set_title(name); axes[j].axhline(y=0, color='gray', ls='--', alpha=0.5)
            axes[j].grid(True, alpha=0.3)
        for j in range(N, len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, 'response_functions.png')); plt.close()

    def plot_contribution_heatmap(self, aux):
        contrib = _to_numpy(aux['contribution'])
        if contrib.ndim == 3: contrib_mean = contrib.mean(axis=0)
        else: contrib_mean = contrib
        N, L = contrib_mean.shape
        fig, ax = plt.subplots(figsize=(12, max(4, N*0.4)))
        names = self.var_names if self.var_names else [f'Var{j}' for j in range(N)]
        im = ax.imshow(contrib_mean, aspect='auto', cmap='RdBu_r')
        ax.set_xticks(range(L)); ax.set_yticks(range(N))
        ax.set_yticklabels(names); ax.set_xlabel('Lag tau'); ax.set_ylabel('Variable')
        ax.set_title('Mean Contribution (Variable x Lag)')
        plt.colorbar(im, ax=ax); plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'contribution_heatmap.png')); plt.close()

    def plot_predictions(self, preds, targets):
        preds = _to_numpy(preds).flatten(); targets = _to_numpy(targets).flatten()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        n_show = min(500, len(targets))
        ax1.plot(targets[:n_show], 'b-', lw=1, alpha=0.7, label='True')
        ax1.plot(preds[:n_show], 'r--', lw=1, alpha=0.7, label='Pred')
        ax1.set_title('Predictions vs Ground Truth'); ax1.legend(); ax1.grid(True, alpha=0.3)
        residuals = preds - targets
        ax2.plot(residuals[:n_show], 'k-', lw=0.5, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_title('Residuals'); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'prediction_plot.png')); plt.close()
