"""Stage1TargetDelayKAN with conditional component construction.

Three configurations via constructor args:
  - StaticGammaStage1 (epsilon=0, use_true_delays=False): delay_prior only
  - FreeTiltStage1 (epsilon>0, use_true_delays=False): delay_prior + scorer
  - TrueDelayOracle (use_true_delays=True): neither delay_prior nor scorer

Active mask support for Oracle modes via active_mask parameter.
"""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from .delay_prior import DiscreteGammaPrior
from .scorer import BoundedLagScorer
from .response_kan import UnivariateKANResponse
from .sequence_ops import (ContiguousWindowBatch, materialize_static_aux,
                           static_depthwise_contribution, windows_to_raw_sequence)


class Stage1TargetDelayKAN(nn.Module):
    def __init__(self, num_variables: int, max_lag: int,
                 hidden_score: int = 8, hidden_kan: int = 8,
                 kan_grid_size: int = 7, kan_spline_order: int = 3,
                 epsilon: float = 0.0, use_true_delays: bool = False,
                 delta: float = 1e-3, active_threshold: float = 1e-6,
                 prior_parametrization: str = "mean_std",
                 delay_mode: str = "static_gamma",
                 active_mask: torch.Tensor = None,
                 input_grid_ranges=None, second_layer_grid_range=(-3, 3),
                 response_execution_mode: str = "auto"):

        super().__init__()
        self.num_variables = num_variables
        self.max_lag = max_lag
        self.epsilon = epsilon
        self.use_true_delays = use_true_delays
        self.active_threshold = active_threshold
        if delay_mode not in {"static_gamma", "free_static_logits"}:
            raise ValueError("delay_mode must be 'static_gamma' or 'free_static_logits'")
        if use_true_delays and delay_mode != "static_gamma":
            raise ValueError("True-delay oracle does not use a learned delay mode")
        self.delay_mode = delay_mode

        self.bias = nn.Parameter(torch.tensor(0.0))

        # ---- Conditional construction ----
        # delay_prior: built for all modes EXCEPT TrueDelayOracle
        if not use_true_delays and delay_mode == "static_gamma":
            self.delay_prior = DiscreteGammaPrior(
                num_variables, max_lag, delta=delta,
                parametrization=prior_parametrization)
        else:
            self.delay_prior = None

        # scorer: built only for FreeTilt (epsilon > 0 AND not oracle)
        if epsilon > 0 and not use_true_delays and delay_mode == "static_gamma":
            self.scorer = BoundedLagScorer(num_variables, hidden_score, epsilon)
        else:
            self.scorer = None

        # response_branches: always built
        # Active mask: if provided, zero out inactive branches
        if active_mask is not None:
            self.register_buffer("active_mask", active_mask.to(torch.bool))
        else:
            self.register_buffer("active_mask", torch.ones(num_variables, dtype=torch.bool))
        # Candidate membership and learned selection are deliberately separate.
        # M2 starts with every candidate enabled and makes selection irreversible.
        self.register_buffer("selection_mask", torch.ones(num_variables, dtype=torch.bool))

        self.response_branches = UnivariateKANResponse(
            num_variables, hidden_kan, kan_grid_size, kan_spline_order,
            input_grid_ranges=input_grid_ranges,
            second_layer_grid_range=second_layer_grid_range,
            active_mask=self.active_mask,
            execution_mode=response_execution_mode)
        # In O1, known-inactive response functions are not part of the hypothesis.
        if use_true_delays and active_mask is not None:
            for j, active in enumerate(self.active_mask.tolist()):
                if not active:
                    for parameter in self.response_branches.branches[j].parameters():
                        parameter.requires_grad_(False)

        # This is intentionally a buffer, rather than a Python attribute: centered
        # checkpoints must remain centered after a process restart.
        self.register_buffer("_is_centered", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("centers", torch.zeros(num_variables))
        self.register_buffer("_true_h", torch.zeros(num_variables, max_lag))
        self.register_buffer("_fixed_q", torch.zeros(num_variables, max_lag))
        # Dummy taus for Oracle mode
        self.register_buffer("_dummy_taus", torch.arange(max_lag, dtype=torch.float32))
        if not use_true_delays and delay_mode == "free_static_logits":
            self.delay_logits = nn.Parameter(torch.zeros(num_variables, max_lag))
        else:
            self.register_parameter("delay_logits", None)
        self._install_delay_gradient_masks()

    def _delay_gradient_mask(self, gradient: torch.Tensor) -> torch.Tensor:
        mask = (self.active_mask & self.selection_mask).to(
            device=gradient.device, dtype=gradient.dtype)
        return gradient * (mask[:, None] if gradient.ndim == 2 else mask)

    def _install_delay_gradient_masks(self) -> None:
        """Mask learned-delay rows without replacing vector Parameters."""
        if self.delay_prior is not None:
            for name in ("raw_mean", "raw_std", "raw_alpha", "raw_beta"):
                parameter = getattr(self.delay_prior, name, None)
                if isinstance(parameter, nn.Parameter):
                    parameter.register_hook(self._delay_gradient_mask)
        if self.delay_logits is not None:
            self.delay_logits.register_hook(self._delay_gradient_mask)

    @torch.no_grad()
    def prune_variable(self, variable: int, optimizer=None) -> None:
        """Permanently remove one response branch and its learned-delay row."""
        if not 0 <= variable < self.num_variables:
            raise IndexError(variable)
        self.selection_mask[variable] = False
        for parameter in self.response_branches.branches[variable].parameters():
            parameter.zero_(); parameter.requires_grad_(False)
            if optimizer is not None:
                optimizer.state.pop(parameter, None)
                for group in optimizer.param_groups:
                    group["params"] = [p for p in group["params"] if p is not parameter]
        if optimizer is not None:
            delay_parameters = []
            if self.delay_prior is not None:
                delay_parameters += [p for p in self.delay_prior.parameters()]
            if self.delay_logits is not None:
                delay_parameters.append(self.delay_logits)
            for parameter in delay_parameters:
                state = optimizer.state.get(parameter, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = state.get(key)
                    if torch.is_tensor(value) and value.ndim and value.shape[0] == self.num_variables:
                        value[variable].zero_()

    @property
    def is_centered(self) -> bool:
        return bool(self._is_centered.item())

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load old uncentered checkpoints safely.

        v14 and older state dicts have no ``_is_centered`` entry.  They are
        explicitly treated as uncentered; we never infer the state from centers.
        """
        if "_is_centered" not in state_dict:
            warnings.warn(
                "Checkpoint has no _is_centered buffer; loading it as an "
                "uncentered legacy checkpoint.", UserWarning, stacklevel=2)
            state_dict = state_dict.copy()
            state_dict["_is_centered"] = torch.tensor(False, dtype=torch.bool)
        if "selection_mask" not in state_dict:
            state_dict = state_dict.copy()
            state_dict["selection_mask"] = torch.ones(
                self.num_variables, dtype=torch.bool)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def set_true_delays(self, true_h: torch.Tensor, order=None):
        """Only accepts LagOrder.CURRENT_TO_PAST canonical physical lag."""
        from .lag_contract import LagOrder
        if order is None:
            raise ValueError(
                "set_true_delays requires explicit order= parameter. "
                "Use order=LagOrder.CURRENT_TO_PAST.")
        if order != LagOrder.CURRENT_TO_PAST:
            raise ValueError(
                "set_true_delays only accepts CURRENT_TO_PAST.")
        if tuple(true_h.shape) != (self.num_variables, self.max_lag):
            raise ValueError(f"Expected [{self.num_variables},{self.max_lag}], got {list(true_h.shape)}")
        if not torch.isfinite(true_h).all():
            raise ValueError("true_h must be finite")
        if (true_h < 0).any():
            raise ValueError("true_h must be nonnegative")
        row_sums = true_h.sum(dim=-1)
        active = self.active_mask.to(device=true_h.device)
        if active.any() and not torch.allclose(row_sums[active], torch.ones_like(row_sums[active]), atol=1e-6):
            raise ValueError("active true_h rows must sum to one")
        inactive_nonzero = (~active) & (row_sums.abs() > 1e-8)
        if inactive_nonzero.any() and not torch.allclose(
                row_sums[inactive_nonzero], torch.ones_like(row_sums[inactive_nonzero]), atol=1e-6):
            raise ValueError("inactive true_h rows must be zero or sum to one")
        self._true_h.copy_(true_h.to(device=self.bias.device, dtype=self.bias.dtype))
        self.use_true_delays = True

    def set_fixed_delays(self, q: torch.Tensor):
        """Use an externally supplied normalized lag distribution (O2 only)."""
        if tuple(q.shape) != (self.num_variables, self.max_lag):
            raise ValueError("fixed q has wrong shape")
        if not torch.isfinite(q).all() or (q < 0).any() or not torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol=1e-6):
            raise ValueError("fixed q must be finite, nonnegative, and row-normalized")
        self._fixed_q.copy_(q.to(self.bias)); self.use_true_delays = False


    def remove_centering(self):
        if self.is_centered:
            self.bias.data = self.bias.data - self.centers.sum()
            self.centers.zero_()
            self._is_centered.fill_(False)

    def _check_trainable(self):
        if self.is_centered and self.training:
            raise RuntimeError(
                "Cannot train a centered model. Call remove_centering() first "
                "or load an uncentered checkpoint.")

    def _flip_input(self, x):
        return torch.flip(x, dims=[-1])

    def _static_q(self) -> torch.Tensor | None:
        """Return a static ``[N,L]`` q, or None for sample-dependent tilts."""
        if self._fixed_q.abs().sum() > 0:
            return self._fixed_q
        if self.use_true_delays and self._true_h.abs().sum() > 0:
            return self._true_h
        if self.delay_mode == "free_static_logits":
            logits = torch.where(
                self.active_mask[:, None], self.delay_logits,
                torch.zeros_like(self.delay_logits))
            return F.softmax(logits, dim=-1)
        if self.scorer is None and self.delay_prior is not None:
            return self.delay_prior()
        return None

    def forward_contiguous(
        self,
        batch_or_windows,
        *,
        return_aux: bool = False,
        validate_windows: bool = True,
        materialize_lag_aux: bool = False,
    ):
        """Exact sequence-first forward for a contiguous static-delay split.

        The response KAN is evaluated on only ``B+L-1`` unique time points,
        rather than on ``B*L`` repeated window entries.  If a dynamic scorer is
        present, the function safely falls back to the ordinary window path.
        """
        self._check_trainable()
        if isinstance(batch_or_windows, ContiguousWindowBatch):
            windows = batch_or_windows.windows
            raw_sequence = batch_or_windows.raw_sequence
        else:
            windows = batch_or_windows
            raw_sequence = windows_to_raw_sequence(
                windows, validate=validate_windows)
        q_static = self._static_q()
        if q_static is None:
            return self.forward(windows, return_aux=return_aux)
        response_sequence = self.response_branches.forward_sequence(raw_sequence)
        forward_mask = (self.active_mask & self.selection_mask).to(
            response_sequence.device, response_sequence.dtype)
        response_sequence = response_sequence * forward_mask[:, None]
        centers = self.centers if self.is_centered else None
        variable_contribution = static_depthwise_contribution(
            response_sequence, q_static, centers=centers)
        y_hat = self.bias + variable_contribution.sum(dim=-1)
        if not return_aux:
            return y_hat.unsqueeze(-1)

        n = self.num_variables
        prior_mean = prior_var = None
        if self.delay_prior is not None:
            prior_mean, prior_var = self.delay_prior.compute_prior_stats()
        else:
            prior_mean = q_static.new_zeros(n)
            prior_var = q_static.new_zeros(n)
        taus = self._dummy_taus.to(q_static)
        posterior_mean_static = (q_static * taus).sum(-1)
        posterior_var_static = (
            q_static * (taus[None, :] - posterior_mean_static[:, None]).square()
        ).sum(-1)
        batch_size = variable_contribution.shape[0]
        if materialize_lag_aux:
            q_batch, response, contribution, variable_contribution_check = materialize_static_aux(
                response_sequence, q_static, centers=centers)
            if not torch.allclose(variable_contribution, variable_contribution_check, atol=1e-6, rtol=1e-6):
                raise RuntimeError("sequence and materialized contributions disagree")
        else:
            # Keep large lag tensors absent in the optimized training path.
            q_batch = q_static.unsqueeze(0).expand(batch_size, -1, -1)
            response = contribution = None
        aux = {
            "pi": q_static,
            "q": q_batch,
            "response": response,
            "contribution": contribution,
            "variable_contribution": variable_contribution,
            "batch_variable_importance": variable_contribution.abs().mean(0),
            "prior_delay_mean": prior_mean,
            "prior_delay_var": prior_var,
            "posterior_delay_mean": posterior_mean_static.unsqueeze(0).expand(batch_size, -1),
            "posterior_delay_var": posterior_var_static.unsqueeze(0).expand(batch_size, -1),
            "branch_norm": self.response_branches.compute_branch_norms().to(q_static),
            "prior_boundary_mass_last3": q_static[:, -3:].sum(-1),
            "posterior_boundary_mass_last3": q_static[:, -3:].sum(-1).unsqueeze(0).expand(batch_size, -1),
            "sequence_fast_path": True,
        }
        return y_hat.unsqueeze(-1), aux

    def _compute_q_and_response(self, x_lag):
        B, N, L = x_lag.shape

        if self._fixed_q.abs().sum() > 0:
            q = self._fixed_q.unsqueeze(0).expand(B, N, L)
        elif self.use_true_delays and self._true_h.abs().sum() > 0:
            q = self._true_h.unsqueeze(0).expand(B, N, L)
        elif self.delay_mode == "free_static_logits":
            # Detached inactive rows make their gradient exactly zero while
            # retaining a valid uniform distribution for every variable.
            logits = self.delay_logits
            uniform_logits = torch.zeros_like(logits)
            logits = torch.where(self.active_mask[:, None], logits, uniform_logits)
            q = F.softmax(logits, dim=-1).unsqueeze(0).expand(B, N, L)
        else:
            pi = self.delay_prior()
            r = self.scorer(x_lag) if self.scorer is not None else torch.zeros_like(x_lag)
            log_pi = torch.log(pi + 1e-12)
            logits = log_pi.unsqueeze(0) + r
            q = F.softmax(logits, dim=-1)

        response_raw = self.response_branches(x_lag)
        # Apply active mask: zero response for inactive variables
        forward_mask = self.active_mask & self.selection_mask
        response_raw = response_raw * forward_mask.view(1, N, 1).to(response_raw.dtype)
        return q, response_raw

    def _compute_contribution(self, q, response_raw):
        if self.is_centered:
            response = response_raw - self.centers.view(1, self.num_variables, 1)
        else:
            response = response_raw
        contribution = q * response
        return contribution, response

    def forward(self, x, return_aux=True):
        self._check_trainable()
        B, N, L = x.shape
        x_lag = self._flip_input(x)
        q, response_raw = self._compute_q_and_response(x_lag)
        contribution, response = self._compute_contribution(q, response_raw)
        variable_contribution = contribution.sum(dim=-1)
        y_hat = self.bias + variable_contribution.sum(dim=-1)

        if not return_aux:
            return y_hat.unsqueeze(-1)

        # Prior stats (placeholder for Oracle mode)
        if self.delay_prior is not None:
            prior_mean, prior_var = self.delay_prior.compute_prior_stats()
            pi = self.delay_prior()
        else:
            prior_mean = q.new_zeros(N); prior_var = q.new_zeros(N)
            pi = (self._true_h if self._true_h.abs().sum() > 0 else
                  (q[0] if self.delay_logits is not None else q.new_ones(N, L) / L))

        taus = self._dummy_taus.to(device=q.device, dtype=q.dtype)
        posterior_mean = (q * taus).sum(dim=-1)
        posterior_var = (q * (taus.unsqueeze(0).unsqueeze(0) - posterior_mean.unsqueeze(-1)) ** 2).sum(dim=-1)
        branch_norms = self.response_branches.compute_branch_norms().to(
            device=q.device, dtype=q.dtype)
        prior_boundary = pi[:, -3:].sum(dim=-1)
        posterior_boundary = q[:, :, -3:].sum(dim=-1)
        batch_variable_importance = variable_contribution.abs().mean(dim=0)

        aux = {
            "pi": pi, "q": q, "response": response, "contribution": contribution,
            "variable_contribution": variable_contribution,
            "batch_variable_importance": batch_variable_importance,
            "prior_delay_mean": prior_mean, "prior_delay_var": prior_var,
            "posterior_delay_mean": posterior_mean, "posterior_delay_var": posterior_var,
            "branch_norm": branch_norms,
            "prior_boundary_mass_last3": prior_boundary,
            "posterior_boundary_mass_last3": posterior_boundary,
        }
        return y_hat.unsqueeze(-1), aux

    def get_active_variables(self):
        norms = self.response_branches.compute_branch_norms()
        return (norms > self.active_threshold) & self.active_mask & self.selection_mask

    @torch.no_grad()
    def fit_centering(self, train_loader):
        if self.is_centered:
            self.remove_centering()
        device = self.bias.device
        sum_vc = torch.zeros(self.num_variables, device=device)
        total_samples = 0
        self.eval()
        for bx, _ in train_loader:
            bx = bx.to(device); x_lag = self._flip_input(bx)
            q, response_raw = self._compute_q_and_response(x_lag)
            raw_vc = (q * response_raw).sum(dim=-1)
            sum_vc += raw_vc.sum(dim=0); total_samples += bx.size(0)
        self.eval()
        centers = sum_vc / total_samples
        self.centers.copy_(centers)
        self.bias.data += centers.sum()
        self._is_centered.fill_(True)

    def parameter_report(self):
        """Detailed parameter count: instantiated, effective, with/without gradient."""
        report = {"instantiated_trainable": 0, "optimizer_trainable": 0,
                  "effective_trainable": 0, "forward_active_trainable": 0,
                  "buffers": 0, "parameters_with_nonzero_gradient": 0}
        by_module = {}; by_parameter = {}
        for name, param in self.named_parameters():
            n = param.numel()
            mod = name.split(".")[0]
            by_module[mod] = by_module.get(mod, 0) + n
            by_parameter[name] = {"numel": n, "requires_grad": bool(param.requires_grad)}
            if param.requires_grad:
                report["instantiated_trainable"] += n
                report["optimizer_trainable"] += n
                report["effective_trainable"] += n
                report["forward_active_trainable"] += n
                if param.grad is not None and param.grad.abs().sum() > 0:
                    report["parameters_with_nonzero_gradient"] += n
        for name, buf in self.named_buffers():
            report["buffers"] += buf.numel()
        report["by_module"] = by_module
        report["by_parameter"] = by_parameter
        return report

    def extra_repr(self):
        return (f"N={self.num_variables}, L={self.max_lag}, "
                f"epsilon={self.epsilon}, oracle={self.use_true_delays}, "
                f"centered={self.is_centered}, "
                f"has_prior={self.delay_prior is not None}, "
                f"has_scorer={self.scorer is not None}, "
                f"response_mode={self.response_branches.execution_mode}")
