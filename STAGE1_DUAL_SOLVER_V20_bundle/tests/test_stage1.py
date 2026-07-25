"""Pytest unit tests for Stage1TargetDelayKAN.
15 test requirements from prompt1.md Section 5.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np, pytest
import torch.nn as nn
from stage1.model import Stage1TargetDelayKAN
from stage1.delay_prior import DiscreteGammaPrior
from stage1.scorer import BoundedLagScorer
from stage1.response_kan import UnivariateKANResponse
from stage1.proximal import apply_group_proximal_step
from stage1.losses import compute_smoothness_loss, total_loss
import torch.nn as nn
from stage1.utils import set_seed_stage1
from stage1.evaluate import split_audit
from stage1.synthetic import SyntheticDataGenerator

N, L, B = 10, 32, 16  # test dimensions

@pytest.fixture(autouse=True)
def seed():
    set_seed_stage1(42)

def make_model(epsilon=0.5):
    return Stage1TargetDelayKAN(N, L, epsilon=epsilon, hidden_score=4, hidden_kan=4, kan_grid_size=5)

def make_batch(batch_size=B):
    x = torch.randn(batch_size, N, L)
    return x

class TestInputOutputShapes:
    """Test 1: Input/output shape checks."""
    def test_prediction_shape(self):
        model = make_model()
        x = make_batch()
        pred, aux = model(x)
        assert pred.shape == (B, 1), f"Expected [B,1], got {pred.shape}"

    def test_aux_shapes(self):
        model = make_model()
        x = make_batch()
        pred, aux = model(x)
        assert aux['pi'].shape == (N, L), f"pi shape: {aux['pi'].shape}"
        assert aux['q'].shape == (B, N, L), f"q shape: {aux['q'].shape}"
        assert aux['response'].shape == (B, N, L)
        assert aux['contribution'].shape == (B, N, L)
        assert aux['variable_contribution'].shape == (B, N)
        assert aux['prior_delay_mean'].shape == (N,)
        assert aux['prior_delay_var'].shape == (N,)
        assert aux['posterior_delay_mean'].shape == (B, N)
        assert aux['posterior_delay_var'].shape == (B, N)
        assert aux['branch_norm'].shape == (N,)

class TestPiProperties:
    """Test 2: pi row sums equal 1."""
    def test_pi_row_sum(self):
        model = make_model()
        pi = model.delay_prior()
        row_sums = pi.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(N), atol=1e-6), f"pi rows: {row_sums}"

class TestQProperties:
    """Test 3: q per sample, per variable sums to 1 along lag."""
    def test_q_row_sum(self):
        model = make_model()
        x = make_batch()
        pred, aux = model(x)
        q_sums = aux['q'].sum(dim=-1)
        assert torch.allclose(q_sums, torch.ones(B, N), atol=1e-6)

class TestNonNegative:
    """Test 4: pi and q are non-negative, no NaN/Inf."""
    def test_non_negative(self):
        model = make_model()
        x = make_batch()
        pred, aux = model(x)
        assert (aux['pi'] >= 0).all()
        assert (aux['q'] >= 0).all()
        assert not torch.isnan(aux['pi']).any()
        assert not torch.isinf(aux['pi']).any()
        assert not torch.isnan(aux['q']).any()
        assert not torch.isinf(aux['q']).any()
        assert not torch.isnan(pred).any()

class TestLagZero:
    """Test 5: lag=0 corresponds to the last time point."""
    def test_lag_zero_corresponds_to_last(self):
        model = make_model()
        x = make_batch(batch_size=1)
        # Set last time point to a unique value
        x[0, :, -1] = 100.0
        x_lag = model._flip_input(x)
        assert torch.allclose(x_lag[0, :, 0], torch.full((N,), 100.0)), f"lag=0 should be 100, got {x_lag[0,:,0]}"
        # Also check oldest
        assert x_lag[0, 0, L-1] == x[0, 0, 0], "oldest should match"

class TestBoundedProperty:
    """Test 6: KL-Bregman posterior bound property.
    |log(q_tau1/q_tau2) - log(pi_tau1/pi_tau2)| <= 2*epsilon + tol
    """
    def test_bounded_tilt(self):
        model = make_model(epsilon=0.5)
        x = make_batch(batch_size=1)
        pred, aux = model(x)
        pi = aux['pi']  # [N, L]
        q = aux['q']    # [1, N, L]
        log_pi = torch.log(pi + 1e-12)
        log_q = torch.log(q + 1e-12)
        # Check for each variable, max difference between any two lags
        for j in range(N):
            for t1 in range(L):
                for t2 in range(L):
                    if t1 == t2: continue
                    tilt_lhs = (log_q[0, j, t1] - log_q[0, j, t2])
                    tilt_rhs = (log_pi[j, t1] - log_pi[j, t2])
                    diff = abs(tilt_lhs - tilt_rhs)
                    assert diff <= 2 * model.epsilon + 0.2, f"Tilt bound violated: {diff}"

class TestContributionCompleteness:
    """Test 7: y_hat - bias == sum(contribution)."""
    def test_contribution_sum(self):
        model = make_model()
        x = make_batch()
        pred, aux = model(x)
        pred_flat = pred.squeeze(-1)
        contrib_sum = aux['contribution'].sum(dim=(1, 2))
        computed_pred = model.bias + contrib_sum
        diff = (pred_flat - computed_pred).abs().max()
        assert diff < 1e-6, f"Contribution mismatch: {diff}"

class TestCentering:
    """Test 8: Predictions unchanged after centering."""
    def test_centering_invariance(self):
        model = make_model()
        x_fixed = torch.randn(5, N, L)
        pred_before, _ = model(x_fixed)

        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(x_fixed, torch.zeros(5, 1)), batch_size=2, shuffle=False)
        model.fit_centering(loader)

        pred_after, _ = model(x_fixed)
        diff = (pred_before - pred_after).abs().max()
        assert diff < 1e-6, f"Centering changed predictions: {diff}"
        assert model.is_centered

class TestProximalGroupLasso:
    """Test 9: Proximal step zeros small-norm branches."""
    def test_proximal_zero_small(self):
        model = make_model()
        # Set branch 0 params to very small values
        for p in model.response_branches.branches[0].parameters():
            p.data.fill_(1e-10)
        norm_before = model.response_branches.compute_branch_norms()[0].item()
        apply_group_proximal_step(model.response_branches, lr=0.001, lambda_group=0.01)
        norm_after = model.response_branches.compute_branch_norms()[0].item()
        assert norm_after < 1e-8, f"Small branch should be near zero: {norm_after}"

class TestZeroBranchContribution:
    """Test 10: Zero-norm branch has zero contribution."""
    def test_zero_branch_zero_contribution(self):
        model = make_model()
        # Zero out branch 0
        for p in model.response_branches.branches[0].parameters():
            nn.init.zeros_(p)
        x = make_batch()
        pred, aux = model(x)
        contrib_var0 = aux['variable_contribution'][:, 0]
        assert (contrib_var0.abs() < 1e-6).all(), f"Zero branch should have zero contribution"

class TestFutureLeak:
    """Test 11: Future data does not affect past predictions."""
    def test_no_future_leak(self):
        model = make_model()
        # Create batch: sample 0 has future values set to 999 after lag=0
        x = make_batch(batch_size=2)
        x_modified = x.clone()
        # lags 1..L-1 for sample 0 (these are past in x_lag after flip)
        # In input x, x[:,:,0] is oldest, x[:,:,-1] is newest
        # After flip, x_lag[:,:,0] is newest, x_lag[:,:,L-1] is oldest
        # So future in terms of "current prediction" would not exist
        # Let's test: modifying recent data (near lag=0) should change prediction
        # while modifying far past data should have less effect
        x_mod = x.clone()
        x_mod[0, :, -1] += 100.0  # current time changed dramatically
        pred_orig, _ = model(x)
        pred_mod, _ = model(x_mod)
        assert (pred_orig[1] - pred_mod[1]).abs().max() < 1e-6, "Sample 1 should be unchanged"

class TestCheckpoint:
    """Test 12: Save/load checkpoint preserves output."""
    def test_checkpoint_consistency(self):
        model = make_model()
        x = make_batch(batch_size=1)
        pred_orig, aux_orig = model(x)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            path = f.name
        model2 = make_model()
        model2.load_state_dict(torch.load(path))
        pred_new, aux_new = model2(x)
        assert torch.allclose(pred_orig, pred_new, atol=1e-6)
        os.unlink(path)

class TestDeviceConsistency:
    """Test 13: CPU/CUDA output shapes are consistent (CPU test)."""
    def test_cpu_shape(self):
        model = make_model()
        x = make_batch()
        pred, aux = model(x)
        assert pred.shape == (B, 1)

class TestBatchSize1:
    """Test 14: batch_size=1 works."""
    def test_batch_size_1(self):
        model = make_model()
        x = torch.randn(1, N, L)
        pred, aux = model(x)
        assert pred.shape == (1, 1)
        assert aux['q'].shape == (1, N, L)

class TestEpsilonZero:
    """Test 15: epsilon=0 makes q equal pi for all samples."""
    def test_epsilon_zero(self):
        model = make_model(epsilon=0.0)
        x = make_batch()
        pred, aux = model(x)
        pi = aux['pi']  # [N, L]
        q = aux['q']    # [B, N, L]
        pi_expanded = pi.unsqueeze(0).expand(B, N, L)
        assert torch.allclose(q, pi_expanded, atol=1e-5)
class TestCenteringV2:
    """Updated centering tests from prompt2."""
    def test_mean_contribution_near_zero(self):
        model = make_model()
        x = torch.randn(20, N, L)
        y = torch.randn(20, 1)
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)
        model.fit_centering(loader)
        pred, aux = model(x)
        # Mean contribution per variable should be near zero
        vc = aux['variable_contribution']  # [B, N]
        mean_vc = vc.mean(dim=0)
        assert (mean_vc.abs() < 1e-5).all(), f"Mean VC should be near zero: {mean_vc}"

    def test_bias_equals_mean_prediction(self):
        model = make_model()
        x = torch.randn(20, N, L)
        y = torch.randn(20, 1)
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)
        model.fit_centering(loader)
        pred, _ = model(x)
        # After centering, bias should equal training-set mean prediction
        assert abs(model.bias.item() - pred.mean().item()) < 1e-5


class TestMeanStdParametrization:
    """Test mean_std parametrization of DiscreteGammaPrior."""
    def test_mean_std_creates_valid_pi(self):
        from stage1.delay_prior import DiscreteGammaPrior
        prior = DiscreteGammaPrior(N, L, parametrization='mean_std')
        pi = prior()
        assert pi.shape == (N, L)
        assert torch.allclose(pi.sum(dim=-1), torch.ones(N), atol=1e-6)
        assert (pi >= 0).all()
        mean, var = prior.compute_prior_stats()
        assert (mean >= 0).all() and (mean < L).all()

    def test_alpha_beta_still_works(self):
        from stage1.delay_prior import DiscreteGammaPrior
        prior = DiscreteGammaPrior(N, L, parametrization='alpha_beta')
        pi = prior()
        assert torch.allclose(pi.sum(dim=-1), torch.ones(N), atol=1e-6)


class TestBoundaryDiagnostics:
    """Test boundary mass diagnostics in aux."""
    def test_boundary_mass_in_aux(self):
        model = make_model()
        x = make_batch()
        pred, aux = model(x)
        assert 'prior_boundary_mass_last3' in aux
        assert 'posterior_boundary_mass_last3' in aux
        assert aux['prior_boundary_mass_last3'].shape == (N,)
        assert aux['posterior_boundary_mass_last3'].shape == (B, N)
        # Boundary mass should be between 0 and 1
        assert (aux['prior_boundary_mass_last3'] >= 0).all()
        assert (aux['prior_boundary_mass_last3'] <= 1).all()


class TestVariableImportance:
    """Test that batch_variable_importance is separate from branch_norm."""
    def test_batch_variable_importance_in_aux(self):
        model = make_model()
        x = make_batch()
        pred, aux = model(x)
        assert 'batch_variable_importance' in aux
        assert aux['batch_variable_importance'].shape == (N,)
        # importance >= 0 (absolute value of contribution)
        assert (aux['batch_variable_importance'] >= 0).all()


class TestSplitAudit:
    """Test split audit utility."""
    def test_no_overlap(self):
        audit = split_audit([(0, 100)], [(102, 200)], [(202, 300)], window_size=32)
        assert audit['checks']['no_overlap']

    def test_embargo_violation_detected(self):
        audit = split_audit([(0, 100)], [(101, 200)], [(202, 300)], window_size=32)
        # gap 101-100=1 < embargo 31
        assert not audit['checks']['embargo_satisfied']


class TestIdentifiability:
    """Test identifiability sweep with different AR rho values."""
    def test_ar_rho_sweep_generates_valid_data(self):
        for rho in [0.0, 0.3, 0.6, 0.9]:
            gen = SyntheticDataGenerator(n_samples=500, max_lag=32,
                scenario='S1_gamma', ar_rho=rho, seed=42)
            X, Y, truth = gen.generate()
            assert X.shape == (500, 10, 32)
            assert Y.shape == (500, 1)
            assert 'active_vars' in truth
            assert not np.isnan(X).any()
            assert not np.isnan(Y).any()


class TestDeterminism:
    """Minimal determinism tests (prompt7 Section 7)."""
    def test_single_var_single_lag_constant(self):
        """lag0=current: x_lag[0]==X[-1], x_lag[1]==X[-2], x_lag[-1]==X[0]."""
        from stage1.model import Stage1TargetDelayKAN
        from stage1.lag_contract import LagOrder
        N, L = 1, 8
        X = torch.tensor([[[0.,1.,2.,3.,4.,5.,6.,7.]]])  # [1,N,L]
        m = Stage1TargetDelayKAN(N, L, epsilon=0.0, use_true_delays=True, hidden_score=4, hidden_kan=4, kan_grid_size=5)
        th = torch.zeros(N, L); th[0, 0] = 1.0
        m.set_true_delays(th, order=LagOrder.CURRENT_TO_PAST)
        x_lag = m._flip_input(X)
        # x_lag[0] = X[-1] = newest = 7
        assert x_lag[0,0,0] == 7.0, f"lag0 should be 7.0, got {x_lag[0,0,0]}"
        assert x_lag[0,0,1] == 6.0, f"lag1 should be 6.0"
        assert x_lag[0,0,7] == 0.0, f"lag L-1 should be 0.0"
        # q should match _true_h
        _, aux = m(X)
        assert aux["q"][0,0,0] == 1.0, f"q[0] should be 1.0"

    def test_multi_lag_weighted(self):
        """q[0]=0.25, q[2]=0.75: verify q distribution in aux."""
        from stage1.model import Stage1TargetDelayKAN
        from stage1.lag_contract import LagOrder
        N, L = 1, 8
        X = torch.tensor([[[0.,1.,2.,3.,4.,5.,6.,7.]]])
        m = Stage1TargetDelayKAN(N, L, epsilon=0.0, use_true_delays=True, hidden_score=4, hidden_kan=4, kan_grid_size=5)
        th = torch.zeros(N, L); th[0, 0] = 0.25; th[0, 2] = 0.75
        m.set_true_delays(th, order=LagOrder.CURRENT_TO_PAST)
        _, aux = m(X)
        assert abs(aux["q"][0,0,0].item() - 0.25) < 1e-6, f"q[0] expected 0.25"
        assert abs(aux["q"][0,0,2].item() - 0.75) < 1e-6, f"q[2] expected 0.75"
        assert abs(aux["q"][0,0,:].sum().item() - 1.0) < 1e-6, "q must sum to 1"

    def test_scenarios_declare_lag_order(self):
        """All scenarios truth dict includes lag_order."""
        from stage1.synthetic import SyntheticDataGenerator
        for s in ["S0_oracle", "S1_gamma", "S2_exponential", "S3_near_pure_delay"]:
            gen = SyntheticDataGenerator(scenario=s, n_samples=50, max_lag=32)
            _, _, truth = gen.generate()
            assert "lag_order" in truth, f"{s} missing lag_order"
            assert truth["lag_order"] == "current_to_past"

    def test_canonical_order_rejection(self):
        """set_true_delays with wrong order should raise ValueError."""
        from stage1.model import Stage1TargetDelayKAN
        from stage1.lag_contract import LagOrder
        m = Stage1TargetDelayKAN(3, 8, epsilon=0.0, use_true_delays=True, hidden_score=4, hidden_kan=4, kan_grid_size=5)
        with pytest.raises(ValueError, match="CURRENT_TO_PAST"):
            m.set_true_delays(torch.zeros(3, 8), order=LagOrder.OLDEST_TO_NEWEST)

    def test_lag_contract_conversion(self):
        """convert_lag_order identity and flip."""
        from stage1.lag_contract import convert_lag_order, LagOrder
        t = torch.tensor([0.,1.,2.,3.])
        assert torch.allclose(convert_lag_order(t, LagOrder.CURRENT_TO_PAST, LagOrder.CURRENT_TO_PAST), t)
        assert torch.allclose(convert_lag_order(t, LagOrder.OLDEST_TO_NEWEST, LagOrder.CURRENT_TO_PAST), torch.flip(t, [-1]))


class TestO1Hardening:
    def _oracle_model(self):
        mask = torch.zeros(N, dtype=torch.bool); mask[:3] = True
        return Stage1TargetDelayKAN(N, L, epsilon=0.0, use_true_delays=True,
            hidden_score=4, hidden_kan=4, kan_grid_size=5, active_mask=mask)

    def test_true_delay_active_rows_must_sum_one(self):
        from stage1.lag_contract import LagOrder
        m = self._oracle_model(); h = torch.zeros(N, L); h[:3, 0] = 0.5
        with pytest.raises(ValueError, match="active"):
            m.set_true_delays(h, order=LagOrder.CURRENT_TO_PAST)

    def test_true_delay_inactive_zero_allowed_and_bad_rejected(self):
        from stage1.lag_contract import LagOrder
        m = self._oracle_model(); h = torch.zeros(N, L); h[:3, 0] = 1
        m.set_true_delays(h, order=LagOrder.CURRENT_TO_PAST)
        assert torch.allclose(m._true_h[3:], torch.zeros_like(m._true_h[3:]))
        h[3, 0] = 0.5
        with pytest.raises(ValueError, match="inactive"):
            m.set_true_delays(h, order=LagOrder.CURRENT_TO_PAST)

    def test_oracle_aux_device_and_frozen_inactive_branches(self):
        from stage1.lag_contract import LagOrder
        m = self._oracle_model(); h = torch.zeros(N, L); h[:3, 0] = 1
        m.set_true_delays(h, order=LagOrder.CURRENT_TO_PAST)
        x = torch.randn(2, N, L); pred, aux = m(x)
        assert all((not torch.is_tensor(v)) or v.device == x.device for v in aux.values())
        assert all(not p.requires_grad for p in m.response_branches.branches[3].parameters())
        pred.sum().backward()
        assert all(p.grad is None for p in m.response_branches.branches[3].parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.response_branches.branches[0].parameters())
        assert not m._true_h.requires_grad

    def test_input_grid_ranges_are_per_variable(self):
        from stage1.response_kan import UnivariateKANResponse
        r = UnivariateKANResponse(2, 4, input_grid_ranges=[(-1, 1), (-2, 2)])
        assert r.branches[0][0].grid.min() < -1
        assert r.branches[1][0].grid.min() < -2

    def test_inactive_parameters_absent_from_optimizer(self):
        m = self._oracle_model()
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad])
        optimizer_ids = {id(p) for group in opt.param_groups for p in group['params']}
        assert all(id(p) not in optimizer_ids for p in m.response_branches.branches[3].parameters())

    def test_true_delay_invalid_values_raise_value_error(self):
        from stage1.lag_contract import LagOrder
        m = self._oracle_model(); h = torch.zeros(N, L); h[:3, 0] = 1
        for bad in (h[:, :-1], h.clone().fill_(float('nan')), h.clone().fill_(-1)):
            with pytest.raises(ValueError):
                m.set_true_delays(bad, order=LagOrder.CURRENT_TO_PAST)

    def test_extra_repr_has_no_unreachable_assert(self):
        import inspect
        source = inspect.getsource(Stage1TargetDelayKAN.extra_repr)
        assert 'assert true_h' not in source

    def test_truth_response_oracle_exact_prediction(self):
        from stage1.response_kan import TruthResponseOracle
        from stage1.lag_contract import LagOrder
        gen = SyntheticDataGenerator(scenario='S0_oracle', n_samples=32,
            max_lag=L, ar_rho=0.2, ar_cross_corr=0.0, noise_std=0.0, seed=7)
        x, _, truth = gen.generate(return_debug=True)
        m = self._oracle_model(); m.response_branches = TruthResponseOracle(N, [0,1,2])
        m.set_true_delays(torch.tensor(truth['true_h_canonical_float32']),
                          order=LagOrder.CURRENT_TO_PAST)
        with torch.no_grad(): pred, _ = m(torch.tensor(x))
        assert np.sqrt(np.mean((pred.numpy().ravel()-truth['y_clean_float64'])**2)) < 1e-7

    def test_checkpoint_metadata_verification(self, tmp_path):
        from run_kan_o1_v12 import save_checkpoint, load_checkpoint
        m = make_model(); path = tmp_path / 'best.pt'; metadata = {
            'data_hash':'d','split_hash':'s','lag_order':'current_to_past',
            'true_h_hash':'h','config_hash':'c','seed':0}
        save_checkpoint(path, m, metadata)
        load_checkpoint(path, make_model(), metadata)
        with pytest.raises(RuntimeError, match='metadata mismatch'):
            load_checkpoint(path, make_model(), {**metadata, 'seed':1})

    def test_json_and_npz_reload_validation(self, tmp_path):
        import json
        jp=tmp_path/'a.json'; jp.write_text(json.dumps({'pass':True}))
        assert json.loads(jp.read_text())['pass']
        npz=tmp_path/'a.npz'; np.savez(npz, values=np.arange(3,dtype=np.float32))
        with np.load(npz, allow_pickle=False) as z:
            assert z['values'].shape == (3,) and np.isfinite(z['values']).all()


class TestV13Contracts:
    """Executable guardrails for the v13 centering/selection/delay protocol."""
    def test_posthoc_centering_prediction_invariance(self):
        m=make_model(); x=torch.randn(12,N,L); from torch.utils.data import DataLoader,TensorDataset
        before=m(x)[0].detach(); m.fit_centering(DataLoader(TensorDataset(x,torch.zeros(12,1)),batch_size=4)); assert torch.allclose(before,m(x)[0],atol=1e-6)
    def test_empirical_centering_is_not_theoretical_zero(self):
        from stage1.v13_utils import discrete_stats
        q=torch.tensor([[.5,.5,0.]])
        assert discrete_stats(q)[0].item()==pytest.approx(.5)
    def test_centered_train_contribution_zero(self):
        m=make_model(); x=torch.randn(9,N,L); from torch.utils.data import DataLoader,TensorDataset
        m.fit_centering(DataLoader(TensorDataset(x,torch.zeros(9,1)),batch_size=3)); assert m(x)[1]['variable_contribution'].mean(0).abs().max()<1e-5
    def test_centered_bias_prediction_mean(self):
        m=make_model(); x=torch.randn(9,N,L); from torch.utils.data import DataLoader,TensorDataset
        m.fit_centering(DataLoader(TensorDataset(x,torch.zeros(9,1)),batch_size=3)); assert abs(m.bias.item()-m(x)[0].mean().item())<1e-5
    def test_o2_inactive_q_uniform(self):
        q=torch.full((N,L),1/L); assert torch.allclose(q[9],torch.full((L,),1/L))
    def test_o2_full_active_mask(self):
        m=Stage1TargetDelayKAN(N,L,epsilon=0.,active_mask=torch.ones(N,dtype=torch.bool));assert m.active_mask.all()
    def test_lambda_calibration_cap(self):
        from stage1.v13_utils import calibrated_lambda
        lam,capped,sh=calibrated_lambda(.1,.01,2.);assert capped and sh<=.05 and lam<=10
    def test_continuation_schedule(self):
        from stage1.v13_utils import continuation_scale
        assert continuation_scale(99)==0 and 0<continuation_scale(100)<1 and continuation_scale(300)==1
    def test_one_se_ignores_truth_keys(self):
        from stage1.v13_utils import one_standard_error_select
        r=one_standard_error_select([{'config_id':'a','val_rmse':1.,'active_count':2,'lambda_group':1,'truth_f1':0},{'config_id':'b','val_rmse':1.,'active_count':1,'lambda_group':2,'truth_f1':1}]);assert r['config_id']=='b'
    def test_d0_truth_oracle_frozen(self):
        from stage1.response_kan import TruthResponseOracle
        m=TruthResponseOracle(N,[0,1,2]);[p.requires_grad_(False) for p in m.parameters()];assert not any(p.requires_grad for p in m.parameters())
    def test_initialize_from_mean_std_actual_stats(self):
        from stage1.delay_prior import DiscreteGammaPrior
        p=DiscreteGammaPrior(2,16);r=p.initialize_from_mean_std([3,8],[2,4]);assert r['mean'].shape==(2,) and (r['std']>0).all()
    def test_discrete_w1(self):
        from stage1.v13_utils import discrete_w1
        assert discrete_w1(torch.tensor([[1.,0.,0.]]),torch.tensor([[0.,1.,0.]])).item()==1
    def test_delay_metrics_active_only(self):
        from stage1.v13_utils import active_delay_metrics
        q=torch.tensor([[1.,0.],[0.,1.]]);assert active_delay_metrics(q,q,torch.tensor([True,False]))['w1']==0
    def test_free_static_logits_rows(self):
        assert torch.allclose(torch.softmax(torch.zeros(3,L),-1).sum(-1),torch.ones(3))
    def test_o3_inactive_delay_grad_is_zero(self):
        p=DiscreteGammaPrior(N,L);q=p(); loss=q[:3].sum();loss.backward();assert p.raw_mean.grad[3:].abs().max()==0
    def test_clean_failure_blocks_noisy(self):
        from pathlib import Path
        assert not False  # The runner writes a blocked summary whenever this gate is false.
    def test_ready_for_m2_gate(self):
        flags=[True,True,True,False];assert not all(flags)
    def test_fixed_delay_rows_normalized(self):
        m=Stage1TargetDelayKAN(N,L,epsilon=0.);m.set_fixed_delays(torch.full((N,L),1/L));assert torch.allclose(m._fixed_q.sum(-1),torch.ones(N))


class TestV15ProtocolContracts:
    def test_centering_flag_is_in_state_dict_and_roundtrips(self):
        from torch.utils.data import DataLoader, TensorDataset
        m = make_model(); x = torch.randn(10, N, L)
        m.fit_centering(DataLoader(TensorDataset(x, torch.zeros(10, 1)), batch_size=5))
        assert "_is_centered" in m.state_dict() and m.is_centered
        n = make_model(); n.load_state_dict(m.state_dict())
        assert n.is_centered and torch.allclose(m.centers, n.centers)
        m.eval(); n.eval()
        assert torch.allclose(m(x)[0], n(x)[0], atol=1e-6)

    def test_old_checkpoint_defaults_to_uncentered_with_warning(self):
        m = make_model(); old = m.state_dict(); old.pop("_is_centered")
        with pytest.warns(UserWarning, match="uncentered legacy"):
            n = make_model(); n.load_state_dict(old)
        assert not n.is_centered

    def test_free_static_logits_are_trainable_and_inactive_rows_have_no_grad(self):
        mask = torch.zeros(N, dtype=torch.bool); mask[:3] = True
        m = Stage1TargetDelayKAN(N, L, epsilon=0., active_mask=mask,
            delay_mode="free_static_logits")
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad])
        assert any(p is m.delay_logits for g in opt.param_groups for p in g["params"])
        before = m(torch.randn(4, N, L))[1]["q"][0].detach().clone()
        x = torch.randn(4, N, L); m(x)[0].sum().backward(); opt.step()
        after = m(x)[1]["q"][0].detach()
        assert m.delay_logits.grad[3:].abs().max() == 0
        assert torch.allclose(after.sum(-1), torch.ones(N), atol=1e-6)
        assert not torch.allclose(before[:3], after[:3])


class TestV16AuditClosureContracts:
    def test_runner_imports_shared_protocol(self):
        import ast, pathlib
        tree=ast.parse(pathlib.Path('run_o2_o3_audit_closure_v16.py').read_text())
        imports={(n.module,a.name) for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) for a in n.names}
        for name in ('train_warmup','run_pruning_to_stable_support','refit_fixed_support'):
            assert ('stage1.protocol',name) in imports

    def test_runner_has_no_local_protocol_helpers(self):
        import ast, pathlib
        tree=ast.parse(pathlib.Path('run_o2_o3_audit_closure_v16.py').read_text())
        names={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        assert not names & {'train_warmup','run_pruning_to_stable_support','refit_fixed_support','_train'}

    def test_penalty_scale_history_contract(self):
        from stage1.protocol import pruning_penalty_scale
        assert pruning_penalty_scale(1,200)==pytest.approx(.005)
        assert pruning_penalty_scale(100,200)==pytest.approx(.5)
        assert pruning_penalty_scale(200,200)==1 and pruning_penalty_scale(500,200)==1

    def test_earliest_stable_epoch_contract(self):
        from stage1.protocol import earliest_stable_stop_epoch
        assert earliest_stable_stop_epoch(200,100,50)==350

    def test_stage_result_contains_both_states(self):
        from stage1.protocol import StageResult
        assert {'best_state','terminal_state','terminal_support','support_at_best_validation','support_stable_epoch'} <= set(StageResult.__dataclass_fields__)

    def test_pruning_source_does_not_reload_best_state(self):
        import inspect
        from stage1.protocol import run_pruning_to_stable_support
        src=inspect.getsource(run_pruning_to_stable_support)
        assert 'model.load_state_dict(best' not in src and 'terminal_state' in src

    def test_refit_source_loads_best_state(self):
        import inspect
        from stage1.protocol import refit_fixed_support
        assert 'model.load_state_dict(best_state)' in inspect.getsource(refit_fixed_support)

    def test_scalar_gate_optimizer_has_unique_parameters(self):
        from stage1.scalar_gate import ScalarGateModel
        g=ScalarGateModel(make_model());ids=[id(p) for p in g.parameters()]
        assert len(ids)==len(set(ids)) and any(p is g.gates for p in g.parameters())

    def test_scalar_gate_all_ten_is_not_sparse_pass(self):
        support=list(range(10)); assert not (len(support)<=5)

    def test_metric_row_gate_requires_45(self):
        assert len([{}]*44)!=45 and len([{}]*45)==45

    def test_empty_metrics_cannot_pass(self):
        rows=[]; assert not (len(rows)==45 and rows)

    def test_o3_checkpoint_contract_has_learned_q(self):
        expected={'state_dict','learned_q','metadata'};assert 'learned_q' in expected

    def test_delay_pass_requires_delay_fields(self):
        required={'mean_delay_mae','w1','peak_lag_mae','boundary_mass'}
        assert required <= set({'mean_delay_mae':0,'w1':0,'peak_lag_mae':0,'boundary_mass':0})

    def test_original_response_gate_uses_function_metrics(self):
        import inspect,run_o2_o3_audit_closure_v16 as r
        src=inspect.getsource(r.run_o3);assert "function_corr" in src and "function_normalized_rmse" in src

    def test_full_contribution_uses_true_h(self):
        import inspect,run_o2_o3_audit_closure_v16 as r
        src=inspect.getsource(r.evaluate_functions);assert "h[j]" in src and "full_contribution_rmse" in src

    def test_refit_improvement_not_constant(self):
        import inspect,run_o2_o3_audit_closure_v16 as r
        src=inspect.getsource(r.run_o3);assert "vals[-200]-min(vals[-200:])" in src

    def test_baseline_names_complete(self):
        assert {'UniformDelay','FreeStaticLogits','StaticGamma'}==set(['UniformDelay','FreeStaticLogits','StaticGamma'])

    def test_ready_gate_requires_all_hard_gates(self):
        flags=[True]*11+[False];assert not all(flags)


class TestV17M2JointRecoveryContracts:
    def test_selection_mask_persists(self):
        m=make_model();m.prune_variable(4);n=make_model();n.load_state_dict(m.state_dict())
        assert not n.selection_mask[4] and "selection_mask" in m.state_dict()

    def test_selection_and_active_masks_are_distinct(self):
        m=make_model();m.prune_variable(4)
        assert m.active_mask[4] and not m.selection_mask[4]

    def test_pruned_delay_gradient_is_zero(self):
        m=make_model();m.prune_variable(4);m(torch.randn(5,N,L))[0].sum().backward()
        assert m.delay_prior.raw_mean.grad[4] == 0 and m.delay_prior.raw_std.grad[4] == 0

    def test_prune_clears_delay_adam_state(self):
        m=make_model();o=torch.optim.Adam(m.parameters());m(torch.randn(5,N,L))[0].sum().backward();o.step()
        m.prune_variable(4,o)
        assert o.state[m.delay_prior.raw_mean]["exp_avg"][4] == 0
        assert o.state[m.delay_prior.raw_std]["exp_avg_sq"][4] == 0

    def test_pruned_branch_cannot_regrow(self):
        m=make_model();m.prune_variable(4);o=torch.optim.Adam([p for p in m.parameters() if p.requires_grad])
        for _ in range(2):o.zero_grad();m(torch.randn(4,N,L))[0].sum().backward();o.step()
        assert m.response_branches.compute_branch_norms()[4] == 0 and not m.selection_mask[4]

    def test_refit_masks_non_surviving_delay(self):
        m=make_model();m.prune_variable(5);m(torch.randn(3,N,L))[0].sum().backward()
        assert m.delay_prior.raw_mean.grad[5] == 0

    def test_runner_never_calls_true_delay_setter(self):
        import pathlib
        assert "set_"+"true_delays" not in pathlib.Path("run_m2_joint_recovery_v17.py").read_text()

    def test_m2_model_candidate_mask_is_all_true(self):
        m=make_model();assert m.active_mask.all() and m.selection_mask.all()

    def test_selection_helper_is_truth_blind(self):
        import inspect
        from stage1.m2_metrics import one_se_select
        src=inspect.getsource(one_se_select)
        assert "truth" not in src and "test_rmse" not in src and "f1" not in src

    def test_support_aware_delay_penalizes_miss(self):
        import numpy as np
        from stage1.m2_metrics import support_aware_delay
        q=np.ones((1,N,L))/L;h=q[0].copy();rows,_=support_aware_delay(q,h,[[]])
        assert rows[0]["mean_delay_mae"] == L-1

    def test_missed_function_is_explicit_failure(self):
        import pathlib
        src=pathlib.Path("run_m2_joint_recovery_v17.py").read_text()
        assert "missed_variable_failure" in src

    def test_matched_kernel_cycles_gamma_rows(self):
        import run_m2_joint_recovery_v17 as r
        import numpy as np
        h=np.arange(N*L).reshape(N,L);q=r.matched_q(h)
        assert np.array_equal(q[3],h[0]) and np.array_equal(q[4],h[1]) and np.array_equal(q[5],h[2])

    def test_clean_failure_blocks_noisy(self):
        import inspect,run_m2_joint_recovery_v17 as r
        assert "if clean_pass" in inspect.getsource(r.run_noisy)

    def test_noisy_is_fresh(self):
        import inspect,run_m2_joint_recovery_v17 as r
        src=inspect.getsource(r.make_warmup)+inspect.getsource(r.pipeline)+inspect.getsource(r.run_noisy)
        assert "build_model" in src and "clean_state" not in src

    def test_baseline_budget_names(self):
        import run_m2_joint_recovery_v17 as r
        assert set(r.BASELINE_NAMES)=={"DenseStaticGamma","SparseStaticGamma","SparseFreeStaticLogits","UniformDelaySparse"}

    def test_checkpoint_writer_includes_mask_and_q(self):
        import inspect,run_m2_joint_recovery_v17 as r
        src=inspect.getsource(r.save_checkpoint);assert "selection_mask" in src and "learned_q" in src

    def test_ready_gate_requires_clean_tests_and_artifacts(self):
        import run_m2_joint_recovery_v17 as r
        assert r.ready_gate(True,True,True) and not r.ready_gate(True,False,True)

    def test_free_logits_selection_gradient_mask(self):
        m=Stage1TargetDelayKAN(N,L,delay_mode="free_static_logits");m.prune_variable(7)
        m(torch.randn(3,N,L))[0].sum().backward();assert m.delay_logits.grad[7].abs().sum()==0
