#!/usr/bin/env python
"""Synthetic data generator: canonical physical lag, truth_functions.py, double-precision."""
import numpy as np
import torch
from scipy.linalg import cholesky

class SyntheticDataGenerator:
    def __init__(self, n_active=3, n_inactive=7, max_lag=32, n_samples=5000,
                 noise_std=0.1, seed=42, scenario='S1_gamma',
                 ar_rho=0.7, ar_cross_corr=0.1, burn_in=2000):
        self.n_active = n_active; self.n_inactive = n_inactive
        self.n_vars = n_active + n_inactive; self.max_lag = max_lag
        self.n_samples = n_samples; self.noise_std = noise_std
        self.seed = seed; self.scenario = scenario
        self.ar_rho = ar_rho; self.ar_cross_corr = ar_cross_corr
        self.burn_in = burn_in
        self.rng = np.random.RandomState(seed)
        Sigma = (1.0 - ar_cross_corr) * np.eye(self.n_vars) + ar_cross_corr * np.ones((self.n_vars, self.n_vars))
        self.L_chol = cholesky(Sigma, lower=True)

    def generate(self, return_debug=False):
        return getattr(self, f'_gen_{self.scenario}')(return_debug=return_debug)

    def _gen_base_data(self):
        n_vars, rho = self.n_vars, self.ar_rho
        assert abs(rho) < 0.99
        n_total = self.n_samples + self.max_lag + self.burn_in
        raw = np.zeros((n_total, n_vars), dtype=np.float64)
        for t in range(1, n_total):
            eps = self.rng.randn(n_vars).astype(np.float64)
            raw[t] = rho*raw[t-1] + np.sqrt(max(1-rho*rho,0))*(self.L_chol@eps)
        raw = raw[self.burn_in:].copy()
        assert np.all(np.isfinite(raw))
        return raw

    def _make_windows(self, raw):
        N, L = len(raw), self.max_lag
        X64 = np.zeros((N-L, self.n_vars, L), dtype=np.float64)
        for i in range(N-L): X64[i] = raw[i:i+L].T.copy()
        X32 = X64.astype(np.float32)
        return X32, X64

    def _build_truth(self, raw, X64, X32, Y_clean, Y_obs, true_h_canonical,
                      active_vars, extra=None, return_debug=False):
        B, L = len(X64), self.max_lag
        xl64 = X64[:,:,::-1].copy(); xl32 = X32[:,:,::-1].copy()
        truth = {
            "active_vars": list(active_vars),
            "active_mask": [int(j in active_vars) for j in range(self.n_vars)],
            "true_bias": 0.0, "horizon": 0, "lag_order": "current_to_past",
            "true_h_canonical_float64": true_h_canonical.astype(np.float64),
            "true_h_canonical_float32": true_h_canonical.astype(np.float32),
            "true_h_window_order_float64": true_h_canonical[:,::-1].astype(np.float64),
            "scenario": self.scenario,
            "window_raw_indices": np.array([[i+k for k in range(L)] for i in range(B)], dtype=np.int32),
            "target_raw_indices": np.array([i+L-1 for i in range(B)], dtype=np.int32),
        }
        if extra: truth.update(extra)
        if return_debug:
            truth["raw_float64"] = raw.astype(np.float64)
            truth["X_window_float64"] = X64.astype(np.float64)
            truth["X_window_float32"] = X32.astype(np.float64)
            truth["x_lag_float64"] = xl64.astype(np.float64)
            truth["x_lag_float32"] = xl32.astype(np.float64)
            truth["y_clean_float64"] = Y_clean.astype(np.float64)
            truth["y_observed_float64"] = Y_obs.astype(np.float64)
        return truth

    def _gen_static(self, true_h_canonical, active_vars, func_ids, noise=None, return_debug=False):
        from stage1.truth_functions import get_true_function, TRUE_FUNCTION_NAMES
        raw = self._gen_base_data()
        X32, X64 = self._make_windows(raw)
        xl64 = X64[:,:,::-1].copy()
        B = len(X64); ns = noise if noise is not None else self.noise_std
        Y_clean = np.zeros(B, dtype=np.float64)
        for b in range(B):
            y = 0.0
            for j in active_vars:
                y += np.sum(true_h_canonical[j]*get_true_function(j)(xl64[b,j]))
            Y_clean[b] = y
        Y_obs = Y_clean + self.rng.randn(B)*ns*np.std(Y_clean)
        assert np.all(np.isfinite(X32)) and np.all(np.isfinite(Y_obs))
        truth = self._build_truth(raw, X64, X32, Y_clean, Y_obs, true_h_canonical,
            active_vars, {"function_names":{str(j):TRUE_FUNCTION_NAMES[j] for j in active_vars}},
            return_debug=return_debug)
        return X32, Y_obs.astype(np.float32).reshape(-1,1), truth

    def _make_gamma(self, a, b):
        L = self.max_lag; lr = (a-1)*np.log(np.arange(L)+1e-3)-(np.arange(L)+1e-3)/b
        lr -= lr.max(); h = np.exp(lr); return h/h.sum()

    def _gen_S0_oracle(self, return_debug=False):
        th = np.zeros((self.n_vars, self.max_lag), dtype=np.float64)
        for j,(a,b) in enumerate([(4.0,1.8),(5.0,2.0),(6.0,2.5)]): th[j] = self._make_gamma(a,b)
        return self._gen_static(th,[0,1,2],[0,1,2],noise=0.05,return_debug=return_debug)

    def _gen_S1_gamma(self, return_debug=False):
        th = np.zeros((self.n_vars, self.max_lag), dtype=np.float64)
        for j in range(min(3,self.n_active)): th[j] = self._make_gamma(3+j,5+j*2)
        return self._gen_static(th,list(range(min(3,self.n_active))),
            [0,1,2][:min(3,self.n_active)],return_debug=return_debug)

    def _gen_S2_exponential(self, return_debug=False):
        L=self.max_lag;th=np.zeros((self.n_vars,L),dtype=np.float64)
        lags=np.arange(L,dtype=np.float64)
        for j in range(min(3,self.n_active)):
            b=2.0+j*1.5;hr=-lags/b;hr-=hr.max();h=np.exp(hr);th[j]=h/h.sum()
        return self._gen_static(th,list(range(min(3,self.n_active))),
            [0,1,2][:min(3,self.n_active)],return_debug=return_debug)

    def _gen_S3_near_pure_delay(self, return_debug=False):
        L=self.max_lag;th=np.zeros((self.n_vars,L),dtype=np.float64)
        lags=np.arange(L,dtype=np.float64)
        for j in range(min(3,self.n_active)):
            c=[5,10,20][j];hr=-0.5*(lags-c)**2;hr-=hr.max();h=np.exp(hr);th[j]=h/h.sum()
        return self._gen_static(th,list(range(min(3,self.n_active))),
            [0,1,2][:min(3,self.n_active)],return_debug=return_debug)

    def _gen_S4_correlated_distractor(self, return_debug=False):
        th=np.zeros((self.n_vars,self.max_lag),dtype=np.float64)
        th[0]=self._make_gamma(4.0,1.8)
        return self._gen_static(th,[0],[0],return_debug=return_debug)

    def _gen_S5_mild_dynamic_delay(self, return_debug=False):
        from stage1.truth_functions import get_true_function, TRUE_FUNCTION_NAMES
        raw=self._gen_base_data();X32,X64=self._make_windows(raw)
        xl64=X64[:,:,::-1].copy();B=len(X64);L=self.max_lag
        lags=np.arange(L,dtype=np.float64);base=10.0
        th=np.zeros((self.n_vars,L),dtype=np.float64)
        Yc=np.zeros(B,dtype=np.float64)
        for b in range(B):
            c=base+2*np.tanh(float(X32[b,0,-1]))
            hr=-0.3*(lags-c)**2;hr-=hr.max();h=np.exp(hr);h/=h.sum()
            Yc[b]=np.sum(h*get_true_function(0)(xl64[b,0]))
        Yo=Yc+self.rng.randn(B)*self.noise_std*np.std(Yc)
        truth=self._build_truth(raw,X64,X32,Yc,Yo,th,[0],
            {"function_names":{"0":TRUE_FUNCTION_NAMES[0]},"note":"Dynamic delay"},
            return_debug=return_debug)
        return X32,Yo.astype(np.float32).reshape(-1,1),truth

    def _gen_S6_signed_oscillatory_violation(self, return_debug=False):
        from stage1.truth_functions import get_true_function, TRUE_FUNCTION_NAMES
        raw=self._gen_base_data();X32,X64=self._make_windows(raw)
        xl64=X64[:,:,::-1].copy();B=len(X64);L=self.max_lag
        lags=np.arange(L,dtype=np.float64)
        h_osc=np.exp(-0.1*lags)*np.cos(0.5*lags)
        th=np.zeros((self.n_vars,L),dtype=np.float64);th[0]=h_osc
        Yc=np.zeros(B,dtype=np.float64)
        for b in range(B):Yc[b]=np.sum(h_osc*get_true_function(0)(xl64[b,0]))
        Yo=Yc+self.rng.randn(B)*self.noise_std*np.std(Yc)
        truth=self._build_truth(raw,X64,X32,Yc,Yo,th,[0],
            {"function_names":{"0":TRUE_FUNCTION_NAMES[0]},"note":"Signed kernel violation"},
            return_debug=return_debug)
        return X32,Yo.astype(np.float32).reshape(-1,1),truth

    def _gen_default(self, return_debug=False):
        return self._gen_S1_gamma(return_debug=return_debug)


def generate_all_scenarios():
    scenarios = {}
    for s in ['S1_gamma','S2_exponential','S3_near_pure_delay',
              'S4_correlated_distractor','S5_mild_dynamic_delay',
              'S6_signed_oscillatory_violation']:
        gen = SyntheticDataGenerator(scenario=s, n_samples=2000, max_lag=32, noise_std=0.1)
        X, Y, truth = gen.generate()
        scenarios[s] = (X, Y, truth)
    return scenarios
