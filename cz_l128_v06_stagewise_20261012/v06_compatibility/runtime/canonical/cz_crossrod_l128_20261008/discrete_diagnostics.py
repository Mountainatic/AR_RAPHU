"""Offline spectra and synthetic counterfactuals; never imported by training.

All estimates are descriptive. No whiteness/grid acceptance or route rules.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import signal, linalg
from .discrete_audit import observation_rows

SEED = 20261010
QUANTIZER_TOLERANCE_IN_GRID_UNITS = 1e-10


def quantize_half_up(x, delta):
    """Half-up (including negative ties), with explicit grid-unit tie tolerance."""
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Positive grid required')
    z = np.asarray(x, dtype=float) / delta + .5
    nearest = np.rint(z)
    z = np.where(np.abs(z-nearest) <= QUANTIZER_TOLERANCE_IN_GRID_UNITS, nearest, z)
    return delta * np.floor(z)


def grid_candidate_evidence(values, delta=.01, tolerance=1e-8):
    """Fraction near absolute zero-origin grid; does not identify instrument ADC."""
    stored = np.asarray(values)
    stored_dtype = str(stored.dtype)
    finite = np.isfinite(stored)
    # Frozen C1 may store float32. Half an ULP is representation uncertainty,
    # not a relaxed measurement/ADC claim; retain the raw distance as evidence.
    uncertainty = np.abs(np.spacing(stored[finite]))*.5 if stored.dtype.kind=='f' else np.zeros(int(finite.sum()))
    values = np.asarray(stored[finite], float)
    if not len(values):
        return dict(status='NOT_AVAILABLE', finite_rows=0)
    error = np.abs(values/delta - np.rint(values/delta)) * delta
    precision_evidence='DECLARED_ARRAY_DTYPE'
    # Some frozen workbook values are float64 containing exact promoted
    # binary32 values. Report this empirical compatibility separately from
    # strict decimal-grid membership; do not claim an instrument precision.
    binary32=values.astype(np.float32)
    binary32_compatible=bool(np.array_equal(values,binary32.astype(np.float64)))
    if stored.dtype==np.dtype('float64') and binary32_compatible:
        uncertainty=np.abs(np.spacing(binary32)).astype(float)*.5
        precision_evidence='EMPIRICAL_EXACT_BINARY32_ROUNDTRIP_NOT_INSTRUMENT_EVIDENCE'
    allowed = tolerance+uncertainty
    fraction = float(np.mean(error <= allowed))
    return dict(output_record_grid_candidate=delta, grid_evidence_status='observed_grid_candidate' if fraction>=.999 else 'unknown',
        algorithm='ABSOLUTE_ZERO_ORIGIN_NEAREST_GRID_DISTANCE_WITH_STORED_HALF_ULP', tolerance_recorded_unit=tolerance,
        stored_dtype=stored_dtype,maximum_representation_uncertainty=float(uncertainty.max()),
        fraction_strict_decimal_grid=float(np.mean(error<=tolerance)),
        exact_binary32_roundtrip_compatible=binary32_compatible,precision_evidence=precision_evidence,
        finite_rows=len(values), fraction_on_candidate_grid=fraction, maximum_grid_distance=float(error.max()),
        instrument_verified=False, identifiability='Formatting/rounding/processing also compatible; not ADC evidence')


def step_distribution(values, delta=.01):
    d = np.diff(np.asarray(values, float))
    d = d[np.isfinite(d)]
    tol = max(1e-10, delta*1e-8)
    counts = {name: int(np.sum(np.isclose(d, value, atol=tol, rtol=0)))
              for name, value in [('zero', 0.), ('plus_delta', delta), ('minus_delta', -delta)]}
    counts['other'] = int(len(d)-sum(counts.values()))
    return dict(steps=len(d), delta=delta, counts=counts,
        frequencies={k: v/len(d) if len(d) else None for k, v in counts.items()})


def spectrum_acf(values, sample_period_seconds, *, maximum_lag=64, detrend='constant'):
    x = np.asarray(values, float)
    if x.ndim != 1 or not np.isfinite(x).all() or sample_period_seconds <= 0:
        raise ValueError('Finite regular scalar segment and positive cadence required')
    n = len(x)
    if n < 4:
        return dict(status='SHORT_SEGMENT', N=n, unsupported_lags_above=max(0,n-1)), None, None
    fs = 1/sample_period_seconds
    frequency, density = signal.periodogram(x, fs=fs, window='boxcar', detrend=detrend, scaling='density')
    centered = signal.detrend(x, type=detrend) if detrend in {'constant', 'linear'} else x.copy()
    covariance = signal.correlate(centered, centered, mode='full', method='fft')[n-1:]/n
    lag_count = min(maximum_lag, n//4)
    if covariance[0] <= np.finfo(float).tiny:
        acf = np.full(lag_count+1, np.nan); acf[0] = 1
        pacf = np.full(lag_count+1, np.nan); pacf[0] = 1
        dependence_status = 'ZERO_VARIANCE_ACF_PACF_UNDEFINED'
    else:
        acf = covariance[:lag_count+1]/covariance[0]
        pacf = np.zeros(lag_count+1); pacf[0] = 1
        for lag in range(1, lag_count+1):
            try:
                pacf[lag] = linalg.solve_toeplitz((acf[:lag], acf[:lag]), acf[1:lag+1])[-1]
            except linalg.LinAlgError:
                pacf[lag] = np.nan
        dependence_status = 'DESCRIPTIVE_BIASED_ACF_YULE_WALKER_PACF'
    usable = frequency > 0
    peak = int(np.flatnonzero(usable)[np.argmax(density[usable])])
    near = (frequency >= .133) & (frequency <= .153)
    near_index = int(np.flatnonzero(near)[np.argmax(density[near])]) if near.any() else None
    df = fs/n
    total_energy = float(density.sum()*df)
    summary = dict(status='OK', N=n, fs_Hz=fs, sample_period_seconds=sample_period_seconds,
        frequency_resolution_Hz=df, detrend=detrend, window='boxcar', max_supported_reported_lag=lag_count,
        unsupported_lags_above=lag_count, acf_pacf_status=dependence_status,
        peak_frequency_Hz=float(frequency[peak]), peak_density=float(density[peak]),
        annotation_band_Hz=[.133,.153], near_0143_peak_Hz=float(frequency[near_index]) if near_index is not None else None,
        near_0143_density=float(density[near_index]) if near_index is not None else None,
        near_0143_band_energy=float(density[near].sum()*df), total_energy=total_energy,
        near_0143_energy_fraction=float(density[near].sum()*df/total_energy) if total_energy else None,
        iid_acf_reference_band=1.96/np.sqrt(n), reference_band_role='DESCRIPTIVE_ONLY_NOT_ACCEPTANCE',
        acf_lag1=float(acf[1]) if len(acf)>1 and np.isfinite(acf[1]) else None)
    return summary, pd.DataFrame(dict(frequency_Hz=frequency, density=density)), pd.DataFrame(
        dict(lag_samples=np.arange(lag_count+1), lag_seconds=np.arange(lag_count+1)*sample_period_seconds, acf=acf, pacf=pacf))


def segmented_diagnostics(data, spec, *, value_columns=('recorded_level',), maximum_lag=64, detrend='constant'):
    """Never concatenate runs, including finite-value holes and skipped origins."""
    rows = observation_rows(data, spec)
    output = []
    for key, group in rows.groupby('diagnostic_segment_id', sort=False):
        for column in value_columns:
            if column not in group:
                output.append((dict(status='NOT_AVAILABLE', quantity=column, diagnostic_segment_id=int(key)), None, None))
                continue
            finite = np.isfinite(group[column].to_numpy(float))
            starts = np.flatnonzero(finite & np.r_[True, ~finite[:-1]])
            ends = np.flatnonzero(finite & np.r_[~finite[1:], True])+1
            if not len(starts):
                output.append((dict(status='NOT_AVAILABLE', quantity=column, diagnostic_segment_id=int(key)), None, None))
            for subrun, (start, end) in enumerate(zip(starts, ends)):
                run = group.iloc[start:end]
                if spec.sample_period_seconds is None:
                    summary, psd, acf = dict(status='CADENCE_NOT_AVAILABLE', N=len(run)), None, None
                else:
                    summary, psd, acf = spectrum_acf(run[column], spec.sample_period_seconds,
                        maximum_lag=maximum_lag, detrend=detrend)
                summary.update(quantity=column, entity_id=str(run.entity_id.iloc[0]),
                    segment_id=str(run.segment_id.iloc[0]), split_id=str(run.split_id.iloc[0]),
                    diagnostic_segment_id=int(key), finite_subrun=subrun,
                    raw_index_start=int(run.raw_index.iloc[0]), raw_index_end=int(run.raw_index.iloc[-1]),
                    timestamp_evidence=str(run.timestamp_evidence.iloc[0]))
                output.append((summary, psd, acf))
    return output


def synthetic_scenes(n=4096, sample_period=2., seed=SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(n)*sample_period
    delta = .01
    scenes = {}
    s1 = {}
    for label, ratio in [('1_over_7',1/7), ('2_over_7',2/7), ('3_over_8',3/8), ('sqrt2_over_10',np.sqrt(2)/10)]:
        s = .1234+delta*ratio*np.arange(n)
        q = quantize_half_up(s,delta)
        s1.update({label+'_recorded_level_unrounded':s, label+'_recorded_level_quantized':q,
            label+'_adjacent_difference_unrounded':np.diff(s), label+'_adjacent_difference_quantized':np.diff(q)})
    scenes['S1'] = (dict(N=n, Ts_seconds=sample_period, delta=delta, no_sine=True,
        ratios=[1/7,2/7,3/8,float(np.sqrt(2)/10)], q7_fundamental_seconds=14., seven_seconds_is_possible_harmonic=True), s1)
    noise = rng.normal(0,.0025,n)
    slow = .1234+.00003*t+.007*np.sin(2*np.pi*t/700.)
    measured = slow+noise
    q = quantize_half_up(measured,delta)
    scenes['S2'] = (dict(N=n,Ts_seconds=sample_period,delta=delta,noise_sd=.0025,
        noise_empirical_mean=float(noise.mean()), quantization_error_iid_claim=False),
        dict(recorded_level_unrounded=measured,recorded_level_quantized=q,
            adjacent_difference_unrounded=np.diff(measured),adjacent_difference_quantized=np.diff(q)))
    s3={}
    dense_t=np.arange(0,n*sample_period,.2)
    for f in [1/7, .137, .149]:
        tag=format(f,'.6f')
        before=.1234+.003*np.sin(2*np.pi*f*dense_t)
        sampled=.1234+.003*np.sin(2*np.pi*f*t)
        rounded=quantize_half_up(sampled,delta)
        s3.update({tag+'_before_sampling':before,tag+'_sampled':sampled,tag+'_quantized':rounded,
            tag+'_adjacent_difference':np.diff(rounded)})
    scenes['S3']=(dict(N=n,Ts_seconds=sample_period,delta=delta,before_sampling_Ts_seconds=.2,
        injected_frequencies_Hz=[1/7,.137,.149],amplitude=.003, proves_quantization_root=False),s3)
    f0=.357142857;fs=1/sample_period;alias=abs((f0+fs/2)%fs-fs/2)
    sampled=.1234+.003*np.sin(2*np.pi*f0*t)
    rounded=quantize_half_up(sampled,delta)
    scenes['S4']=(dict(N=n,Ts_seconds=sample_period,delta=delta,f0_Hz=f0,fs_Hz=fs,
        alias_Hz=alias,measured_sensor_high_frequency_claim=False,before_sampling_Ts_seconds=.2),
        dict(before_sampling=.1234+.003*np.sin(2*np.pi*f0*dense_t),sampled=sampled,
            quantized=rounded,adjacent_difference=np.diff(rounded)))
    return scenes


def run_synthetic(output):
    """Four figures + exact arrays/summaries + independent N/seed comparisons."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    all_summaries=[];parameters={}
    for scene,(params,arrays) in synthetic_scenes().items():
        np.savez_compressed(output/(scene+'.npz'),**arrays)
        parameters[scene]=params
        fig, axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for name, x in arrays.items():
            cadence=.2 if 'before_sampling' in name else params['Ts_seconds']
            curve_label=name.replace('recorded_level_','level_').replace('adjacent_difference_','diff_')+f' (N={len(x)}, fs={1/cadence:g} Hz)'
            summary,psd,acf=spectrum_acf(x,cadence)
            summary.update(scene=scene,quantity=name,seed=SEED,
                step_distribution=step_distribution(x) if 'difference' not in name else None,
                array_sha256=hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest())
            all_summaries.append(summary)
            if psd is not None:
                psd.to_csv(output/(scene+'_'+name+'_PSD.csv'),index=False)
                acf.to_csv(output/(scene+'_'+name+'_ACF_PACF.csv'),index=False)
                axes[0,1].plot(psd.frequency_Hz,psd.density,label=curve_label,alpha=.7)
                axes[1,0].plot(acf.lag_seconds,acf.acf,label=name,alpha=.7)
            axes[0,0].plot(np.arange(min(80,len(x)))*cadence,x[:80],label=name,alpha=.7)
            if 'difference' in name:
                steps,counts=np.unique(np.round(x,10),return_counts=True)
                axes[1,1].plot(steps,counts/len(x),'o',label=name,alpha=.7)
        axes[0,0].set(xlabel='Time (s)',ylabel='Recorded value / difference (unverified unit)')
        axes[0,1].set(xlabel='Frequency (Hz)',ylabel='PSD (recorded unit squared / Hz)',yscale='log')
        axes[0,1].axvline(1/7,color='black',linestyle=':',label='1/7 Hz annotation only')
        if scene in {'S3','S4'}:
            axes[0,1].set_xlim(0,.55) # retain injected f0 and sampled alias, suppress uninformative dense Nyquist tail
        axes[1,0].set(xlabel='Lag (s)',ylabel='Descriptive ACF')
        axes[1,1].set(xlabel='Adjacent recorded step (unit)',ylabel='Frequency')
        for ax in axes.flat:
            ax.grid(alpha=.2)
        axes[0,1].legend(fontsize=4,ncol=2)
        pre='; pre: N=40960, fs=5 Hz' if scene in {'S3','S4'} else ''
        fig.suptitle(f'{scene}: level N=4096, diff N=4095, fs=0.5 Hz{pre}\nGrid delta=0.01, half-up; constant detrend',fontsize=11)
        fig.savefig(output/(scene+'.png'),dpi=130);plt.close(fig)
    robustness=[]
    for n in [2048,4096,8192]:
        for seed in [SEED,SEED+1]:
            for scene,(params,arrays) in synthetic_scenes(n,seed=seed).items():
                for name,x in arrays.items():
                    if 'before_sampling' in name:
                        continue
                    summary,_,_=spectrum_acf(x,2.)
                    robustness.append(dict(scene=scene,quantity=name,N=len(x),seed=seed,
                        near_0143_peak_Hz=summary.get('near_0143_peak_Hz'),
                        peak_frequency_Hz=summary.get('peak_frequency_Hz'),acf_lag1=summary.get('acf_lag1')))
    pd.DataFrame(robustness).to_csv(output/'N_SEED_CONTROLS.csv',index=False)
    (output/'PARAMETERS.json').write_text(json.dumps(dict(seed=SEED,scenes=parameters,
        quantizer_tie_rule='half-up floor(x/delta+0.5)',
        quantizer_tolerance_grid_units=QUANTIZER_TOLERANCE_IN_GRID_UNITS),indent=2))
    (output/'SUMMARY.json').write_text(json.dumps(all_summaries,indent=2,allow_nan=False))
    return all_summaries
