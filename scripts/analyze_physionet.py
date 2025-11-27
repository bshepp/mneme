#!/usr/bin/env python
"""
Comprehensive PhysioNet bioelectric analysis script.
Downloads and analyzes multiple datasets to validate Lyapunov spectrum implementation.
"""

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
import os

# Import Mneme tools
from mneme.core import compute_lyapunov_spectrum, kaplan_yorke_dimension
from mneme.core.attractors import classify_attractor_by_lyapunov, embed_trajectory


def analyze_ecg_hrv(signal, fs, description):
    """Extract HRV from ECG and compute Lyapunov spectrum."""
    nyq = fs / 2
    
    # Bandpass filter for QRS detection
    low = 5 / nyq
    high = min(15, nyq - 0.1) / nyq
    b_qrs, a_qrs = butter(4, [low, high], btype='band')
    ecg_qrs = filtfilt(b_qrs, a_qrs, signal)
    
    # Find R peaks
    peaks, _ = find_peaks(ecg_qrs, distance=int(0.5*fs), height=0.3*np.max(np.abs(ecg_qrs)))
    
    if len(peaks) < 50:
        return None
    
    # RR intervals in milliseconds
    rr_intervals = np.diff(peaks) / fs * 1000
    
    # Embed and analyze
    trajectory = embed_trajectory(rr_intervals, embedding_dimension=4, time_delay=1)
    dt_hrv = np.mean(rr_intervals) / 1000  # Average beat interval in seconds
    
    spectrum = compute_lyapunov_spectrum(trajectory, dt=dt_hrv, n_neighbors=10, orthog_interval=5)
    d_ky = kaplan_yorke_dimension(spectrum)
    atype = classify_attractor_by_lyapunov(spectrum)
    
    return {
        'n_beats': len(peaks),
        'mean_hr': 60000 / np.mean(rr_intervals),
        'rr_std': np.std(rr_intervals),
        'spectrum': spectrum,
        'd_ky': d_ky,
        'type': str(atype)
    }


def main():
    print('=' * 70)
    print('COMPREHENSIVE PHYSIONET BIOELECTRIC ANALYSIS')
    print('=' * 70)
    print()
    
    os.makedirs('data/raw/physionet', exist_ok=True)
    
    # Dataset configurations: (record, database, description, sampto)
    DATASETS = [
        ('100', 'mitdb', 'MIT-BIH Arrhythmia', 50000),
        ('101', 'mitdb', 'MIT-BIH Arrhythmia #2', 50000),
        ('16265', 'nsrdb', 'Normal Sinus Rhythm', 100000),
        ('16272', 'nsrdb', 'Normal Sinus Rhythm #2', 100000),
        ('chf01', 'chfdb', 'Congestive Heart Failure', 100000),
        ('chf02', 'chfdb', 'CHF Patient #2', 100000),
    ]
    
    results = []
    
    for record_name, db_name, description, sampto in DATASETS:
        print(f'Analyzing: {description} ({record_name})')
        print('-' * 50)
        
        try:
            # Download record
            record = wfdb.rdrecord(record_name, pn_dir=db_name, sampto=sampto)
            signal = record.p_signal[:, 0]
            fs = record.fs
            
            print(f'  Samples: {len(signal)}, Fs: {fs} Hz')
            
            # Bandpass filter ECG (0.5-40 Hz)
            nyq = fs / 2
            low = 0.5 / nyq
            high = min(40, nyq - 0.1) / nyq
            b, a = butter(4, [low, high], btype='band')
            signal_filtered = filtfilt(b, a, signal)
            signal_filtered = (signal_filtered - np.mean(signal_filtered)) / np.std(signal_filtered)
            
            # Analyze HRV
            result = analyze_ecg_hrv(signal_filtered, fs, description)
            
            if result:
                print(f'  Beats: {result["n_beats"]}, Mean HR: {result["mean_hr"]:.0f} bpm')
                print(f'  HRV (SDNN): {result["rr_std"]:.1f} ms')
                print(f'  Lyapunov: [{result["spectrum"][0]:.4f}, {result["spectrum"][1]:.4f}, ...]')
                print(f'  D_KY: {result["d_ky"]:.2f}, Type: {result["type"].split(".")[-1]}')
                
                results.append({
                    'dataset': description,
                    'record': record_name,
                    'database': db_name,
                    **result
                })
            else:
                print('  Not enough beats detected')
            
            print()
            
        except Exception as e:
            print(f'  ERROR: {e}')
            print()
    
    # Summary table
    print('=' * 70)
    print('SUMMARY: Lyapunov Analysis of Cardiac Data')
    print('=' * 70)
    print()
    print(f'{"Dataset":<30} {"HR":>6} {"SDNN":>8} {"L1":>9} {"L2":>9} {"D_KY":>6}  Type')
    print('-' * 90)
    
    for r in results:
        typename = r['type'].split('.')[-1]
        print(f'{r["dataset"]:<30} {r["mean_hr"]:>5.0f}  {r["rr_std"]:>6.1f}ms  '
              f'{r["spectrum"][0]:>+8.4f}  {r["spectrum"][1]:>+8.4f}  {r["d_ky"]:>5.2f}  {typename}')
    
    print()
    print('Interpretation:')
    print('  - Healthy hearts: L1 > 0 (chaotic), D_KY ~ 2-3')
    print('  - Heart failure: Often shows reduced chaos (less adaptive)')
    print('  - Higher SDNN = more variability = healthier')
    print()
    
    # Save results
    np.savez('data/raw/physionet/multi_dataset_results.npz', 
             results=results,
             summary={
                 'n_datasets': len(results),
                 'mean_lambda1': np.mean([r['spectrum'][0] for r in results]),
                 'mean_dky': np.mean([r['d_ky'] for r in results])
             })
    print('Results saved to data/raw/physionet/multi_dataset_results.npz')


if __name__ == '__main__':
    main()

