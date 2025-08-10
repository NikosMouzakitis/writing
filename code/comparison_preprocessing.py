import numpy as np
import matplotlib.pyplot as plt
import pywt
import mne
from scipy.stats import median_abs_deviation
from scipy.signal import welch
from scipy import signal

NUM_CHAN = 6
PLOT_START = 40
PLOT_END = 44

def load_eeg_data(edf_path, num_channels=NUM_CHAN, l_freq=0.5, h_freq=60.0):
    try:
        temp_raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        unique_ch_names = list(dict.fromkeys(temp_raw.ch_names))
        channels = unique_ch_names[:num_channels]
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick(channels)
        raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir',
                 iir_params=dict(order=4, ftype='butter'),
                 phase='zero', verbose=True)
        print(f"Applied bandpass filter ({l_freq}–{h_freq} Hz)")
        return raw, channels, raw.info['sfreq']
    except Exception as e:
        print(f"Error loading EDF file: {str(e)}")
        return None, None, None

def wavelet_denoise(signal, wavelet='sym4', mode='hard', level=7, threshold_scale=2.0):
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = median_abs_deviation(coeffs[-1]) / 0.6745
        threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs[1] = np.zeros_like(coeffs[1])
        coeffs[2:] = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[2:]]
        return pywt.waverec(coeffs, wavelet)
    except Exception as e:
        print(f"Error in wavelet denoising: {e}")
        return signal

def bandpass_filter(signal, sfreq, l_freq=1.0, h_freq=60.0):
    try:
        raw_temp = mne.io.RawArray(signal[np.newaxis, :], 
                                  mne.create_info(['temp'], sfreq, ch_types='eeg'))
        raw_temp.filter(l_freq=l_freq, h_freq=h_freq, method='iir',
                       iir_params=dict(order=4, ftype='butter'),
                       phase='zero', verbose=True)
        return raw_temp.get_data()[0]
    except Exception as e:
        print(f"Error in bandpass filtering: {e}")
        return signal

def calculate_residual_snr(raw_signal, denoised_signal):
    try:
        residual = raw_signal - denoised_signal
        signal_power = np.mean(denoised_signal ** 2)
        noise_power = np.mean(residual ** 2)
        return 10 * np.log10(signal_power / (noise_power + 1e-12))
    except Exception as e:
        print(f"Error in SNR calculation: {e}")
        return np.nan

def calculate_rmse(original, processed):
    return np.sqrt(np.mean((original - processed)**2))

def calculate_nrmse(original, processed):
    rmse = calculate_rmse(original, processed)
    signal_range = np.max(original) - np.min(original)
    return (rmse / signal_range) * 100

def calculate_correlation(original, processed):
    return np.corrcoef(original, processed)[0, 1]

def calculate_prd(original, processed):
    numerator = np.sum((original - processed)**2)
    denominator = np.sum(original**2)
    return 100 * np.sqrt(numerator / denominator)

def calculate_all_metrics(original, processed):
    return {
        'SNR (dB)': calculate_residual_snr(original, processed),
        'RMSE (μV)': calculate_rmse(original, processed),
        'NRMSE (%)': calculate_nrmse(original, processed),
        'Correlation': calculate_correlation(original, processed),
        'PRD (%)': calculate_prd(original, processed)
    }

def plot_eeg_signals(raw, channels, sfreq, plot_start=PLOT_START, plot_end=PLOT_END):
    try:
        full_data = raw.get_data(picks=channels)
        start_sample = int(plot_start * sfreq)
        stop_sample = int(plot_end * sfreq)
        times = np.arange(start_sample, stop_sample) / sfreq + plot_start

        filters = [
            ('DB4 L2 T1', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=2, threshold_scale=1.0), sfreq, 1.0, 60.0)),
            ('DB4 L2 T2', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=2, threshold_scale=2.0), sfreq, 1.0, 60.0)),
            ('DB4 L2 T3', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=2, threshold_scale=3.0), sfreq, 1.0, 60.0)),
            ('DB4 L2 T4', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=2, threshold_scale=4.0), sfreq, 1.0, 60.0)),
            ('DB4 L2 T5', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=2, threshold_scale=5.0), sfreq, 1.0, 60.0)),

            ('DB4 L3 T1', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=3, threshold_scale=1.0), sfreq, 1.0, 60.0)),
            ('DB4 L3 T2', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=3, threshold_scale=2.0), sfreq, 1.0, 60.0)),
            ('DB4 L3 T3', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=3, threshold_scale=3.0), sfreq, 1.0, 60.0)),
            ('DB4 L3 T4', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=3, threshold_scale=4.0), sfreq, 1.0, 60.0)),
            ('DB4 L3 T5', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=3, threshold_scale=5.0), sfreq, 1.0, 60.0)),
        ]

        metrics_results = {}
        filtered_full_data = {}

        for filt_name, filt_func in filters:
            print(f"\nProcessing {filt_name}")
            filtered_full_data[filt_name] = np.zeros_like(full_data)
            metrics_results[filt_name] = {}

            for ch_idx, ch_name in enumerate(channels):
                original = full_data[ch_idx]
                processed = filt_func(original)
                filtered_full_data[filt_name][ch_idx] = processed
                metrics = calculate_all_metrics(original, processed)
                metrics_results[filt_name][ch_name] = metrics
                
                # Print metrics with proper units
                print(f"{ch_name}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.2f}")

            # Visualization
            plt.figure(figsize=(12, 8))
            plt.suptitle(f'EEG: Original vs {filt_name} ({plot_start}s–{plot_end}s)', y=1.02)
            
            for ch_idx, ch_name in enumerate(channels):
                plt.subplot(NUM_CHAN, 1, ch_idx+1)
                orig_segment = full_data[ch_idx, start_sample:stop_sample]
                filt_segment = filtered_full_data[filt_name][ch_idx, start_sample:stop_sample]
                
                plt.plot(times, orig_segment/np.std(orig_segment), 'b', alpha=0.5, label='Original')
                plt.plot(times, filt_segment/np.std(filt_segment), 'r', alpha=0.8, label='Processed')
                plt.ylabel(ch_name)
                
                # Format metrics for display
                metrics = metrics_results[filt_name][ch_name]
                metric_text = "\n".join([f"{k}: {v:.2f}" for k,v in metrics.items()])
                plt.text(0.02, 0.70, metric_text, transform=plt.gca().transAxes,
                         fontsize=7, bbox=dict(facecolor='white', alpha=0.7))
                
                if ch_idx == 0:
                    plt.legend(loc='upper right')
                if ch_idx == NUM_CHAN-1:
                    plt.xlabel('Time (s)')
                else:
                    plt.xticks([])
            
            plt.tight_layout()
            plt.show()

        return metrics_results

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return None

if __name__ == "__main__":
    EDF_PATH = "/home/nicko/implementations/THESIS/main-thesis-folder/physionet.org/files/chbmit/1.0.0/chb02/chb02_16.edf"
    raw, channels, sfreq = load_eeg_data(EDF_PATH)
    
    if raw is not None:
        print(f"\nProcessing {len(channels)} channels:")
        print(channels)
        
        all_metrics = plot_eeg_signals(raw, channels, sfreq)
        
        if all_metrics:
            print("\n=== FINAL METRICS SUMMARY ===")
            for filt_name, ch_metrics in all_metrics.items():
                print(f"\nFilter: {filt_name}")
                print("-"*50)
                for ch_name, metrics in ch_metrics.items():
                    print(f"\nChannel: {ch_name}")
                    for metric, value in metrics.items():
                        print(f"{metric:>12}: {value:.5f}")
        else:
            print("No metrics returned due to processing error")
    else:
        print("Failed to load EEG data")
