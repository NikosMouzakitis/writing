import numpy as np
import matplotlib.pyplot as plt
import pywt
import mne
from scipy.stats import median_abs_deviation
from scipy.signal import welch

NUM_CHAN = 6
PLOT_START = 40  # Start time in seconds
PLOT_END = 44
# End time in seconds

def load_eeg_data(edf_path, num_channels=NUM_CHAN, l_freq=0.5, h_freq=60.0):
    """Load EEG data, apply bandpass filter"""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick(raw.ch_names[:num_channels])
        sfreq = raw.info['sfreq']


        #application of a bandpass filter
        raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=dict(order=4, ftype='butter'), phase='zero', verbose=True)
        channels = raw.ch_names
        print(f"Applied bandpass filter ({l_freq}–{h_freq} Hz)")
        return raw, channels, sfreq
    except Exception as e:
        print(f"Error loading EDF file: {e}")
        return None, None, None

def wavelet_denoise(signal, wavelet='sym4', mode='hard', level=7, threshold_scale=2.0):
    """Wavelet denoising"""
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = median_abs_deviation(coeffs[-1]) / 0.6745
        threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs[1] = np.zeros_like(coeffs[1])  # Zero D1 (32–64 Hz)
        coeffs[2:] = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[2:]]
        print(f"Wavelet: {wavelet}, Threshold: {threshold:.2e}, Sigma: {sigma:.2e}")
        return pywt.waverec(coeffs, wavelet)
    except Exception as e:
        print(f"Error in wavelet denoising: {e}")
        return signal

def fir_denoise(signal, sfreq, l_freq=0.5, h_freq=30.0):
    """FIR bandpass filter"""
    try:
        raw_temp = mne.io.RawArray(signal[np.newaxis, :], mne.create_info(['temp'], sfreq, ch_types='eeg'))
        raw_temp.filter(l_freq=l_freq, h_freq=h_freq, method='fir', fir_design='firwin', phase='zero', filter_length='100', verbose=True)
        return raw_temp.get_data()[0]
    except Exception as e:
        print(f"Error in FIR denoising: {e}")
        return signal

def bandpass_filter(signal, sfreq, l_freq=1.0, h_freq=60.0):
    """Apply bandpass filter (1–60 Hz) after wavelet denoising"""
    try:
        raw_temp = mne.io.RawArray(signal[np.newaxis, :], mne.create_info(['temp'], sfreq, ch_types='eeg'))
        raw_temp.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=dict(order=4, ftype='butter'), phase='zero', verbose=True)
        return raw_temp.get_data()[0]
    except Exception as e:
        print(f"Error in bandpass filtering: {e}")
        return signal

def calculate_residual_snr(raw_signal, denoised_signal):
    """Calculate SNR using residual method: filtered as signal, (raw-filtered) as noise."""
    try:
        residual = raw_signal - denoised_signal
        signal_power = np.mean(denoised_signal ** 2)
        noise_power = np.mean(residual ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))
        return snr_db
    except Exception as e:
        print(f"Error in residual SNR calculation: {e}")
        return np.nan
        
def plot_eeg_signals(raw, channels, sfreq, plot_start=PLOT_START, plot_end=PLOT_END):
    """Evaluate filters with Diff SNR and generate plots"""
    try:
        # Load full data (all time points)
        full_data = raw.get_data(picks=channels)
        start_sample = int(plot_start * sfreq)
        stop_sample = int(plot_end * sfreq)
        times = np.arange(start_sample, stop_sample) / sfreq + plot_start

        filters = [
                ('DB4 Wavelet LVL: 2 thresh: 1', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=2, threshold_scale=1.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),
                ('DB4 Wavelet LVL: 2 thresh: 2', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=2, threshold_scale=2.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),
                ('DB4 Wavelet LVL: 2 thresh: 3', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=2, threshold_scale=3.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),
                ('DB4 Wavelet LVL: 2 thresh: 4', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=2, threshold_scale=4.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),

                ('DB4 Wavelet LVL: 3 thresh: 1', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=3, threshold_scale=1.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),
                ('DB4 Wavelet LVL: 3 thresh: 2', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=3, threshold_scale=2.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),
                ('DB4 Wavelet LVL: 3 thresh: 3', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=3, threshold_scale=3.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),
                ('DB4 Wavelet LVL: 3 thresh: 4', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=3, threshold_scale=4.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),

                ('DB4 Wavelet LVL: 4 thresh: 1', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=4, threshold_scale=1.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),
                ('DB4 Wavelet LVL: 4 thresh: 2', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=4, threshold_scale=2.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),
                ('DB4 Wavelet LVL: 4 thresh: 3', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=4, threshold_scale=3.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),
                ('DB4 Wavelet LVL: 4 thresh: 4', lambda x: bandpass_filter(wavelet_denoise(x, wavelet='db4', level=4, threshold_scale=4.0, mode='hard'), sfreq, l_freq=1.0, h_freq=60.0)),

        ]
        diff_snrs = {filt_name: [] for filt_name, _ in filters}
        filtered_full_data = {filt_name: np.zeros_like(full_data) for filt_name, _ in filters}

        for filt_name, filt_func in filters:
            print(f"\nProcessing {filt_name} (full recording):")
            # Apply filter to full data
            for i in range(len(channels)):
                filtered_full_data[filt_name][i] = filt_func(full_data[i])
            
            # Extract segment for plotting and SNR
            filtered_data = filtered_full_data[filt_name][:, start_sample:stop_sample]
            
            for i in range(len(channels)):
                snr_residual = calculate_residual_snr(full_data[i, start_sample:stop_sample], filtered_data[i])
                diff_snrs[filt_name].append(snr_residual)
                print(f"Channel {channels[i]} - Residual SNR: {snr_residual:.1f} dB")
            
            # 4x1 subplot
            plt.figure(figsize=(12, 8))
            plt.suptitle(f'EEG Signals: Bandpass-Filtered vs {filt_name} ({plot_start}s–{plot_end}s)', y=1.02)
            
            for ch_idx, ch_name in enumerate(channels):
                norm_raw = full_data[ch_idx, start_sample:stop_sample] / (np.std(full_data[ch_idx, start_sample:stop_sample]) + 1e-12)
                norm_filt = filtered_data[ch_idx] / (np.std(filtered_data[ch_idx]) + 1e-12)
                
                plt.subplot(NUM_CHAN, 1, ch_idx + 1)
                plt.plot(times, norm_raw, 'b', alpha=0.7, label='Bandpass-Filtered (0.5–60 Hz)')
                plt.plot(times, norm_filt, 'r', alpha=0.7, label=filt_name)
                plt.ylabel(f"{ch_name} (Norm)")
                plt.xlim([times[0], times[-1]])
                if ch_idx == 0:
                    plt.legend(loc='upper right')
                if ch_idx == 3:
                    plt.xlabel('Time (s)')
                else:
                    plt.xticks([])
                plt.text(0.02, 0.85, f"Diff SNR: {diff_snrs[filt_name][ch_idx]:.1f} dB",
                         transform=plt.gca().transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
            
            plt.tight_layout()
            plt.show()
            ''' 
            # PSD plots
            for ch_idx, ch_name in enumerate(channels):
                f, Pxx_raw = welch(full_data[ch_idx, start_sample:stop_sample], sfreq, nperseg=512)
                f, Pxx_filt = welch(filtered_data[ch_idx], sfreq, nperseg=512)
                f, Pxx_diff = welch(full_data[ch_idx, start_sample:stop_sample] - filtered_data[ch_idx], sfreq, nperseg=512)
                plt.figure(figsize=(10, 5))
                plt.semilogy(f, Pxx_raw, 'b', alpha=0.7, label='Bandpass-Filtered (0.5–60 Hz)')
                plt.semilogy(f, Pxx_filt, 'r', alpha=0.7, label=filt_name)
                plt.semilogy(f, Pxx_diff, 'g', alpha=0.7, label='Difference (Raw - Denoised)')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power Spectral Density')
                plt.title(f'PSD Comparison for {ch_name} ({filt_name})')
                plt.axvline(x=60, color='k', linestyle='--', label='60 Hz Cutoff')
                plt.legend()
                plt.show()
            '''
    except Exception as e:
        print(f"Error in processing and plotting: {e}")

if __name__ == "__main__":
    EDF_PATH = "/home/nicko/implementations/THESIS/main-thesis-folder/physionet.org/files/chbmit/1.0.0/chb02/chb02_16.edf"
    raw, channels, sfreq = load_eeg_data(EDF_PATH, l_freq=0.5, h_freq=60.0)
    if raw is not None:
        plot_eeg_signals(raw, channels, sfreq, plot_start=PLOT_START, plot_end=PLOT_END)
    else:
        print("Failed to load EEG data. Exiting.")
