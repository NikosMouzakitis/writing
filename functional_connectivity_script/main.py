import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import mne
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle
import seaborn as sns
import os

# Step 1: Load the CSV file and assign channel names
file_path = '/home/nicko/implementations/THESIS/papers/bibliographic_reference/writing/eeg_data1/data/s00.csv'
channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4',
            'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
eeg_data = pd.read_csv(file_path, header=None)  # No header in CSV
eeg_data.columns = channels  # Assign channel names

# Step 2: Define parameters
sampling_rate = 500  # Hz, as specified in the dataset description
n_channels = len(channels)

# Verify number of columns
if eeg_data.shape[1] != n_channels:
    raise ValueError(f"Expected {n_channels} columns, but found {eeg_data.shape[1]}.")

# Step 3: Create MNE Info and Raw objects
info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)
data = eeg_data[channels].values.T  # MNE expects channels x samples
raw = mne.io.RawArray(data, info)

# Step 4: Minimal preprocessing (notch filter for 50 Hz)
raw.notch_filter(freqs=50)

# Step 5: Time-series plot for selected channels
selected_channels = ['F3', 'C3', 'O1']
time = np.arange(len(eeg_data)) / sampling_rate
plt.figure(figsize=(12, 6))
for ch in selected_channels:
    plt.plot(time, eeg_data[ch], label=ch, alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('EEG Time Series - Selected Channels')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Power Spectral Density (PSD) for a selected channel
selected_channel = 'Cz'
f, Pxx = welch(eeg_data[selected_channel], fs=sampling_rate, nperseg=2048)
plt.figure(figsize=(10, 5))
plt.semilogy(f, Pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (µV²/Hz)')
plt.title(f'PSD of Channel {selected_channel}')
plt.grid(True)
plt.xlim(0, 45)
plt.axvspan(8, 12, color='gray', alpha=0.3, label='Alpha (8-12 Hz)')
plt.axvspan(4, 8, color='blue', alpha=0.1, label='Theta (4-8 Hz)')
plt.legend()
plt.show()

# Step 7: Check Power Across Frequency Bands
# Alpha band (8-12 Hz)
psd_alpha = raw.compute_psd(fmin=8, fmax=12, n_fft=2048)
alpha_power = psd_alpha.get_data().mean(axis=1)
print("Alpha Band Power (µV²) per Channel:")
for ch, power in zip(channels, alpha_power):
    print(f"{ch}: {power:.2e}")

# Theta band (4-8 Hz)
psd_theta = raw.compute_psd(fmin=4, fmax=8, n_fft=2048)
theta_power = psd_theta.get_data().mean(axis=1)
print("\nTheta Band Power (µV²) per Channel:")
for ch, power in zip(channels, theta_power):
    print(f"{ch}: {power:.2e}")

# Step 8: Topographic Plot (Alpha Band, working version)
raw.plot_psd_topomap(bands={'Alpha (8-12 Hz)': (8, 12)}, ch_type='eeg', normalize=True, cmap='viridis')
plt.title('Alpha Band Power (8-12 Hz)')
plt.show()

# Step 9: Topographic Plot (Alpha Band, fixed compute_psd)
psd = raw.compute_psd(fmin=8, fmax=12, n_fft=2048)
vlim = (0, 2e1)  # Tighter range based on alpha power
psd.plot_topomap(bands={'Alpha (8-12 Hz)': (8, 12)}, ch_type='eeg', normalize=False, vlim=vlim, cmap='viridis')
plt.title('Alpha Band Power (8-12 Hz, Non-Normalized)')
plt.show()

# Step 10: Topographic Plot (Theta Band, working version)
raw.plot_psd_topomap(bands={'Theta (4-8 Hz)': (4, 8)}, ch_type='eeg', normalize=True, cmap='viridis')
plt.title('Theta Band Power (4-8 Hz)')
plt.show()

# Step 11: Functional Connectivity Analysis (Coherence)
epoch_duration = 2
n_samples_per_epoch = int(epoch_duration * sampling_rate)
epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True)

# Compute spectral coherence (Alpha band)
freq_band = (8, 12)
conn = spectral_connectivity_epochs(
    epochs,
    method='coh',
    mode='multitaper',
    fmin=freq_band[0],
    fmax=freq_band[1],
    faverage=True,
    tmin=0,
    tmax=epoch_duration - 1/sampling_rate,
    mt_adaptive=False,
    n_jobs=1
)
conn_matrix = conn.get_data(output='dense')[:, :, 0]

# Step 12: Check Coherence Values
print("\nCoherence Matrix (Alpha Band) - Max Value:", conn_matrix.max())
print("Number of Connections > 0.01:", np.sum(conn_matrix > 0.01))
print("Sample Coherence Values (first 5x5):\n", conn_matrix[:5, :5])

# Step 13: Visualize Connectivity Matrix (Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(conn_matrix, xticklabels=channels, yticklabels=channels, cmap='viridis',
            vmin=0, vmax=1, annot=True, fmt='.2f')
plt.title('Coherence Connectivity Matrix (Alpha Band: 8-12 Hz)')
plt.show()

# Step 14: Visualize Connectivity as a Circular Graph
conn_matrix_thresholded = np.where(conn_matrix > 0.01, conn_matrix, 0)
plot_connectivity_circle(conn_matrix_thresholded, channels, title='Coherence Connectivity (Alpha Band: 8-12 Hz)',
                        vmin=0, vmax=1, colormap='viridis', fontsize_names=8)
plt.show()

# Step 15: 3D Sensor Connectivity Plot
from mne.viz import plot_sensors
fig = plot_sensors(info, kind='3d', ch_type='eeg', show_names=True)
ax = fig.gca()
ch_pos = montage.get_positions()['ch_pos']
coords_3d = np.array([ch_pos[ch] for ch in channels])
# Normalize coordinates to improve rendering
coords_3d = coords_3d / np.max(np.abs(coords_3d))  # Scale to [-1, 1]
for i in range(n_channels):
    for j in range(i + 1, n_channels):
        if conn_matrix_thresholded[i, j] > 0:
            ax.plot3D(
                [coords_3d[i][0], coords_3d[j][0]],
                [coords_3d[i][1], coords_3d[j][1]],
                [coords_3d[i][2], coords_3d[j][2]],
                color='blue', alpha=min(conn_matrix_thresholded[i, j] * 5, 1), linewidth=3
            )
plt.title('3D Sensor Connectivity (Alpha Band: 8-12 Hz)')
plt.show()

# Step 16: Source-Space Connectivity (Optional)
# Set SUBJECTS_DIR environment variable
os.environ['SUBJECTS_DIR'] = '/home/nicko/mne_data/MNE-fsaverage-data'

from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne import setup_source_space, make_forward_solution
from nilearn import plotting

# Download fsaverage if needed
mne.datasets.fetch_fsaverage(subjects_dir='/home/nicko/.mne')

# Setup source space
src = setup_source_space('fsaverage', spacing='oct6', add_dist=False, subjects_dir='/home/nicko/.mne')
fwd = make_forward_solution(raw.info, trans='fsaverage', src=src, bem='fsaverage', eeg=True, subjects_dir='/home/nicko/.mne')
inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov=mne.compute_raw_covariance(raw))

# Compute source-space time series
stc = apply_inverse(epochs.average(), inverse_operator, lambda2=1.0/9.0, method='dSPM')

# Compute source-space connectivity
conn_source = spectral_connectivity_epochs(
    epochs,
    method='coh',
    mode='multitaper',
    fmin=8,
    fmax=12,
    faverage=True,
    tmin=0,
    tmax=epoch_duration - 1/sampling_rate,
    indices=(np.arange(len(src[0]['vertno'])), np.arange(len(src[0]['vertno']))),
    n_jobs=1
)

# Visualize on 3D brain surface
plotting.plot_connectome(
    conn_source.get_data(output='dense')[:, :, 0],
    src[0]['rr'][src[0]['vertno']],
    title='Source-Space Connectivity (Alpha Band: 8-12 Hz)',
    edge_cmap='viridis',
    edge_vmin=0,
    edge_vmax=1
)
plt.show()
