
# Step 1: Load the CSV file
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import mne

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

# Step 3: Time-series plot for selected channels
selected_channels = ['F3', 'C3', 'O1']  # Plot a subset for clarity
time = np.arange(len(eeg_data)) / sampling_rate  # Time axis in seconds

plt.figure(figsize=(12, 6))
for ch in selected_channels:
    plt.plot(time, eeg_data[ch], label=ch, alpha=0.8)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('EEG Time Series - Selected Channels')
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Power Spectral Density (PSD) for a selected channel
selected_channel = 'Cz'  # Example channel
f, Pxx = welch(eeg_data[selected_channel], fs=sampling_rate, nperseg=2048)  # Adjusted nperseg for 500 Hz

plt.figure(figsize=(10, 5))
plt.semilogy(f, Pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (µV²/Hz)')
plt.title(f'PSD of Channel {selected_channel}')
plt.grid(True)
plt.xlim(0, 45)  # Focus on 0-45 Hz (dataset has 45 Hz low-pass filter)
plt.show()

# Step 5: Topographic plot using MNE
# Create MNE Info object
info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types='eeg')
# Use standard 10-20 montage
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Create MNE Raw object from data
data = eeg_data[channels].values.T  # MNE expects channels x samples
raw = mne.io.RawArray(data, info)

# Plot topographic map (average power in alpha band: 8-12 Hz)
raw.plot_psd_topomap(bands={'Alpha (8-12 Hz)': (8, 12)}, ch_type='eeg', normalize=True)
plt.show()
