import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import mne
from scipy import signal
import networkx as nx


file_path = '/home/nicko/implementations/THESIS/papers/bibliographic_reference/writing/eeg_data1/data/s00.csv'
channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4',
            'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
eeg_data = pd.read_csv(file_path, header=None)  # No header in CSV
eeg_data.columns = channels  # Assign channel names

sampling_rate = 500  # Hz, as specified in the dataset description
n_channels = len(channels)

if eeg_data.shape[1] != n_channels:
    raise ValueError(f"Expected {n_channels} columns, but found {eeg_data.shape[1]}.")


time = np.arange(len(eeg_data)) / sampling_rate  # Time axis in seconds

from matplotlib.widgets import Slider

fig, ax = plt.subplots(figsize=(15, 8))
plt.subplots_adjust(bottom=0.25)

# Initial plot of first few channels
lines = []
for i, ch in enumerate(channels[:]):
    line, = ax.plot(time, eeg_data[ch] - i*100, label=ch)
    lines.append(line)
ax.legend()
ax.grid(True)

# Add slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Channel Offset', 0, len(channels)-5, valinit=0, valstep=1)

def update(val):
    offset = int(slider.val)
    for i, line in enumerate(lines):
        if offset + i < len(channels):
            ch = channels[offset + i]
            line.set_ydata(eeg_data[ch] - i*100)
            line.set_label(ch)
    ax.legend()
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()


# Compute Pearson Correlation Matrix
corr_matrix = eeg_data.corr().values

# Compute Phase Locking Value (PLV)
def compute_plv(data, sfreq):
    n_channels = data.shape[1]
    plv_matrix = np.zeros((n_channels, n_channels))

    # Compute analytic signal (Hilbert transform)
    analytic_signal = signal.hilbert(data)
    phases = np.angle(analytic_signal)

    for i in range(n_channels):
        for j in range(i+1, n_channels):
            phase_diff = phases[:,i] - phases[:,j]
            plv = np.abs(np.mean(np.exp(1j*phase_diff)))
            plv_matrix[i,j] = plv
            plv_matrix[j,i] = plv

    np.fill_diagonal(plv_matrix, 1)
    return plv_matrix

plv_matrix = compute_plv(eeg_data.values, sampling_rate)

# Visualize the matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

im1 = ax1.imshow(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
ax1.set_title('Pearson Correlation')
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(plv_matrix, cmap='viridis', vmin=0, vmax=1)
ax2.set_title('Phase Locking Value (PLV)')
plt.colorbar(im2, ax=ax2)

ax1.set_xticks(range(len(channels)))
ax1.set_yticks(range(len(channels)))
ax1.set_xticklabels(channels, rotation=90)
ax1.set_yticklabels(channels)

ax2.set_xticks(range(len(channels)))
ax2.set_yticks(range(len(channels)))
ax2.set_xticklabels(channels, rotation=90)
ax2.set_yticklabels(channels)

plt.tight_layout()
plt.show()



##thresholding in order to create an adjusency matrix
threshold = 0.5  # Adjust based on your data
adj_matrix = (plv_matrix > threshold).astype(int)
np.fill_diagonal(adj_matrix, 0)  # Remove self-connections

# Create graph
G = nx.from_numpy_array(adj_matrix)
pos = nx.circular_layout(G)  # Circular arrangement for EEG channels

# Draw graph
plt.figure(figsize=(10,8))
nx.draw(G, pos, with_labels=True, labels=dict(enumerate(channels)),
        node_color='skyblue', node_size=800, edge_color='gray')
plt.title(f'Functional Connectivity Network (PLV > {threshold})')
plt.show()







