import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from scipy.stats import kurtosis
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

# 1. Parameters
EDF_FILE = "/home/nicko/implementations/THESIS/main-thesis-folder/physionet.org/files/chbmit/1.0.0/chb02/chb02_16.edf"
SEIZURE_START = 130  # seconds
SEIZURE_END = 212    # seconds
EPOCH_DURATION = 4   # seconds for each epoch

##parameters to test the self-RQA now.
RQA_PARAMS = {
    'embedding_dim': 3,
    'time_delay': 10,
    'radius': 0.2,  # Recurrence threshold
    'lmin': 2       # Minimal diagonal line length
}

# 2. Helper Functions
def compute_rqa(signal, params):
    """Properly compute Recurrence Quantification Analysis metrics"""
    N = len(signal)
    emb_dim = params['embedding_dim']
    tau = params['time_delay']

    # 1. Time-delay embedding
    embedded = np.zeros((N - (emb_dim-1)*tau, emb_dim))
    for i in range(emb_dim):
        embedded[:, i] = signal[i*tau : N - (emb_dim-i-1)*tau]

    # 2. Distance matrix (normalized)
    dist_matrix = squareform(pdist(embedded, 'euclidean'))
    dist_matrix /= np.max(dist_matrix)  # Normalize to [0,1]

    # 3. Recurrence matrix
    recurrence_matrix = (dist_matrix < params['radius']).astype(int)
    np.fill_diagonal(recurrence_matrix, 0)  # Remove main diagonal

    # 4. Calculate RQA metrics
    N = len(recurrence_matrix)
    if N == 0:
        return {'RR': 0, 'DET': 0, 'L': 0, 'Lmax': 0}

    RR = np.sum(recurrence_matrix) / (N*N)  # Recurrence rate

    # Diagonal line analysis
    diag_lines = []
    for i in range(-N+1, N):
        diag = np.diag(recurrence_matrix, i)
        changes = np.where(np.diff(np.concatenate(([0], diag, [0])))[0])
        segments = [(changes[j], changes[j+1]) for j in range(len(changes)-1)]
        for start, end in segments:
            if diag[start] == 1 and (end-start) >= params['lmin']:
                diag_lines.append(end-start)

    if diag_lines:
        DET = np.sum(diag_lines) / np.sum(recurrence_matrix)
        L = np.mean(diag_lines)
        Lmax = np.max(diag_lines)
    else:
        DET, L, Lmax = 0, 0, 0
    print("RR")
    print(RR)
    print("DET")
    print(DET)
    print("L")
    print(L)
    print("Lmax")
    print(Lmax)
    return {
        'RR': float(RR),
        'DET': float(DET),
        'L': float(L),
        'Lmax': float(Lmax)
    }


# 3. Main Processing Pipeline
def main():
    # Load and prepare data
    raw = mne.io.read_raw_edf(EDF_FILE, preload=True)

    ## annotating the data seizure start and end time in seconds
    annotations = mne.Annotations(onset=[SEIZURE_START], 
                                duration=[SEIZURE_END-SEIZURE_START], 
                                description=['Seizure'])
    raw.set_annotations(annotations)
    
    # Apply bandpass filter via the MNE library(1-40 Hz)
    raw.filter(1, 40, fir_design='firwin')
    
    # part of Independent Component Analysis for the purpose of
    # removing the artifacts
    print("Running ICA")
    ica = ICA(n_components=15, random_state=42)
    ica.fit(raw.copy().pick_types(eeg=True))
    
    # Detection artifact components
    sources = ica.get_sources(raw).get_data()
    kurt_values = kurtosis(sources, axis=1, fisher=True)
    exclude = np.where(kurt_values > np.percentile(kurt_values, 75))[0]
    print(f"Excluding ICA components: {exclude}")
    

    raw_cleaned = raw.copy()
    #getting the clean after excluding the components that didn't pass.
    ica.apply(raw_cleaned, exclude=exclude)
    
    # Testing some non-overlapping epochs of EPOCH DURATION seconds.
    epochs = mne.make_fixed_length_epochs(raw_cleaned, duration=EPOCH_DURATION, 
                                        overlap=0, preload=True)
    print(f"Created {len(epochs)} epochs of {EPOCH_DURATION} seconds")
    
    # Compute RQA metrics for each channel and epoch
    ## its like self-RQA on each channel per segment(epoch) here.
    rqa_results = {}
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        print(f"\nProcessing channel: {ch_name} ({ch_idx+1}/{len(epochs.ch_names)})")
        channel_data = epochs.get_data(picks=ch_name)[:, 0, :]  # Shape: (n_epochs, n_times)
        
        # Compute RQA for each epoch
        channel_results = []
        for epoch_idx, epoch_data in enumerate(channel_data):
            rqa_metrics = compute_rqa(epoch_data, RQA_PARAMS)
            channel_results.append(rqa_metrics)
        
        rqa_results[ch_name] = channel_results
    
    # 4. Visualization
    print("\nPlotting RQA results with epoch-wise evolution...")
    metrics = ['RR', 'DET', 'L', 'Lmax']
    time_axis = np.arange(len(epochs)) * EPOCH_DURATION + EPOCH_DURATION/2  # Center of each epoch

    # Create a figure for each metric
    for metric in metrics:
        plt.figure(figsize=(15, 8))
        
        # Prepare data matrix: channels × epochs
        data_matrix = np.zeros((len(epochs.ch_names), len(epochs)))
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            data_matrix[ch_idx, :] = [rqa[metric] for rqa in rqa_results[ch_name]]
        
        # Normalize per channel for better visualization
        norm_data = (data_matrix - data_matrix.mean(axis=1, keepdims=True)) / data_matrix.std(axis=1, keepdims=True)
        
        # Create a colormap for channels
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(epochs.ch_names))]
        
        # Plot each channel's metric evolution
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            plt.plot(time_axis, norm_data[ch_idx], 
                    color=colors[ch_idx], 
                    alpha=0.7, 
                    label=ch_name,
                    linewidth=1.5)
        
        # Marking the seizure period on the plot.
        plt.axvspan(SEIZURE_START, SEIZURE_END, color='red', alpha=0.2, label='Seizure')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel(f'Normalized {metric}', fontsize=12)
        plt.title(f'Evolution of {metric} Across Epochs (All Channels)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add epoch boundaries
        for t in np.arange(0, raw.times[-1], EPOCH_DURATION):
            plt.axvline(t, color='gray', linestyle=':', alpha=0.3)
        
        plt.show()


if __name__ == "__main__":
    main()
