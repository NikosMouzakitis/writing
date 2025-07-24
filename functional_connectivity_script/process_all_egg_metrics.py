import os
import shutil
import pandas as pd
import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import welch
from scipy.integrate import simpson  # Updated from simps to simpson
from antropy import spectral_entropy, sample_entropy, petrosian_fd, hjorth_params

# Configuration
data_dir = '/home/nicko/implementations/THESIS/papers/bibliographic_reference/writing/eeg_data1/data/'
output_dir = '/home/nicko/implementations/THESIS/papers/bibliographic_reference/writing/functional_connectivity_script/results/'

# Clear results directory before starting
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4',
            'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
sampling_rate = 500
threshold = 0.5

# Frequency bands
freq_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Initialize results storage with additional metrics
results = {
    'subject': [],
    'mean_plv': [],
    'num_connections': [],
    'global_efficiency': [],
    'modularity': [],
    'mean_power_delta': [],
    'mean_power_theta': [],
    'mean_power_alpha': [],
    'mean_power_beta': [],
    'mean_power_gamma': [],
    'spectral_entropy': [],
    'sample_entropy': [],
    'petrosian_fd': [],
    'hjorth_mobility': [],
    'hjorth_complexity': [],
    'mean_skewness': [],
    'mean_kurtosis': [],
    'mean_hurst': [],
    'mean_correlation_dimension': []
}

def compute_plv(data, sfreq):
    """Compute Phase Locking Value matrix"""
    analytic_signal = signal.hilbert(data)
    phases = np.angle(analytic_signal)
    n_channels = data.shape[1]
    plv_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            phase_diff = phases[:,i] - phases[:,j]
            plv = np.abs(np.mean(np.exp(1j*phase_diff)))
            plv_matrix[i,j] = plv
            plv_matrix[j,i] = plv
    
    np.fill_diagonal(plv_matrix, 1)
    return plv_matrix

def analyze_connectivity(plv_matrix, threshold):
    """Analyze connectivity properties"""
    adj_matrix = (plv_matrix > threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    
    # Create graph
    G = nx.from_numpy_array(adj_matrix)
    
    # Calculate metrics
    try:
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        modularity = nx.algorithms.community.modularity(G, communities)
    except:
        modularity = 0  # Handle cases where modularity can't be computed
    
    metrics = {
        'mean_plv': np.mean(plv_matrix[np.triu_indices_from(plv_matrix, k=1)]),
        'num_connections': np.sum(adj_matrix)/2,
        'global_efficiency': nx.global_efficiency(G),
        'modularity': modularity
    }
    return metrics, adj_matrix

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band."""
    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = min(256, len(data))
    
    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)
    
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    
    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    
    # Integral approximation of the spectrum using Simpson's rule
    bp = simpson(psd[idx_band], dx=freq_res)  # Changed from simps to simpson
    
    if relative:
        bp /= simpson(psd, dx=freq_res)  # Changed from simps to simpson
    return bp

def compute_hurst_exponent(time_series):
    """Returns the Hurst Exponent of the time series"""
    lags = range(2, 100)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

def compute_correlation_dimension(time_series, emb_dim=10):
    """Estimate correlation dimension"""
    n = len(time_series)
    corr_sum = 0
    r = 0.1 * np.std(time_series)
    
    for i in range(n - emb_dim):
        for j in range(i + 1, n - emb_dim):
            dist = np.linalg.norm(time_series[i:i+emb_dim] - time_series[j:j+emb_dim])
            if dist < r:
                corr_sum += 1
                
    if corr_sum == 0:
        return 0
    return np.log(corr_sum) / np.log(r)

def extract_time_domain_features(data):
    """Extract time domain features from EEG data"""
    features = {
        'skewness': stats.skew(data, axis=0),
        'kurtosis': stats.kurtosis(data, axis=0),
        'hurst': [compute_hurst_exponent(data[:, ch]) for ch in range(data.shape[1])],
        'correlation_dimension': [compute_correlation_dimension(data[:, ch]) for ch in range(data.shape[1])]
    }
    return features

# Process all files
for file_name in sorted(os.listdir(data_dir)):
    if file_name.endswith('.csv'):
        subject_id = file_name.split('.')[0]
        print(f"Processing {subject_id}...")
        
        try:
            # Load data
            file_path = os.path.join(data_dir, file_name)
            eeg_data = pd.read_csv(file_path, header=None)
            eeg_data.columns = channels
            
            # Convert to numpy array
            eeg_array = eeg_data.values
            
            # Compute PLV
            plv_matrix = compute_plv(eeg_array, sampling_rate)
            
            # Analyze connectivity
            metrics, adj_matrix = analyze_connectivity(plv_matrix, threshold)
            
            # Compute spectral features
            band_powers = {}
            for band in freq_bands:
                try:
                    band_powers[f'mean_power_{band}'] = np.mean([bandpower(eeg_array[:, i], sampling_rate, freq_bands[band]) 
                                                              for i in range(eeg_array.shape[1])])
                except:
                    band_powers[f'mean_power_{band}'] = np.nan
            
            # Compute entropy measures
            entropy_metrics = {
                'spectral_entropy': np.mean([spectral_entropy(eeg_array[:, i], sf=sampling_rate, method='welch') 
                                           for i in range(eeg_array.shape[1])]),
                'sample_entropy': np.mean([sample_entropy(eeg_array[:, i], order=2) 
                                         for i in range(eeg_array.shape[1])]),
                'petrosian_fd': np.mean([petrosian_fd(eeg_array[:, i]) 
                                      for i in range(eeg_array.shape[1])])
            }
            
            # Compute Hjorth parameters
            hjorth_mobility, hjorth_complexity = zip(*[hjorth_params(eeg_array[:, i]) 
                                                     for i in range(eeg_array.shape[1])])
            hjorth_metrics = {
                'hjorth_mobility': np.mean(hjorth_mobility),
                'hjorth_complexity': np.mean(hjorth_complexity)
            }
            
            # Compute time domain features
            time_features = extract_time_domain_features(eeg_array)
            time_metrics = {
                'mean_skewness': np.mean(time_features['skewness']),
                'mean_kurtosis': np.mean(time_features['kurtosis']),
                'mean_hurst': np.mean(time_features['hurst']),
                'mean_correlation_dimension': np.mean(time_features['correlation_dimension'])
            }
            
            # Store all results
            results['subject'].append(subject_id)
            for metric, value in metrics.items():
                results[metric].append(value)
            for metric, value in band_powers.items():
                results[metric].append(value)
            for metric, value in entropy_metrics.items():
                results[metric].append(value)
            for metric, value in hjorth_metrics.items():
                results[metric].append(value)
            for metric, value in time_metrics.items():
                results[metric].append(value)
            
            # Save individual subject plots
            plt.figure(figsize=(10,8))
            G = nx.from_numpy_array(adj_matrix)
            pos = nx.circular_layout(G)
            nx.draw(G, pos, with_labels=True, labels=dict(enumerate(channels)),
                    node_color='skyblue', node_size=800, edge_color='gray')
            plt.title(f'{subject_id} Connectivity (PLV > {threshold})')
            plt.savefig(os.path.join(output_dir, f'{subject_id}_connectivity.png'))
            plt.close()
            
            # Save PLV matrix
            np.save(os.path.join(output_dir, f'{subject_id}_plv.npy'), plv_matrix)
            
        except Exception as e:
            print(f"Error processing {subject_id}: {str(e)}")
            continue

# Save summary results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)

# Plot summary statistics
plt.figure(figsize=(20, 15))
metrics_to_plot = results_df.columns[1:]  # Skip subject column

for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(5, 4, i)
    plt.bar(results_df['subject'], results_df[metric])
    plt.title(metric.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'summary_statistics.png'))
plt.show()

print("Processing complete. Results saved to:", output_dir)
