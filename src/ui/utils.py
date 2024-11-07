import numpy as np
from scipy import signal
from typing import Tuple, List
import logging

def create_bandpass_filter(
    sampling_rate: float,
    lowcut: float,
    highcut: float,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Create Butterworth bandpass filter coefficients"""
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    return signal.butter(order, [low, high], btype='band')

def create_lowpass_filter(
    sampling_rate: float,
    cutoff: float,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Create Butterworth lowpass filter coefficients"""
    nyq = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyq
    return signal.butter(order, normal_cutoff, btype='low')

def calculate_signal_quality(
    data: np.ndarray,
    sampling_rate: float
) -> float:
    """Calculate signal quality metric"""
    try:
        # Compute power in relevant frequency bands
        freqs, psd = signal.welch(data, sampling_rate)
        
        # Calculate signal-to-noise ratio
        signal_band = (0.1, 40)  # Hz
        noise_band = (40, sampling_rate/2)  # Hz
        
        signal_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
        noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
        
        signal_power = np.mean(psd[signal_mask])
        noise_power = np.mean(psd[noise_mask])
        
        if noise_power == 0:
            return 0.0
            
        snr = 10 * np.log10(signal_power / noise_power)
        
        # Normalize to 0-1 range
        quality = 1 / (1 + np.exp(-0.1 * (snr + 20)))
        
        return quality
        
    except Exception as e:
        logging.error(f"Error calculating signal quality: {str(e)}")
        return 0.0

def detect_artifacts(
    data: np.ndarray,
    sampling_rate: float,
    threshold: float = 3.0
) -> np.ndarray:
    """Detect artifacts in signal"""
    try:
        # Compute rolling statistics
        window = int(sampling_rate * 0.1)  # 100ms window
        rolling_std = np.zeros_like(data)
        rolling_mean = np.zeros_like(data)
        
        for i in range(window, len(data)):
            segment = data[i-window:i]
            rolling_std[i] = np.std(segment)
            rolling_mean[i] = np.mean(segment)
            
        # Detect artifacts
        z_scores = np.abs((data - rolling_mean) / (rolling_std + 1e-6))
        artifacts = z_scores > threshold
        
        # Connect nearby artifacts
        min_gap = int(sampling_rate * 0.05)  # 50ms
        for i in range(len(artifacts) - min_gap):
            if artifacts[i] and artifacts[i + min_gap]:
                artifacts[i:i + min_gap] = True
                
        return artifacts
        
    except Exception as e:
        logging.error(f"Error detecting artifacts: {str(e)}")
        return np.zeros_like(data, dtype=bool)

def calculate_heart_rate(
    ppg_data: np.ndarray,
    sampling_rate: float
) -> float:
    """Calculate heart rate from PPG signal"""
    try:
        # Filter signal
        nyq = 0.5 * sampling_rate
        low = 0.5 / nyq
        high = 3.5 / nyq
        b, a = signal.butter(2, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ppg_data)
        
        # Find peaks
        peaks, _ = signal.find_peaks(
            filtered,
            distance=int(sampling_rate * 0.5),  # Minimum 0.5s between peaks
            height=np.mean(filtered)
        )
        
        if len(peaks) < 2:
            return 0.0
            
        # Calculate average interval
        intervals = np.diff(peaks) / sampling_rate
        avg_interval = np.mean(intervals)
        
        # Convert to BPM
        hr = 60 / avg_interval
        
        return hr if 40 <= hr <= 200 else 0.0
        
    except Exception as e:
        logging.error(f"Error calculating heart rate: {str(e)}")
        return 0.0
