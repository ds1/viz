import numpy as np
from scipy import signal
from scipy.integrate import simps
from typing import Tuple, List, Dict, Optional
import logging
from dataclasses import dataclass

from src.constants import ProcessingConfig, DataType, StreamConfig
from src.custom_types import (
    SignalValue,  # For signal data typing
    ProcessedSignal,  # If used for processed data
    FilterParameters,  # For filter configurations
    QualityMetrics  # Instead of local SignalQuality
)

# If you want to keep SignalQuality local because it's different from QualityMetrics:
@dataclass
class SignalQuality:
    """Signal quality metrics"""
    noise_level: float
    artifact_ratio: float
    signal_strength: float
    overall_quality: float

class FilterUtils:
    """Utilities for signal filtering"""
    
    @staticmethod
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

    @staticmethod
    def create_lowpass_filter(
        sampling_rate: float,
        cutoff: float,
        order: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create Butterworth lowpass filter coefficients"""
        nyq = 0.5 * sampling_rate
        normal_cutoff = cutoff / nyq
        return signal.butter(order, normal_cutoff, btype='low')

    @staticmethod
    def apply_filter_with_mirror(
        data: np.ndarray,
        b: np.ndarray,
        a: np.ndarray,
        axis: int = -1
    ) -> np.ndarray:
        """Apply filter with signal mirroring to reduce edge effects"""
        # Mirror the signal
        mirror_len = len(data) // 2
        mirrored = np.concatenate([
            data[..., :mirror_len][..., ::-1],
            data,
            data[..., -mirror_len:][..., ::-1]
        ], axis=axis)
        
        # Apply filter
        filtered = signal.filtfilt(b, a, mirrored, axis=axis)
        
        # Return central portion
        return filtered[..., mirror_len:-mirror_len]

class SpectralAnalysis:
    """Utilities for spectral analysis"""
    
    @staticmethod
    def get_bandpower(
        data: np.ndarray,
        sampling_rate: float,
        band_range: Tuple[float, float],
        window: Optional[str] = 'hann',
        relative: bool = False
    ) -> np.ndarray:
        """Calculate bandpower in specific frequency range"""
        # Compute spectrum
        freqs, psd = signal.welch(
            data,
            sampling_rate,
            window=window,
            nperseg=min(len(data), 256),
            noverlap=None,
            scaling='density'
        )
        
        # Find frequency range indexes
        idx_band = np.logical_and(freqs >= band_range[0], freqs <= band_range[1])
        
        # Calculate bandpower
        bandpower = simps(psd[idx_band], freqs[idx_band])
        
        if relative:
            total_power = simps(psd, freqs)
            bandpower /= total_power
            
        return bandpower

    @staticmethod
    def get_psd_multitaper(
        data: np.ndarray,
        sampling_rate: float,
        freq_range: Tuple[float, float],
        bandwidth: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PSD using multitaper method"""
        if bandwidth is None:
            bandwidth = 4.0  # Default bandwidth
            
        # Compute optimal parameters
        N = data.shape[-1]
        NW = max(bandwidth * N / (2 * sampling_rate), 4.0)
        K = int(2 * NW) - 1  # Number of tapers
        
        freqs, psd = signal.pmtm(
            data,
            NW,
            K,
            sampling_rate,
            method='eigen'
        )
        
        # Trim to frequency range
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        return freqs[mask], psd[mask]

class ArtifactDetection:
    """Utilities for artifact detection"""
    
    @staticmethod
    def detect_artifacts(
        data: np.ndarray,
        sampling_rate: float,
        threshold_std: float = 3.0,
        window_size: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Detect artifacts using adaptive thresholding"""
        if window_size is None:
            window_size = ProcessingConfig.ARTIFACT_WINDOW
            
        window_samples = int(window_size * sampling_rate)
        
        # Initialize outputs
        artifacts = np.zeros_like(data, dtype=bool)
        metrics = {}
        
        try:
            # Calculate rolling statistics
            window = np.hamming(window_samples)
            rolling_mean = signal.convolve(
                data,
                window / window.sum(),
                mode='same'
            )
            
            rolling_std = np.sqrt(
                signal.convolve(
                    (data - rolling_mean) ** 2,
                    window / window.sum(),
                    mode='same'
                )
            )
            
            # Detect artifacts
            z_scores = np.abs((data - rolling_mean) / (rolling_std + 1e-6))
            artifacts = z_scores > threshold_std
            
            # Connect nearby artifacts
            min_gap = int(sampling_rate * 0.1)  # 100ms
            for i in range(len(artifacts) - min_gap):
                if artifacts[i] and artifacts[i + min_gap]:
                    artifacts[i:i + min_gap] = True
                    
            # Calculate metrics
            metrics = {
                'artifact_ratio': np.mean(artifacts),
                'max_deviation': np.max(z_scores),
                'mean_deviation': np.mean(z_scores)
            }
            
        except Exception as e:
            logging.error(f"Error detecting artifacts: {str(e)}")
            
        return artifacts, metrics

    @staticmethod
    def interpolate_artifacts(
        data: np.ndarray,
        artifacts: np.ndarray,
        method: str = 'linear'
    ) -> np.ndarray:
        """Interpolate detected artifacts"""
        clean_data = data.copy()
        
        # Find artifact segments
        changes = np.diff(artifacts.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        # Handle edge cases
        if artifacts[0]:
            starts = np.r_[0, starts]
        if artifacts[-1]:
            ends = np.r_[ends, len(artifacts)]
            
        # Interpolate each segment
        for start, end in zip(starts, ends):
            if start == 0:
                clean_data[start:end] = data[end]
            elif end == len(data):
                clean_data[start:end] = data[start-1]
            else:
                t = np.arange(end - start + 2)
                yp = np.array([data[start-1], data[end]])
                clean_data[start:end] = np.interp(
                    t[1:-1],
                    [t[0], t[-1]],
                    yp
                )
                
        return clean_data

class SignalQualityMetrics:
    """Utilities for signal quality assessment"""
    
    @staticmethod
    def calculate_signal_quality(
        data: np.ndarray,
        sampling_rate: float,
        data_type: DataType
    ) -> SignalQuality:
        """Calculate comprehensive signal quality metrics"""
        try:
            # Calculate noise level
            detrended = signal.detrend(data)
            noise_level = np.std(detrended) / np.std(data)
            
            # Detect artifacts
            artifacts, artifact_metrics = ArtifactDetection.detect_artifacts(
                data,
                sampling_rate
            )
            artifact_ratio = artifact_metrics['artifact_ratio']
            
            # Calculate signal strength
            signal_strength = SpectralAnalysis.get_bandpower(
                data,
                sampling_rate,
                ProcessingConfig.FILTER_CONFIGS[data_type]['default']['bandpass'],
                relative=True
            )
            
            # Calculate overall quality
            weights = [0.3, 0.4, 0.3]  # Noise, artifacts, signal
            metrics = [
                1 - noise_level,
                1 - artifact_ratio,
                signal_strength
            ]
            overall_quality = np.average(metrics, weights=weights)
            
            return SignalQuality(
                noise_level=noise_level,
                artifact_ratio=artifact_ratio,
                signal_strength=signal_strength,
                overall_quality=overall_quality
            )
            
        except Exception as e:
            logging.error(f"Error calculating signal quality: {str(e)}")
            return SignalQuality(1.0, 1.0, 0.0, 0.0)

class HeartRateAnalysis:
    """Utilities for PPG heart rate analysis"""
    
    @staticmethod
    def calculate_heart_rate(
        ppg_data: np.ndarray,
        sampling_rate: float,
        method: str = 'peaks'
    ) -> Tuple[float, float]:
        """Calculate heart rate from PPG signal"""
        try:
            # Filter signal for heart rate band
            b, a = FilterUtils.create_bandpass_filter(
                sampling_rate,
                0.7,  # ~42 BPM
                3.5,  # ~210 BPM
                order=2
            )
            filtered = FilterUtils.apply_filter_with_mirror(ppg_data, b, a)
            
            if method == 'peaks':
                # Find peaks
                peaks, properties = signal.find_peaks(
                    filtered,
                    distance=int(sampling_rate * 0.5),  # Minimum 0.5s between peaks
                    prominence=0.1
                )
                
                if len(peaks) >= 2:
                    # Calculate intervals
                    intervals = np.diff(peaks) / sampling_rate
                    
                    # Remove outliers
                    good_intervals = np.abs(intervals - np.median(intervals)) < (2 * np.std(intervals))
                    if np.any(good_intervals):
                        clean_intervals = intervals[good_intervals]
                        
                        # Calculate heart rate and confidence
                        hr = 60 / np.mean(clean_intervals)
                        confidence = 1 - (np.std(clean_intervals) / np.mean(clean_intervals))
                        
                        return hr, confidence
                        
            elif method == 'fft':
                # Use FFT method as backup
                freqs, psd = SpectralAnalysis.get_psd_multitaper(
                    filtered,
                    sampling_rate,
                    (0.7, 3.5)  # Same as filter band
                )
                
                # Find dominant frequency
                peak_idx = np.argmax(psd)
                hr = freqs[peak_idx] * 60
                
                # Calculate confidence based on peak prominence
                prominence = psd[peak_idx] / np.mean(psd)
                confidence = min(prominence / 10, 1.0)
                
                return hr, confidence
                
        except Exception as e:
            logging.error(f"Error calculating heart rate: {str(e)}")
            
        return 0.0, 0.0