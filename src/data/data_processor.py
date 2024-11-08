from PyQt5.QtCore import QObject, QMutex, pyqtSignal
import numpy as np
from typing import Optional, Dict, Any, Tuple
from collections import deque
import logging
from dataclasses import dataclass
from threading import Lock

from src.data.utils import (
    FilterUtils, SpectralAnalysis, ArtifactDetection,
    SignalQualityMetrics, HeartRateAnalysis, SignalQuality
)
from src.constants import (
    DataType, ProcessingConfig, StreamConfig,
    DisplayConfig
)
# Add if you need specific types:
from src.custom_types import ProcessedSignal, FilterParameters, QualityMetrics

@dataclass
class ProcessedData:
    """Container for processed data and metrics"""
    data: np.ndarray
    timestamp: float
    quality: Dict[str, SignalQuality]
    heart_rate: Optional[float] = None
    heart_rate_confidence: Optional[float] = None
    artifacts: Optional[np.ndarray] = None

class CircularBuffer:
    """Thread-safe circular buffer for real-time data"""
    
    def __init__(self, channels: int, size: int):
        self.buffer = np.zeros((channels, size))
        self.size = size
        self.position = 0
        self.filled = False
        self.lock = Lock()
        
    def add(self, data: np.ndarray) -> None:
        """Add new data to buffer"""
        with self.lock:
            n_samples = data.shape[1]
            
            if n_samples >= self.size:
                self.buffer = data[:, -self.size:]
                self.position = 0
                self.filled = True
                return
                
            remaining = self.size - self.position
            if n_samples > remaining:
                # Split the data
                first_part = n_samples - remaining
                self.buffer[:, self.position:] = data[:, :remaining]
                self.buffer[:, :first_part] = data[:, remaining:]
                self.position = first_part
            else:
                self.buffer[:, self.position:self.position + n_samples] = data
                self.position = (self.position + n_samples) % self.size
                
            self.filled = self.filled or self.position == 0
            
    def get_data(self) -> Optional[np.ndarray]:
        """Retrieve data from buffer"""
        with self.lock:
            if not self.filled and self.position == 0:
                return None
            if self.position == 0 or not self.filled:
                return self.buffer[:, :self.position]
            return np.concatenate((
                self.buffer[:, self.position:],
                self.buffer[:, :self.position]
            ), axis=1)
            
    def clear(self) -> None:
        """Clear buffer"""
        with self.lock:
            self.buffer.fill(0)
            self.position = 0
            self.filled = False

class DataProcessor(QObject):
    """Processes real-time data streams with signal quality monitoring"""
    
    # Signals
    processed_data = pyqtSignal(ProcessedData)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, data_type: DataType):
        super().__init__()
        
        # Configuration
        self.data_type = data_type
        self.sampling_rate = StreamConfig.SAMPLING_RATES[data_type]
        self.channels = StreamConfig.CHANNELS[data_type]
        self.n_channels = len(self.channels)
        
        # State
        self.mutex = QMutex()
        self.processing_enabled = True
        self.current_filter = 'default'
        self.current_quality = {}
        
        # Initialize buffers
        self.setup_buffers()
        
        # Initialize filters
        self.setup_filters()
        
    def setup_buffers(self) -> None:
        """Initialize data buffers"""
        buffer_size = ProcessingConfig.BUFFER_SIZES[self.data_type]
        
        # Main data buffer
        self.raw_buffer = CircularBuffer(self.n_channels, buffer_size)
        
        # Quality monitoring buffers
        quality_size = int(ProcessingConfig.QUALITY_WINDOW * self.sampling_rate)
        self.quality_buffers = {
            name: CircularBuffer(1, quality_size)
            for name in self.channels
        }
        
        # Heart rate buffer for PPG
        if self.data_type == DataType.PPG:
            hr_buffer_size = int(4 * self.sampling_rate)  # 4 seconds
            self.hr_buffer = CircularBuffer(1, hr_buffer_size)
        
    def setup_filters(self) -> None:
        """Initialize filters based on configuration"""
        self.filters = {}
        filter_configs = ProcessingConfig.FILTER_CONFIGS[self.data_type]
        
        for name, config in filter_configs.items():
            if 'bandpass' in config:
                self.filters[name] = FilterUtils.create_bandpass_filter(
                    self.sampling_rate,
                    config['bandpass'][0],
                    config['bandpass'][1]
                )
            elif 'lowpass' in config:
                self.filters[name] = FilterUtils.create_lowpass_filter(
                    self.sampling_rate,
                    config['lowpass']
                )
                
    def process_data(self, new_data: np.ndarray, timestamp: float) -> None:
        """Process new data with thread safety"""
        if not self.processing_enabled or new_data is None:
            return
            
        try:
            with QMutex():
                # Add to raw buffer
                self.raw_buffer.add(new_data)
                
                # Get complete data for processing
                data = self.raw_buffer.get_data()
                if data is None or len(data) < ProcessingConfig.MIN_SAMPLES:
                    return
                    
                # Apply current filter if needed
                if self.current_filter != 'off':
                    data = self.apply_filter(data)
                    
                # Update quality metrics
                self.update_quality_metrics(data)
                
                # Detect artifacts
                artifacts = self.detect_artifacts(data)
                
                # Process PPG heart rate
                heart_rate = heart_rate_confidence = None
                if self.data_type == DataType.PPG:
                    heart_rate, heart_rate_confidence = self.process_heart_rate(data)
                
                # Create processed data object
                processed = ProcessedData(
                    data=data,
                    timestamp=timestamp,
                    quality=self.current_quality,
                    heart_rate=heart_rate,
                    heart_rate_confidence=heart_rate_confidence,
                    artifacts=artifacts
                )
                
                # Emit processed data
                self.processed_data.emit(processed)
                
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply current filter to data"""
        if self.current_filter not in self.filters:
            return data
            
        filter_coeffs = self.filters[self.current_filter]
        return FilterUtils.apply_filter_with_mirror(
            data,
            filter_coeffs[0],
            filter_coeffs[1]
        )
        
    def update_quality_metrics(self, data: np.ndarray) -> None:
        """Update signal quality metrics"""
        for i, (name, config) in enumerate(self.channels.items()):
            channel_data = data[i]
            
            # Calculate quality metrics
            quality = SignalQualityMetrics.calculate_signal_quality(
                channel_data,
                self.sampling_rate,
                self.data_type
            )
            
            self.current_quality[name] = quality
            
    def detect_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect artifacts in all channels"""
        artifacts = np.zeros((self.n_channels, data.shape[1]), dtype=bool)
        
        for i in range(self.n_channels):
            artifacts[i], _ = ArtifactDetection.detect_artifacts(
                data[i],
                self.sampling_rate
            )
            
        return artifacts
        
    def process_heart_rate(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculate heart rate from PPG data"""
        if self.data_type != DataType.PPG:
            return None, None
            
        # Use IR channel for heart rate
        ir_index = 1  # Ambient, IR, Red
        ir_data = data[ir_index]
        
        return HeartRateAnalysis.calculate_heart_rate(
            ir_data,
            self.sampling_rate
        )
        
    def set_filter(self, filter_name: str) -> None:
        """Change current filter"""
        if filter_name.lower() == 'off':
            self.current_filter = 'off'
        elif filter_name.lower() in ProcessingConfig.FILTER_CONFIGS[self.data_type]:
            self.current_filter = filter_name.lower()
            
    def set_data_type(self, data_type: DataType) -> None:
        """Change data type and reinitialize"""
        self.data_type = data_type
        self.sampling_rate = StreamConfig.SAMPLING_RATES[data_type]
        self.channels = StreamConfig.CHANNELS[data_type]
        self.n_channels = len(self.channels)
        
        # Reinitialize
        self.setup_buffers()
        self.setup_filters()
        self.current_filter = 'default'
        self.current_quality = {}
        
    def enable_processing(self, enabled: bool) -> None:
        """Enable or disable data processing"""
        self.processing_enabled = enabled
        
    def clear_buffers(self) -> None:
        """Clear all data buffers"""
        self.raw_buffer.clear()
        for buffer in self.quality_buffers.values():
            buffer.clear()
            
        if hasattr(self, 'hr_buffer'):
            self.hr_buffer.clear()