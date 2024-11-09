from PyQt5.QtCore import QObject, QThread, QRecursiveMutex, QMutex, QMutexLocker, pyqtSignal
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
    # heart_rate: Optional[float] = None
    # heart_rate_confidence: Optional[float] = None
    # artifacts: Optional[np.ndarray] = None

class CircularBuffer:
    def __init__(self, channels: int, size: int):
        # Pre-allocate numpy arrays for better performance
        self.buffer = np.zeros((channels, size), dtype=np.float32)  # Use float32 instead of float64
        self.size = size
        self.position = 0
        self.filled = False
        self.lock = Lock()
    
    def add(self, data: np.ndarray) -> None:
        """Optimized buffer adding"""
        with self.lock:
            n_samples = data.shape[1]
            if n_samples >= self.size:
                # Fast path for large chunks
                self.buffer = data[:, -self.size:]
                self.position = 0
                self.filled = True
                return
                
            # Use numpy's optimized operations
            if self.position + n_samples <= self.size:
                self.buffer[:, self.position:self.position + n_samples] = data
                self.position += n_samples
            else:
                # Split the write operation
                first_part = self.size - self.position
                self.buffer[:, self.position:] = data[:, :first_part]
                remainder = n_samples - first_part
                if remainder > 0:
                    self.buffer[:, :remainder] = data[:, first_part:]
                self.position = remainder
            
            self.filled = self.filled or self.position >= self.size - 1
            
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
    processed_data = pyqtSignal(object)  # for processed data
    error_occurred = pyqtSignal(str)     # for error message
    
    def __init__(self, data_type: DataType):
        super().__init__()
        self.data_type = data_type  # Store as enum
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

        # Add minimum chunk size based on filter requirements
        self.min_chunk_size = 8
        self.data_accumulator = []
        
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
        if not self.processing_enabled or new_data is None:
            return
            
        try:
            # Process immediately without mutex lock for better performance
            data = new_data
            
            if self.current_filter != 'off':
                try:
                    data = self.apply_filter(data)
                except ValueError:
                    pass  # Skip filtering if chunk too small
            
            # Emit processed data immediately
            processed = ProcessedData(
                data=data,
                timestamp=timestamp
            )
            
            self.processed_data.emit(processed)
            
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply current filter to data with size check"""
        if self.current_filter not in self.filters:
            return data
            
        # Check if data chunk is large enough
        filter_coeffs = self.filters[self.current_filter]
        pad_len = len(filter_coeffs[0]) - 1
        
        if data.shape[1] <= pad_len:
            return data  # Return unfiltered data if chunk too small
            
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

class DataProcessorThread(QThread):
    """Thread for running data processing"""
    
    # Signals
    processed_data = pyqtSignal(object)  # For processed data
    error_occurred = pyqtSignal(str)
    
    def __init__(self, processor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.running = False
        
    def run(self):
        """Thread's main loop"""
        self.running = True
        while self.running:
            try:
                # Process data if available
                if hasattr(self.processor, 'process_data'):
                    result = self.processor.process_data()
                    if result is not None:
                        print(f"Sending processed data: {result.data.shape}")
                        self.processed_data.emit(result)
                
                # Sleep briefly to prevent CPU overuse
                self.msleep(10)
                
            except Exception as e:
                logging.error(f"Processing error: {str(e)}")
                self.error_occurred.emit(str(e))
                
    def stop(self):
        """Stop the processing thread"""
        self.running = False
        self.wait()