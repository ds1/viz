import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QMutex, QTimer
from typing import Optional, Dict, Any
from scipy import signal
from dataclasses import dataclass
from collections import deque
import logging

from src.constants import DataType, ProcessingConfig, StreamConfig
from src.utils import create_bandpass_filter, create_lowpass_filter

@dataclass
class ProcessedData:
    data: np.ndarray
    timestamp: float
    data_type: DataType
    quality: Dict[str, float]  # Channel quality metrics

class CircularBuffer:
    """Efficient circular buffer implementation for real-time data"""
    
    def __init__(self, channels: int, size: int):
        self.buffer = np.zeros((channels, size))
        self.size = size
        self.position = 0
        self.filled = False
        
    def add(self, data: np.ndarray):
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
        
    def get_data(self) -> np.ndarray:
        if not self.filled and self.position == 0:
            return None
        if self.position == 0 or not self.filled:
            return self.buffer[:, :self.position]
        return np.concatenate((
            self.buffer[:, self.position:],
            self.buffer[:, :self.position]
        ), axis=1)

class DataProcessor(QObject):
    """Processes real-time data streams with efficient buffering and filtering"""
    
    processed_data = pyqtSignal(ProcessedData)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, data_type: DataType):
        super().__init__()
        
        # Configuration
        self.data_type = data_type
        self.sampling_rate = StreamConfig.SAMPLING_RATES[data_type]
        self.n_channels = StreamConfig.CHANNEL_COUNTS[data_type]
        self.buffer_size = ProcessingConfig.BUFFER_SIZES[data_type]
        
        # Initialize buffers
        self.raw_buffer = CircularBuffer(self.n_channels, self.buffer_size)
        self.processed_buffers: Dict[str, CircularBuffer] = {}
        
        # Initialize filters
        self.filters = self._initialize_filters()
        
        # State
        self.current_filter = 'default'
        self.mutex = QMutex()
        self.processing_enabled = True
        
        # Quality monitoring
        self.quality_window = int(self.sampling_rate * 1)  # 1 second
        self.quality_buffers = {
            ch: deque(maxlen=self.quality_window)
            for ch in StreamConfig.CHANNEL_NAMES[data_type]
        }
        
    def _initialize_filters(self) -> Dict[str, Any]:
        """Initialize filters based on configuration"""
        filters = {}
        filter_configs = ProcessingConfig.FILTER_CONFIGS[self.data_type]
        
        for name, config in filter_configs.items():
            if 'bandpass' in config:
                filters[name] = create_bandpass_filter(
                    self.sampling_rate,
                    config['bandpass'][0],
                    config['bandpass'][1]
                )
            elif 'lowpass' in config:
                filters[name] = create_lowpass_filter(
                    self.sampling_rate,
                    config['lowpass']
                )
                
        return filters
        
    def process_data(self, new_data: np.ndarray, timestamp: float):
        """Process new data with thread safety"""
        if not self.processing_enabled or new_data is None:
            return
            
        try:
            with QMutex():
                # Add to raw buffer
                self.raw_buffer.add(new_data)
                
                # Update signal quality metrics
                self._update_quality_metrics(new_data)
                
                # Apply current filter
                if self.current_filter in self.filters:
                    filtered_data = self._apply_filter(
                        self.raw_buffer.get_data(),
                        self.current_filter
                    )
                else:
                    filtered_data = self.raw_buffer.get_data()
                
                # Create processed data object
                processed = ProcessedData(
                    data=filtered_data,
                    timestamp=timestamp,
                    data_type=self.data_type,
                    quality=self._get_quality_metrics()
                )
                
                # Emit processed data
                self.processed_data.emit(processed)
                
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            self.error_occurred.emit(str(e))
            
    def _apply_filter(self, data: np.ndarray, filter_name: str) -> np.ndarray:
        """Apply specified filter to data"""
        if data is None or len(data) == 0:
            return np.array([])
            
        filt = self.filters[filter_name]
        return signal.filtfilt(filt[0], filt[1], data, axis=1)
        
    def _update_quality_metrics(self, new_data: np.ndarray):
        """Update signal quality metrics for each channel"""
        for i, channel in enumerate(StreamConfig.CHANNEL_NAMES[self.data_type]):
            # Calculate quality metric (e.g., variance, amplitude)
            quality = np.var(new_data[i])
            self.quality_buffers[channel].append(quality)
            
    def _get_quality_metrics(self) -> Dict[str, float]:
        """Get current quality metrics for all channels"""
        return {
            channel: np.mean(list(buffer))
            for channel, buffer in self.quality_buffers.items()
        }
        
    def set_filter(self, filter_name: str):
        """Change current filter"""
        if filter_name in self.filters or filter_name == 'none':
            self.current_filter = filter_name
            
    def enable_processing(self, enabled: bool):
        """Enable or disable data processing"""
        self.processing_enabled = enabled

class DataProcessorThread(QThread):
    """Thread for running the data processor"""
    
    def __init__(self, processor: DataProcessor):
        super().__init__()
        self.processor = processor
        self.running = True
        
    def run(self):
        while self.running:
            # Process would be driven by incoming data
            self.msleep(1)
            
    def stop(self):
        self.running = False
        self.processor.enable_processing(False)
        self.wait()
