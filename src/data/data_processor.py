from PyQt5.QtCore import QObject, QThread, QRecursiveMutex, QMutex, QMutexLocker, pyqtSignal
import numpy as np
from typing import Optional, Dict, Any, Tuple
from collections import deque
import logging
from dataclasses import dataclass
from threading import Lock

from src.data.utils import (
    FilterUtils, SpectralAnalysis, ArtifactDetection, HeartRateAnalysis
)
from src.constants import (
    DataType, ProcessingConfig, StreamConfig,
    DisplayConfig
)
# Add if you need specific types:
from src.custom_types import ProcessedSignal, FilterParameters

@dataclass
class ProcessedData:
    """Container for processed data and metrics"""
    data: np.ndarray
    timestamp: float

class DataProcessor(QObject):
    """Processes real-time data streams with signal quality monitoring"""
    
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
        
        # Buffer initialization
        buffer_size = ProcessingConfig.BUFFER_SIZES[self.data_type]
        self.raw_buffer = np.zeros((self.n_channels, buffer_size))
        self.buffer_position = 0
        self.buffer_full = False
        
        # Initialize filters
        self.setup_filters()
        
    def process_data(self, new_data: np.ndarray, timestamp: float) -> Optional[ProcessedData]:
        """Process new data and return processed result"""
        if not self.processing_enabled or new_data is None:
            return None
            
        try:
            with QMutexLocker(self.mutex):
                logging.debug(f"Processing data shape: {new_data.shape}")
                
                # Add to buffer
                self.raw_buffer[:, self.buffer_position:self.buffer_position + new_data.shape[1]] = new_data
                self.buffer_position = (self.buffer_position + new_data.shape[1]) % self.raw_buffer.shape[1]
                
                # Process data
                if self.current_filter != 'off':
                    try:
                        data = self.apply_filter(new_data)
                        logging.debug("Filter applied successfully")
                    except ValueError as e:
                        logging.warning(f"Filter error: {str(e)}, using unfiltered data")
                        data = new_data
                else:
                    data = new_data
                
                processed = ProcessedData(
                    data=data,
                    timestamp=timestamp
                )
                
                logging.debug(f"Processed data shape: {processed.data.shape}")
                return processed

        except Exception as e:
            logging.error(f"Error processing data: {str(e)}", exc_info=True)
            raise
            
    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply current filter to data"""
        if self.current_filter not in self.filters:
            return data
            
        try:
            filter_coeffs = self.filters[self.current_filter]
            pad_length = max(3 * max(len(filter_coeffs[0]), len(filter_coeffs[1])), 32)
            
            if data.shape[1] <= pad_length:
                return data
                
            return FilterUtils.apply_filter_with_mirror(
                data,
                filter_coeffs[0],
                filter_coeffs[1],
                pad_length=pad_length
            )
            
        except ValueError as e:
            logging.warning(f"Filtering error: {str(e)}")
            return data
            
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
        
        # Reinitialize buffer
        buffer_size = ProcessingConfig.BUFFER_SIZES[data_type]
        self.raw_buffer = np.zeros((self.n_channels, buffer_size))
        self.buffer_position = 0
        self.buffer_full = False
        
        # Reinitialize filters
        self.setup_filters()
        self.current_filter = 'default'
        self.current_quality = {}
        
    def enable_processing(self, enabled: bool) -> None:
        """Enable or disable data processing"""
        self.processing_enabled = enabled
        
    def clear_buffers(self) -> None:
        """Clear all data buffers"""
        self.raw_buffer.fill(0)
        self.buffer_position = 0
        self.buffer_full = False
