from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass

class DataType(Enum):
    EEG = "EEG"
    PPG = "PPG"
    ACCELEROMETER = "ACCEL"
    GYROSCOPE = "GYRO"

    @classmethod
    def from_string(cls, value: str) -> 'DataType':
        """Safely convert string to DataType"""
        try:
            return next(dt for dt in cls if dt.value == value)
        except StopIteration:
            raise ValueError(f"Invalid DataType value: {value}")

class StreamStatus(Enum):
    """LSL stream connection states"""
    DISCONNECTED = "Disconnected"
    SEARCHING = "Searching"
    CONNECTING = "Connecting"
    CONNECTED = "Connected"
    RECONNECTING = "Reconnecting"

@dataclass
class StreamChannelConfig:  # Renamed from ChannelConfig to avoid conflict
    """Channel configuration"""
    name: str
    unit: str
    range: Tuple[float, float]
    color: str

class StreamConfig:
    """Stream-specific configurations"""
    
    # Sampling rates for each data type
    SAMPLING_RATES: Dict[DataType, int] = {
        DataType.EEG: 256,
        DataType.PPG: 64,
        DataType.ACCELEROMETER: 52,
        DataType.GYROSCOPE: 52
    }
    
    # Channel configurations
    CHANNELS: Dict[DataType, Dict[str, StreamChannelConfig]] = {
        DataType.EEG: {
            'TP9': StreamChannelConfig('Left Ear', 'µV', (-500, 500), '#4CAF50'),
            'FP1': StreamChannelConfig('Left Forehead', 'µV', (-500, 500), '#2196F3'),
            'FP2': StreamChannelConfig('Right Forehead', 'µV', (-500, 500), '#F44336'),
            'TP10': StreamChannelConfig('Right Ear', 'µV', (-500, 500), '#FFC107'),
            'AUX': StreamChannelConfig('Auxiliary', 'µV', (-500, 500), '#9C27B0')
        },
        DataType.PPG: {
            'Ambient': StreamChannelConfig('Ambient', 'units', (0, 4096), '#9C27B0'),
            'IR': StreamChannelConfig('IR', 'units', (0, 4096), '#E91E63'),
            'Red': StreamChannelConfig('Red', 'units', (0, 4096), '#FF5722')
        },
        DataType.ACCELEROMETER: {
            'X': StreamChannelConfig('X-Axis', 'g', (-2, 2), '#2196F3'),
            'Y': StreamChannelConfig('Y-Axis', 'g', (-2, 2), '#4CAF50'),
            'Z': StreamChannelConfig('Z-Axis', 'g', (-2, 2), '#FFC107')
        },
        DataType.GYROSCOPE: {
            'X': StreamChannelConfig('X-Axis', '°/s', (-250, 250), '#2196F3'),
            'Y': StreamChannelConfig('Y-Axis', '°/s', (-250, 250), '#4CAF50'),
            'Z': StreamChannelConfig('Z-Axis', '°/s', (-250, 250), '#FFC107')
        }
    }

class ProcessingConfig:
    """Signal processing configurations"""
    
    # Buffer sizes in samples
    BUFFER_SIZES: Dict[DataType, int] = {
        DataType.EEG: 1024,  # 4 seconds at 256 Hz
        DataType.PPG: 256,   # 4 seconds at 64 Hz
        DataType.ACCELEROMETER: 208,  # 4 seconds at 52 Hz
        DataType.GYROSCOPE: 208       # 4 seconds at 52 Hz
    }
    
    # Filter configurations
    FILTER_CONFIGS: Dict[DataType, Dict[str, Dict]] = {
        DataType.EEG: {
            'default': {'bandpass': (1, 50)},
            'delta': {'bandpass': (1, 4)},
            'theta': {'bandpass': (4, 8)},
            'alpha': {'bandpass': (8, 13)},
            'beta': {'bandpass': (13, 30)},
            'gamma': {'bandpass': (30, 50)}
        },
        DataType.PPG: {
            'default': {'bandpass': (0.5, 5)},
            'heart_rate': {'bandpass': (0.7, 3.5)}
        },
        DataType.ACCELEROMETER: {
            'default': {'lowpass': 20},
            'movement': {'bandpass': (0.5, 10)}
        },
        DataType.GYROSCOPE: {
            'default': {'lowpass': 20},
            'movement': {'bandpass': (0.5, 10)}
        }
    }
    
    # Filter names for display
    FILTER_NAMES: Dict[DataType, List[str]] = {
        DataType.EEG: ['Off', 'Default', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
        DataType.PPG: ['Off', 'Default', 'Heart Rate'],
        DataType.ACCELEROMETER: ['Off', 'Default', 'Movement'],
        DataType.GYROSCOPE: ['Off', 'Default', 'Movement']
    }
    
    # Quality thresholds
    QUALITY_THRESHOLDS: Dict[str, float] = {
        'excellent': 0.8,
        'good': 0.6,
        'fair': 0.4,
        'poor': 0.0
    }
    
    # Processing parameters
    ARTIFACT_WINDOW = 0.5  # seconds
    QUALITY_WINDOW = 1.0   # seconds
    MIN_SAMPLES = 32       # minimum samples for processing

class DisplayConfig:
    """Visualization configurations"""
    
    # Time windows in seconds
    TIME_WINDOWS = [2, 4, 8]
    DEFAULT_TIME_WINDOW = 4
    
    # Vertical scale factors
    SCALE_FACTORS = [0.5, 1.0, 2.0, 5.0]
    DEFAULT_SCALE = 1.0
    MIN_SCALE = 0.1
    MAX_SCALE = 10.0
    
    # Display update rate
    DISPLAY_REFRESH_RATE = 60  # Hz
    MINIMUM_REFRESH_INTERVAL = 1000 // DISPLAY_REFRESH_RATE  # ms
    
    # Grid configuration
    MAJOR_GRID_INTERVAL = 1.0  # seconds
    MINOR_GRID_INTERVAL = 0.2  # seconds
    
    # Animation
    SMOOTH_SCROLL = True
    SCROLL_FACTOR = 0.95  # For smooth scrolling
    
    # Quality visualization
    QUALITY_COLORS = {
        'excellent': '#4CAF50',
        'good': '#8BC34A',
        'fair': '#FFC107',
        'poor': '#F44336'
    }

class ErrorMessages:
    """Standard error messages"""
    
    STREAM_NOT_FOUND = "No {stream_type} stream found"
    CONNECTION_FAILED = "Failed to connect to {stream_type} stream: {error}"
    STREAM_DISCONNECTED = "{stream_type} stream disconnected"
    DATA_ERROR = "Error processing {stream_type} data: {error}"
    RECONNECTING = "Attempting to reconnect ({attempt}/{max_attempts})"
    QUALITY_WARNING = "Poor signal quality detected on channels: {channels}"
    DEVICE_ERROR = "Device error: {error}"