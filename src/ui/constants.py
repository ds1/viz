from enum import Enum
from typing import Dict, Tuple

class DataType(Enum):
    EEG = "EEG"
    PPG = "PPG"
    ACCELEROMETER = "ACCEL"
    GYROSCOPE = "GYRO"

class StreamConfig:
    SAMPLING_RATES: Dict[DataType, int] = {
        DataType.EEG: 256,
        DataType.PPG: 64,
        DataType.ACCELEROMETER: 52,
        DataType.GYROSCOPE: 52
    }
    
    CHANNEL_COUNTS: Dict[DataType, int] = {
        DataType.EEG: 4,
        DataType.PPG: 3,
        DataType.ACCELEROMETER: 3,
        DataType.GYROSCOPE: 3
    }
    
    CHANNEL_NAMES: Dict[DataType, list] = {
        DataType.EEG: ['TP9', 'FP1', 'FP2', 'TP10'],
        DataType.PPG: ['Ambient', 'IR', 'Red'],
        DataType.ACCELEROMETER: ['X', 'Y', 'Z'],
        DataType.GYROSCOPE: ['X', 'Y', 'Z']
    }
    
    VALUE_RANGES: Dict[DataType, Dict[str, Tuple[float, float]]] = {
        DataType.EEG: {
            'TP9': (-500, 500),
            'FP1': (-500, 500),
            'FP2': (-500, 500),
            'TP10': (-500, 500)
        },
        DataType.PPG: {
            'Ambient': (0, 4096),
            'IR': (0, 4096),
            'Red': (0, 4096)
        },
        DataType.ACCELEROMETER: {
            'X': (-2, 2),
            'Y': (-2, 2),
            'Z': (-2, 2)
        },
        DataType.GYROSCOPE: {
            'X': (-250, 250),
            'Y': (-250, 250),
            'Z': (-250, 250)
        }
    }

class ProcessingConfig:
    # Buffer sizes (in samples)
    BUFFER_SIZES: Dict[DataType, int] = {
        DataType.EEG: 1024,  # 4 seconds at 256 Hz
        DataType.PPG: 256,   # 4 seconds at 64 Hz
        DataType.ACCELEROMETER: 208,  # 4 seconds at 52 Hz
        DataType.GYROSCOPE: 208       # 4 seconds at 52 Hz
    }
    
    # Downsampling factors
    DOWNSAMPLE_FACTORS: Dict[DataType, int] = {
        DataType.EEG: 1,
        DataType.PPG: 1,
        DataType.ACCELEROMETER: 1,
        DataType.GYROSCOPE: 1
    }
    
    # Filter configurations
    FILTER_CONFIGS: Dict[DataType, Dict[str, Dict[str, Tuple[float, float]]]] = {
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

class DisplayConfig:
    # Time windows (in seconds)
    TIME_WINDOWS = [2, 4, 8]
    DEFAULT_TIME_WINDOW = 4
    
    # Vertical scale factors
    SCALE_FACTORS: Dict[DataType, list] = {
        DataType.EEG: [50, 100, 200, 500],  # µV
        DataType.PPG: [1000, 2000, 4000],   # Raw units
        DataType.ACCELEROMETER: [0.5, 1, 2], # g
        DataType.GYROSCOPE: [50, 100, 250]   # deg/s
    }
    
    # Update rates
    DISPLAY_REFRESH_RATE = 30  # Hz
    MINIMUM_REFRESH_INTERVAL = 1000 // DISPLAY_REFRESH_RATE  # ms

class ErrorMessages:
    STREAM_NOT_FOUND = "No {stream_type} stream found. Please check your device connection."
    STREAM_DISCONNECTED = "{stream_type} stream disconnected. Attempting to reconnect..."
    DATA_ERROR = "Error processing {stream_type} data: {error_details}"
    RECONNECT_ATTEMPT = "Attempting to reconnect to {stream_type} stream ({attempt}/{max_attempts})..."
