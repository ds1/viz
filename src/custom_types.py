from typing import TypedDict, Union, NewType
from numpy import ndarray
from dataclasses import dataclass
from datetime import datetime

# Custom types
Timestamp = NewType('Timestamp', float)
SignalValue = NewType('SignalValue', float)
ChannelName = NewType('ChannelName', str)

# Signal processing types
@dataclass
class ProcessedSignal:
    data: ndarray
    timestamp: Timestamp
    sampling_rate: float
    channel_count: int

@dataclass
class FilterParameters:
    order: int
    cutoff_low: float
    cutoff_high: float
    filter_type: str

@dataclass
class QualityMetrics:
    noise_level: float
    artifact_ratio: float
    signal_strength: float
    overall_quality: float
    timestamp: Timestamp

# Configuration types
class ChannelConfig(TypedDict):
    name: str
    unit: str
    range: tuple[float, float]
    color: str

class DisplayConfig(TypedDict):
    time_window: float
    vertical_scale: float
    refresh_rate: int
    grid_opacity: float

# Data types
SignalData = Union[ndarray, list[float]]
TimePoints = Union[ndarray, list[float]]
ChannelData = dict[ChannelName, SignalData]

@dataclass
class StreamMetadata:
    stream_name: str
    device_id: str
    start_time: datetime
    sample_count: int
    dropped_samples: int
