from pylsl import StreamInlet, resolve_stream, local_clock, TimeoutError
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import logging
from threading import Lock

from src.constants import (
    DataType, StreamConfig, StreamStatus,
    ProcessingConfig, ErrorMessages
)
# Add if you need specific types:
from src.custom_types import Timestamp, SignalData, StreamMetadata

@dataclass
class StreamInfo:
    """LSL stream information"""
    name: str
    type: DataType
    channel_count: int
    sampling_rate: float
    source_id: str
    created_at: float
    version: str

class StreamBuffer:
    """Thread-safe buffer for LSL data"""
    
    def __init__(self, channel_count: int, buffer_size: int):
        self.data = np.zeros((channel_count, buffer_size))
        self.timestamps = np.zeros(buffer_size)
        self.position = 0
        self.buffer_size = buffer_size
        self.lock = Lock()
        
    def add(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        """Add new samples to buffer"""
        with self.lock:
            n_samples = len(timestamps)
            if n_samples == 0:
                return
                
            if n_samples >= self.buffer_size:
                self.data = samples[:, -self.buffer_size:]
                self.timestamps = timestamps[-self.buffer_size:]
                self.position = self.buffer_size
                return
                
            space_left = self.buffer_size - self.position
            if n_samples <= space_left:
                self.data[:, self.position:self.position + n_samples] = samples
                self.timestamps[self.position:self.position + n_samples] = timestamps
                self.position += n_samples
            else:
                # Split the data
                first_part = space_left
                second_part = n_samples - space_left
                
                # Add first part at end
                self.data[:, self.position:] = samples[:, :first_part]
                self.timestamps[self.position:] = timestamps[:first_part]
                
                # Add second part at beginning
                self.data[:, :second_part] = samples[:, first_part:]
                self.timestamps[:second_part] = timestamps[first_part:]
                
                self.position = second_part
                
    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current buffer contents"""
        with self.lock:
            return self.data[:, :self.position], self.timestamps[:self.position]
            
    def clear(self) -> None:
        """Clear buffer contents"""
        with self.lock:
            self.data.fill(0)
            self.timestamps.fill(0)
            self.position = 0

class LSLReceiver(QObject):
    """Robust LSL stream receiver with automatic reconnection"""
    
    # Signals use simple types or object
    data_ready = pyqtSignal(object, object)  # for numpy arrays
    status_changed = pyqtSignal(object)         # for StreamStatus
    stream_info_updated = pyqtSignal(object) # for stream info
    error_occurred = pyqtSignal(str)         # for error message
    quality_updated = pyqtSignal(object)     # for quality dict
    
    def __init__(self, stream_type: str, buffer_size: Optional[int] = None,
                 auto_reconnect: bool = True):
        super().__init__()
        self.stream_type = stream_type  # Store as string
        # Convert to enum for config lookups
        data_type = DataType(stream_type)
        self.required_channels = len(StreamConfig.CHANNELS[data_type])
        self.sampling_rate = StreamConfig.SAMPLING_RATES[data_type]
        self.buffer_size = buffer_size or ProcessingConfig.BUFFER_SIZES[data_type]
        self.auto_reconnect = auto_reconnect
        
        # State
        self.inlet: Optional[StreamInlet] = None
        self.status = StreamStatus.DISCONNECTED
        self.connected = False
        self.stream_info: Optional[StreamInfo] = None
        self.last_timestamp = 0
        self.sample_count = 0
        
        # Buffer
        self.buffer = StreamBuffer(self.required_channels, self.buffer_size)
        
        # Reconnection settings
        self.max_reconnect_attempts = 5
        self.reconnect_interval = 2000  # ms
        self.current_reconnect_attempt = 0
        
        # Quality monitoring
        self.quality_check_interval = 1000  # ms
        self.quality_threshold = ProcessingConfig.QUALITY_THRESHOLDS['fair']
        self.last_quality_check = local_clock()
        
        # Initialize timers
        self.setup_timers()
    
    def set_stream_type(self, stream_type: Union[str, DataType]):
        """Update stream type"""
        if isinstance(stream_type, DataType):
            self.stream_type = stream_type.value
            self.data_type = stream_type
        else:
            self.stream_type = stream_type
            self.data_type = DataType(stream_type)

    def setup_timers(self) -> None:
        """Initialize monitoring timers"""
        # Timer for checking stream health
        self.health_check_timer = QTimer(self)
        self.health_check_timer.timeout.connect(self.check_stream_health)
        self.health_check_timer.start(1000)  # Check every second
        
        # Timer for automatic reconnection
        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.timeout.connect(self.attempt_reconnect)
        
        # Timer for quality monitoring
        self.quality_timer = QTimer(self)
        self.quality_timer.timeout.connect(self.check_signal_quality)
        self.quality_timer.start(self.quality_check_interval)
        
    def connect_to_stream(self) -> None:
        """Connect to LSL stream"""
        self._update_status(StreamStatus.SEARCHING)
        
        try:
            # Find streams matching type
            logging.debug(f"Searching for stream type: {self.stream_type}")
            streams = resolve_stream('type', self.stream_type)
            
            if not streams:
                logging.debug("No streams found")
                msg = ErrorMessages.STREAM_NOT_FOUND.format(
                    stream_type=self.stream_type
                )
                self.error_occurred.emit(msg)
                if self.auto_reconnect:
                    self._start_reconnection()
                return
            
            logging.debug(f"Found {len(streams)} streams")
                
            # Validate stream
            if not self._validate_stream(streams[0]):
                return
                
            # Create inlet
            self.inlet = StreamInlet(
                streams[0],
                max_buflen=self.buffer_size,
                max_chunklen=self.buffer_size // 4,
                recover=True
            )
            
            # Get stream info
            stream = streams[0]  # StreamInfo object is already the info
            self.stream_info = StreamInfo(
                name=stream.name(),
                type=self.stream_type,
                channel_count=stream.channel_count(),
                sampling_rate=stream.nominal_srate(),
                source_id=stream.source_id(),
                created_at=local_clock(),
                version=stream.version()
            )
            
            # Update state
            self.connected = True
            self._update_status(StreamStatus.CONNECTED)
            self.current_reconnect_attempt = 0
            self.stream_info_updated.emit(self.stream_info)
            
            # Start receiving data
            self.receive_data()
            
        except Exception as e:
            logging.error(f"Connection error: {str(e)}")
            self.error_occurred.emit(str(e))
            if self.auto_reconnect:
                self._start_reconnection()
                
    def receive_data(self) -> None:
        """Receive data from stream"""
        while self.connected and self.inlet:
            try:
                # Pull chunk of samples
                samples, timestamps = self.inlet.pull_chunk(
                    timeout=0.0,
                    max_samples=self.buffer_size // 4
                )
                
                if samples:
                    logging.debug(f"Received {len(samples)} samples")

                    # Convert to numpy arrays
                    samples = np.array(samples).T
                    timestamps = np.array(timestamps)
                    
                    # Add to buffer
                    self.buffer.add(samples, timestamps)

                    logging.debug(f"Data shape: {samples.shape}")
                    
                    # Update state
                    self.last_timestamp = timestamps[-1]
                    self.sample_count += len(timestamps)
                    
                    # Emit data
                    self.data_ready.emit(samples, timestamps)
                    
            except TimeoutError:
                # Normal timeout, continue
                continue
                
            except Exception as e:
                logging.error(f"Data receiving error: {str(e)}")
                self.error_occurred.emit(str(e))
                self._handle_connection_loss()
                break
                
    def _validate_stream(self, stream_info) -> bool:
        """Validate stream parameters"""
        try:
            logging.debug(f"Validating stream: {stream_info.name()} ({stream_info.type()})")
            logging.debug(f"Channel count: {stream_info.channel_count()}")
            logging.debug(f"Sampling rate: {stream_info.nominal_srate()}")
            
            if stream_info.channel_count() < 4:  # Require minimum channels
                msg = f"Insufficient channels: {stream_info.channel_count()}, minimum 4 required"
                logging.error(msg)
                self.error_occurred.emit(msg)
                return False
                
            if abs(stream_info.nominal_srate() - self.sampling_rate) > 1:
                msg = f"Invalid sampling rate: {stream_info.nominal_srate()}, expected {self.sampling_rate}"
                logging.error(msg)
                self.error_occurred.emit(msg)
                return False
                
            logging.debug("Stream validation successful")
            return True
            
        except Exception as e:
            logging.error(f"Stream validation error: {str(e)}")
            self.error_occurred.emit(str(e))
            return False
            
    def check_stream_health(self) -> None:
        """Monitor stream health"""
        if self.connected:
            current_time = local_clock()
            
            # Check for data timeout
            if current_time - self.last_timestamp > 2.0:  # 2 second timeout
                logging.warning("Data timeout detected")
                self._handle_connection_loss()
                
    def check_signal_quality(self) -> None:
        """Monitor signal quality"""
        if self.connected:
            data, _ = self.buffer.get_data()
            if len(data) == 0:
                return
                
            # Calculate quality for each channel
            qualities = {}
            for i, name in enumerate(StreamConfig.CHANNELS[self.stream_type]):
                # Simple quality metric based on signal variance
                quality = np.minimum(1.0, np.var(data[i]) / self.quality_threshold)
                qualities[name] = quality
                
            # Emit quality metrics
            self.quality_updated.emit(qualities)
            
            # Check for poor quality
            poor_channels = [
                name for name, quality in qualities.items()
                if quality < self.quality_threshold
            ]
            
            if poor_channels:
                msg = ErrorMessages.QUALITY_WARNING.format(
                    channels=", ".join(poor_channels)
                )
                self.error_occurred.emit(msg)
                
    def _handle_connection_loss(self) -> None:
        """Handle connection loss"""
        self.connected = False
        if self.inlet:
            self.inlet = None
            
        self._update_status(StreamStatus.DISCONNECTED)
        
        if self.auto_reconnect:
            self._start_reconnection()
            
    def _start_reconnection(self) -> None:
        """Initialize reconnection process"""
        if not self.reconnect_timer.isActive():
            self.current_reconnect_attempt = 0
            self.attempt_reconnect()
            
    def attempt_reconnect(self) -> None:
        """Attempt to reconnect to stream"""
        if self.connected or self.current_reconnect_attempt >= self.max_reconnect_attempts:
            self.reconnect_timer.stop()
            return
            
        self.current_reconnect_attempt += 1
        self._update_status(StreamStatus.RECONNECTING)
        
        msg = ErrorMessages.RECONNECTING.format(
            attempt=self.current_reconnect_attempt,
            max_attempts=self.max_reconnect_attempts
        )
        logging.info(msg)
        
        self.connect_to_stream()
        
        if not self.connected:
            self.reconnect_timer.start(self.reconnect_interval)
            
    def _update_status(self, status: StreamStatus):
        """Update and emit status"""
        self.status = status
        self.status_changed.emit(status)  # Now emits StreamStatus object
        
    def get_stream_info(self) -> Optional[StreamInfo]:
        """Get current stream information"""
        return self.stream_info
        
    def disconnect(self) -> None:
        """Clean disconnection"""
        self.connected = False
        self.health_check_timer.stop()
        self.reconnect_timer.stop()
        self.quality_timer.stop()
        
        if self.inlet:
            self.inlet = None
            
        self.buffer.clear()
        self._update_status(StreamStatus.DISCONNECTED)