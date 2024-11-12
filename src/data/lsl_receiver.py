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
        self.data = np.zeros((channel_count, buffer_size), dtype=np.float32)
        self.timestamps = np.zeros(buffer_size, dtype=np.float64)
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
                np.copyto(self.data, samples[:, -self.buffer_size:])
                np.copyto(self.timestamps, timestamps[-self.buffer_size:])
                self.position = self.buffer_size
                return
                
            space_left = self.buffer_size - self.position
            if n_samples <= space_left:
                self.data[:, self.position:self.position + n_samples] = samples
                self.timestamps[self.position:self.position + n_samples] = timestamps
                self.position += n_samples
            else:
                # Split the data using efficient numpy operations
                first_part = space_left
                second_part = n_samples - space_left
                
                np.copyto(self.data[:, self.position:], samples[:, :first_part])
                np.copyto(self.timestamps[self.position:], timestamps[:first_part])
                
                np.copyto(self.data[:, :second_part], samples[:, first_part:])
                np.copyto(self.timestamps[:second_part], timestamps[first_part:])
                
                self.position = second_part
                
    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current buffer contents"""
        with self.lock:
            return self.data[:, :self.position].copy(), self.timestamps[:self.position].copy()
            
    def clear(self) -> None:
        """Clear buffer contents"""
        with self.lock:
            self.data.fill(0)
            self.timestamps.fill(0)
            self.position = 0

class LSLReceiver(QObject):
    """Robust LSL stream receiver with automatic reconnection"""
    
    # Signals
    data_ready = pyqtSignal(object, object)  # for numpy arrays
    status_changed = pyqtSignal(object)      # for StreamStatus
    stream_info_updated = pyqtSignal(object) # for stream info
    error_occurred = pyqtSignal(str)         # for error message
    quality_updated = pyqtSignal(object)     # for quality dict
    
    def __init__(self, data_type: DataType, buffer_size: Optional[int] = None,
                 auto_reconnect: bool = True):
        """Initialize LSL receiver
        
        Args:
            data_type: DataType enum specifying the type of data stream
            buffer_size: Optional buffer size override
            auto_reconnect: Whether to automatically attempt reconnection
        """
        super().__init__()
        
        # Store data type and get configuration
        self.data_type = data_type
        self.stream_type = data_type.value  # Convert enum to string for LSL
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
        
        # Optimize chunk sizes
        self.min_chunk_size = 8
        self.max_chunklen = self.buffer_size // 4

        self.min_samples_required = 64  # Should be greater than max filter order + padding
        self.samples_accumulated = 0

    def set_stream_type(self, data_type: DataType):
        """Update stream type and related configurations"""
        self.data_type = data_type
        self.stream_type = data_type.value
        self.required_channels = len(StreamConfig.CHANNELS[data_type])
        self.sampling_rate = StreamConfig.SAMPLING_RATES[data_type]
        
        # Update buffer if needed
        new_buffer_size = ProcessingConfig.BUFFER_SIZES[data_type]
        if new_buffer_size != self.buffer_size:
            self.buffer_size = new_buffer_size
            self.buffer = StreamBuffer(self.required_channels, self.buffer_size)
        
    def connect_to_stream(self) -> None:
        """Connect to LSL stream"""
        self._update_status(StreamStatus.SEARCHING)
        
        try:
            streams = resolve_stream('type', self.stream_type)
            
            if not streams:
                msg = ErrorMessages.STREAM_NOT_FOUND.format(
                    stream_type=self.stream_type
                )
                self.error_occurred.emit(msg)
                if self.auto_reconnect:
                    self._start_reconnection()
                return
            
            if not self._validate_stream(streams[0]):
                return
                
            self.inlet = StreamInlet(
                streams[0],
                max_buflen=self.buffer_size,
                max_chunklen=self.max_chunklen,
                recover=True
            )
            
            stream = streams[0]
            self.stream_info = StreamInfo(
                name=stream.name(),
                type=self.stream_type,
                channel_count=stream.channel_count(),
                sampling_rate=stream.nominal_srate(),
                source_id=stream.source_id(),
                created_at=local_clock(),
                version=stream.version()
            )
            
            self.connected = True
            self._update_status(StreamStatus.CONNECTED)
            self.current_reconnect_attempt = 0
            self.stream_info_updated.emit(self.stream_info)
            
            self.receive_data()
            
        except Exception as e:
            logging.error(f"Connection error: {str(e)}")
            self.error_occurred.emit(str(e))
            if self.auto_reconnect:
                self._start_reconnection()
    
    def receive_data(self) -> None:
        """Optimized data reception with minimum sample requirement"""
        accumulated_samples = []
        accumulated_timestamps = []
        logging.info("Starting data reception")
        while self.connected and self.inlet:
            try:
                samples, timestamps = self.inlet.pull_chunk(
                    timeout=0.0,
                    max_samples=self.max_chunklen
                )
                
                if samples:
                    if not accumulated_samples:  # Only log first accumulation
                        logging.debug(f"Received chunk: samples shape {np.array(samples).shape}")
                    
                    # Efficient numpy operations
                    samples = np.asarray(samples, dtype=np.float32).T
                    timestamps = np.asarray(timestamps, dtype=np.float64)
                    
                    # Accumulate samples
                    accumulated_samples.append(samples)
                    accumulated_timestamps.append(timestamps)
                    self.samples_accumulated += samples.shape[1]
                    
                    # Only emit when we have enough samples
                    if self.samples_accumulated >= self.min_samples_required:
                        # Concatenate accumulated data
                        combined_samples = np.hstack(accumulated_samples)
                        combined_timestamps = np.concatenate(accumulated_timestamps)
                        
                        logging.debug(f"Emitting data: shape {combined_samples.shape}, "
                                    f"timestamps range {combined_timestamps[0]:.3f} to {combined_timestamps[-1]:.3f}")
                        
                        # Update state and emit
                        self.last_timestamp = combined_timestamps[-1]
                        self.sample_count += len(combined_timestamps)
                        self.data_ready.emit(combined_samples, combined_timestamps)
                        
                        # Reset accumulation
                        accumulated_samples = []
                        accumulated_timestamps = []
                        self.samples_accumulated = 0
                    
            except Exception as e:
                logging.error(f"Data receiving error: {str(e)}")
                self._handle_connection_loss()
                break

    def _validate_stream(self, stream_info) -> bool:
        """Validate stream parameters"""
        try:
            if stream_info.channel_count() < self.required_channels:
                msg = f"Insufficient channels: {stream_info.channel_count()}, minimum {self.required_channels} required"
                self.error_occurred.emit(msg)
                return False
                
            if abs(stream_info.nominal_srate() - self.sampling_rate) > 1:
                msg = f"Invalid sampling rate: {stream_info.nominal_srate()}, expected {self.sampling_rate}"
                self.error_occurred.emit(msg)
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Stream validation error: {str(e)}")
            self.error_occurred.emit(str(e))
            return False
    
    def check_stream_health(self) -> None:
        """Monitor stream health"""
        if self.connected:
            if local_clock() - self.last_timestamp > 2.0:
                logging.warning("Data timeout detected")
                self._handle_connection_loss()
                
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
        self.connect_to_stream()
        
        if not self.connected:
            self.reconnect_timer.start(self.reconnect_interval)
            
    def _update_status(self, status: StreamStatus):
        """Update and emit status"""
        self.status = status
        self.status_changed.emit(status)
        
    def get_stream_info(self) -> Optional[StreamInfo]:
        """Get current stream information"""
        return self.stream_info
        
    def disconnect(self) -> None:
        """Clean disconnection"""
        self.connected = False
        self.health_check_timer.stop()
        self.reconnect_timer.stop()
        
        if self.inlet:
            self.inlet = None
            
        self.buffer.clear()
        self._update_status(StreamStatus.DISCONNECTED)