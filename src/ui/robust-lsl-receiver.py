from pylsl import StreamInlet, resolve_stream, local_clock, TimeoutError
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import numpy as np
from typing import Optional, Dict, Tuple, List
import logging
from dataclasses import dataclass
from time import time

from src.constants import DataType, StreamConfig, ErrorMessages

@dataclass
class StreamInfo:
    name: str
    type: str
    channel_count: int
    sampling_rate: float
    source_id: str

class StreamStatus:
    DISCONNECTED = "Disconnected"
    SEARCHING = "Searching"
    CONNECTING = "Connecting"
    CONNECTED = "Connected"
    RECONNECTING = "Reconnecting"

class LSLReceiver(QObject):
    """Robust LSL stream receiver with automatic reconnection"""
    
    # Signals
    data_received = pyqtSignal(np.ndarray, float)  # data, timestamp
    status_changed = pyqtSignal(str)  # status message
    stream_info_updated = pyqtSignal(StreamInfo)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, stream_type: DataType):
        super().__init__()
        
        # Configuration
        self.stream_type = stream_type
        self.required_channels = StreamConfig.CHANNEL_COUNTS[stream_type]
        self.expected_sampling_rate = StreamConfig.SAMPLING_RATES[stream_type]
        
        # State
        self.inlet: Optional[StreamInlet] = None
        self.status = StreamStatus.DISCONNECTED
        self.connected = False
        self.last_sample_time = 0
        self.sample_count = 0
        
        # Reconnection settings
        self.max_reconnect_attempts = 5
        self.reconnect_interval = 2000  # ms
        self.current_reconnect_attempt = 0
        
        # Quality monitoring
        self.timeout_threshold = 2.0  # seconds
        self.last_data_time = time()
        
        # Initialize timers
        self.setup_timers()
        
    def setup_timers(self):
        """Initialize monitoring timers"""
        # Timer for checking connection health
        self.health_check_timer = QTimer()
        self.health_check_timer.timeout.connect(self.check_connection_health)
        self.health_check_timer.start(1000)  # Check every second
        
        # Timer for automatic reconnection
        self.reconnect_timer = QTimer()
        self.reconnect_timer.timeout.connect(self.attempt_reconnect)
        
    def connect_to_stream(self):
        """Initial connection attempt"""
        self._update_status(StreamStatus.SEARCHING)
        try:
            streams = resolve_stream('type', self.stream_type.value)
        if len(streams) > 0:
                self._update_status(StreamStatus.CONNECTING)
                
                # Validate stream
                stream_info = streams[0].info()
                if not self._validate_stream(stream_info):
                    raise ValueError("Stream validation failed")
                
                # Create inlet
                self.inlet = StreamInlet(
                    streams[0],
                    max_buflen=StreamConfig.SAMPLING_RATES[self.stream_type],
                    max_chunklen=StreamConfig.SAMPLING_RATES[self.stream_type] // 4
                )
                
                # Update state
                self.connected = True
                self._update_status(StreamStatus.CONNECTED)
                self.current_reconnect_attempt = 0
                
                # Emit stream info
                self.stream_info_updated.emit(StreamInfo(
                    name=stream_info.name(),
                    type=stream_info.type(),
                    channel_count=stream_info.channel_count(),
                    sampling_rate=stream_info.nominal_srate(),
                    source_id=stream_info.source_id()
                ))
                
                # Start receiving data
                self.start_receiving()
                
            else:
                msg = ErrorMessages.STREAM_NOT_FOUND.format(
                    stream_type=self.stream_type.value
                )
                self.error_occurred.emit(msg)
                self._start_reconnection()
                
        except Exception as e:
            logging.error(f"Connection error: {str(e)}")
            self.error_occurred.emit(str(e))
            self._start_reconnection()
            
    def _validate_stream(self, info) -> bool:
        """Validate stream parameters"""
        try:
            # Check channel count
            if info.channel_count() != self.required_channels:
                logging.error(f"Invalid channel count: {info.channel_count()}")
                return False
                
            # Check sampling rate
            if abs(info.nominal_srate() - self.expected_sampling_rate) > 1:
                logging.error(f"Invalid sampling rate: {info.nominal_srate()}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Stream validation error: {str(e)}")
            return False
            
    def start_receiving(self):
        """Start receiving data from the stream"""
        if not self.connected or not self.inlet:
            return
            
        try:
            while self.connected:
                # Pull chunk of samples
                samples, timestamps = self.inlet.pull_chunk(
                    timeout=0.0,
                    max_samples=self.expected_sampling_rate // 4
                )
                
                if samples:
                    # Convert to numpy array
                    data = np.array(samples).T
                    
                    # Update monitoring stats
                    self.last_data_time = time()
                    self.sample_count += data.shape[1]
                    
                    # Emit data
                    self.data_received.emit(data, timestamps[-1])
                    
        except Exception as e:
            logging.error(f"Data receiving error: {str(e)}")
            self.error_occurred.emit(str(e))
            self._handle_connection_loss()
            
    def check_connection_health(self):
        """Monitor connection health"""
        if self.connected:
            current_time = time()
            if current_time - self.last_data_time > self.timeout_threshold:
                logging.warning("Data timeout detected")
                self._handle_connection_loss()
                
    def _handle_connection_loss(self):
        """Handle connection loss"""
        self.connected = False
        self._update_status(StreamStatus.DISCONNECTED)
        self._start_reconnection()
        
    def _start_reconnection(self):
        """Initialize reconnection process"""
        if not self.reconnect_timer.isActive():
            self.current_reconnect_attempt = 0
            self.attempt_reconnect()
            
    def attempt_reconnect(self):
        """Attempt to reconnect to the stream"""
        if self.connected or self.current_reconnect_attempt >= self.max_reconnect_attempts:
            self.reconnect_timer.stop()
            return
            
        self.current_reconnect_attempt += 1
        self._update_status(StreamStatus.RECONNECTING)
        
        msg = ErrorMessages.RECONNECT_ATTEMPT.format(
            stream_type=self.stream_type.value,
            attempt=self.current_reconnect_attempt,
            max_attempts=self.max_reconnect_attempts
        )
        logging.info(msg)
        
        # Attempt reconnection
        self.connect_to_stream()
        
        # Schedule next attempt if needed
        if not self.connected:
            self.reconnect_timer.start(self.reconnect_interval)
            
    def _update_status(self, status: str):
        """Update and emit new status"""
        self.status = status
        self.status_changed.emit(status)
        
    def disconnect(self):
        """Clean disconnection"""
        self.connected = False
        self.health_check_timer.stop()
        self.reconnect_timer.stop()
        
        if self.inlet:
            self.inlet = None
            
        self._update_status(StreamStatus.DISCONNECTED)