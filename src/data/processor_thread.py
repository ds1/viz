from PyQt5.QtCore import QThread, pyqtSignal
from collections import deque
import numpy as np
import logging
import time
from threading import Lock
from typing import Optional, Tuple

from .data_processor import DataProcessor

class DataProcessorThread(QThread):
    """Thread for running data processing"""
    
    processed_data = pyqtSignal(object)  # For processed data
    error_occurred = pyqtSignal(str)     # For error messages
    
    def __init__(self, processor: DataProcessor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.running = False
        self.data_queue = deque(maxlen=100)  # Buffer for incoming data
        self.queue_lock = Lock()
        
    def add_data(self, samples: np.ndarray, timestamps: float):
        """Thread-safe method to add data to the processing queue"""
        with self.queue_lock:
            self.data_queue.append((samples, timestamps))
    
    def run(self):
        """Thread's main loop"""
        logging.info("Processing thread started")
        self.running = True
        last_process_time = time.time()

        while self.running:
            # Process any queued data
            data_to_process = None
            with self.queue_lock:
                if self.data_queue:
                    data_to_process = self.data_queue.popleft()
            
            if data_to_process is not None:
                        try:
                            samples, timestamps = data_to_process
                            current_time = time.time()
                            process_interval = current_time - last_process_time
                            logging.debug(f"Processing interval: {process_interval:.3f}s")
                            last_process_time = current_time
                            
                            result = self.processor.process_data(samples, timestamps)
                            if result is not None:
                                self.processed_data.emit(result)
                                
                        except Exception as e:
                            logging.error(f"Processing error: {str(e)}")
                            self.error_occurred.emit(str(e))
                            
    def stop(self):
        """Stop the processing thread"""
        self.running = False
        self.wait()