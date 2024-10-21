import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

class DataProcessor(QObject):
    processed_data = pyqtSignal(np.ndarray)

    def __init__(self, num_channels=5, buffer_size=1000):
        super().__init__()
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.data_buffer = np.zeros((num_channels, buffer_size))

    def process_data(self, new_data):
        # Roll the buffer and add new data
        self.data_buffer = np.roll(self.data_buffer, -1, axis=1)
        self.data_buffer[:, -1] = new_data

        # Here you can add more processing steps as needed
        # For example, filtering, artifact rejection, etc.

        print(f"Processed data shape: {self.data_buffer.shape}")  # Debug print
        self.processed_data.emit(self.data_buffer)