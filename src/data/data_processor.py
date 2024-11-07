import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QMutex, QWaitCondition

class DataProcessor(QObject):
    processed_data = pyqtSignal(np.ndarray)

    def __init__(self, lsl_receiver, num_channels=5, buffer_size=1000, downsample_factor=1):
        super().__init__()
        self.lsl_receiver = lsl_receiver
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.downsample_factor = downsample_factor
        self.data_buffer = np.zeros((num_channels, buffer_size))
        self.mutex = QMutex()
        self.data_available = QWaitCondition()

    def process_data(self, new_data):
        if new_data is None:
            return None

        self.mutex.lock()
        # Roll the buffer and add new data
        self.data_buffer = np.roll(self.data_buffer, -1, axis=1)
        self.data_buffer[:, -1] = new_data[:self.num_channels]

        # Downsample if necessary
        if self.downsample_factor > 1:
            downsampled_data = self.data_buffer[:, ::self.downsample_factor]
        else:
            downsampled_data = self.data_buffer.copy()
        self.mutex.unlock()

        return downsampled_data

    def get_new_data(self):
        return self.lsl_receiver.get_data()

class DataProcessorThread(QThread):
    processed_data = pyqtSignal(np.ndarray)

    def __init__(self, data_processor):
        super().__init__()
        self.data_processor = data_processor
        self.running = True

    def run(self):
        while self.running:
            new_data = self.data_processor.get_new_data()
            if new_data is not None:
                processed_data = self.data_processor.process_data(new_data)
                if processed_data is not None:
                    self.processed_data.emit(processed_data)
            self.msleep(10)  # Adjust sleep time as needed

    def stop(self):
        self.running = False