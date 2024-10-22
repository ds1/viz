from pylsl import StreamInlet, resolve_stream
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import numpy as np

class LSLReceiver(QObject):
    data_received = pyqtSignal(np.ndarray)
    connection_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.inlet = None
        self.is_connected = False
        self.thread = None
        self.latest_sample = None

    def connect_to_stream(self):
        print("Looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        if len(streams) > 0:
            self.inlet = StreamInlet(streams[0])
            self.is_connected = True
            self.connection_changed.emit(True)
            print("Connected to EEG stream")
            self.start_receiving()
        else:
            print("No EEG stream found")
            self.connection_changed.emit(False)

    def start_receiving(self):
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._receive)
        self.thread.start()

    def _receive(self):
        while self.is_connected:
            sample, timestamp = self.inlet.pull_sample(timeout=1.0)
            if sample is not None:
                self.latest_sample = np.array(sample)
                self.data_received.emit(self.latest_sample)
            else:
                print("No sample received")  # Debug print

    def get_data(self):
        return self.latest_sample

    def disconnect(self):
        self.is_connected = False
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        if self.inlet:
            self.inlet = None
        self.connection_changed.emit(False)
        print("Disconnected from EEG stream")