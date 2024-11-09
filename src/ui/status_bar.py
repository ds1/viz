from PyQt5.QtWidgets import QStatusBar, QLabel, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt, pyqtSignal

from src.ui.design_system import DesignSystem
from src.constants import StreamStatus, DataType

class StatusBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.console_log = QLabel("Console Log: Ready")
        self.addWidget(self.console_log, 1)

        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.stream_status = QLabel("Stream Status: Disconnected")
        right_layout.addWidget(self.stream_status)

        self.output_settings = QLabel("Output: N/A")
        right_layout.addWidget(self.output_settings)

        self.device_status = QLabel("Device: N/A")
        right_layout.addWidget(self.device_status)

        self.addPermanentWidget(right_widget)

    def showError(self, message: str):
        """Display error message in status bar"""
        self.console_log.setText(f"Error: {message}")

    def updateStreamStatus(self, status: StreamStatus):
        """Update stream status display"""
        self.stream_status.setText(f"Stream Status: {status}")

    def updateDataType(self, data_type: DataType):
        """Update data type display"""
        self.output_settings.setText(f"Output: {data_type.value}")