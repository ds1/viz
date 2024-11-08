from PyQt5.QtWidgets import (QStatusBar, QLabel, QHBoxLayout, QWidget)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtSvg import QSvgWidget
from typing import Dict, Optional

from src.ui.design_system import DesignSystem
from src.constants import StreamStatus, DataType
from src.ui.svg_icons import IconManager

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

        self.bluetooth_icon = IconManager.get_icon("bluetooth")
        IconManager.update_icon("bluetooth", "bluetooth_disconnected", "#FFFFFF")
        right_layout.addWidget(self.bluetooth_icon)

        self.battery_icon = IconManager.get_icon("battery")
        IconManager.update_icon("battery", "battery_75", "#FFFFFF")
        right_layout.addWidget(self.battery_icon)

        self.battery_percentage = QLabel("75%")
        right_layout.addWidget(self.battery_percentage)

        self.addPermanentWidget(right_widget)

    def update_console_log(self, message):
        self.console_log.setText(f"Console Log: {message[:256]}")

    def update_stream_status(self, status):
        self.stream_status.setText(f"Stream Status: {status}")

    def update_output_settings(self, settings):
        self.output_settings.setText(f"Output: {settings}")

    def update_device_status(self, device):
        self.device_status.setText(f"Device: {device}")

    def update_battery(self, percentage):
        level = min([key for key in [0, 25, 50, 75, 100] if key >= percentage])
        IconManager.update_icon("battery", f"battery_{level}", "#FFFFFF")
        self.battery_percentage.setText(f"{percentage}%")

    def update_bluetooth_status(self, connected):
        status = "connected" if connected else "disconnected"
        IconManager.update_icon("bluetooth", f"bluetooth_{status}", "#FFFFFF")