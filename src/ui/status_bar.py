from PyQt5.QtWidgets import QStatusBar, QLabel, QHBoxLayout, QWidget
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import QByteArray, Qt

class SvgIcon(QSvgWidget):
    def __init__(self, size=24, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.svg_contents = {}

    def add_svg(self, name, content):
        self.svg_contents[name] = QByteArray(content.encode('utf-8'))

    def update_icon(self, name):
        if name in self.svg_contents:
            self.load(self.svg_contents[name])

class IconManager:
    @staticmethod
    def create_battery_svg(level):
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path fill="white" d="M16,{20-level/6.25:.0f}H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z" />
        </svg>"""

    @staticmethod
    def create_bluetooth_svg(connected):
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path fill="white" d="{'M14.88,16.29L13,18.17V14.41M13,5.83L14.88,7.71L13,9.58M17.71,7.71L12,2H11V9.58L6.41,5L5,6.41L10.59,12L5,17.58L6.41,19L11,14.41V22H12L17.71,16.29L13.41,12L17.71,7.71Z' if connected else 'M13,5.83L14.88,7.71L13,9.58L14.88,11.46L13,13.33L14.88,15.21L13,17.08V22H12L7.41,17.41L8.83,16L12,19.17V15.89L9.41,13.31L8,14.72L7.29,14L12,9.29V5.59L9.41,3L8,4.41L7.29,3.71L12,1V5.83M16.59,6L15.17,7.41L17.17,9.41L18.58,8M17.17,14.59L15.17,16.59L16.59,18L18.58,16.09L17.17,14.59Z'}" />
        </svg>"""

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

        self.bluetooth_icon = SvgIcon()
        self.bluetooth_icon.add_svg("connected", IconManager.create_bluetooth_svg(True))
        self.bluetooth_icon.add_svg("disconnected", IconManager.create_bluetooth_svg(False))
        self.bluetooth_icon.update_icon("disconnected")
        right_layout.addWidget(self.bluetooth_icon)

        self.battery_icon = SvgIcon()
        for level in [0, 25, 50, 75, 100]:
            self.battery_icon.add_svg(f"battery_{level}", IconManager.create_battery_svg(level))
        self.battery_icon.update_icon("battery_75")
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
        self.battery_icon.update_icon(f"battery_{level}")
        self.battery_percentage.setText(f"{percentage}%")

    def update_bluetooth_status(self, connected):
        self.bluetooth_icon.update_icon("connected" if connected else "disconnected")