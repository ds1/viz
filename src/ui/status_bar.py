from PyQt5.QtWidgets import QStatusBar, QLabel, QHBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QUrl

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

        self.bluetooth_icon = self.create_icon_widget('M14.88,16.29L13,18.17V14.41M13,5.83L14.88,7.71L13,9.58M17.71,7.71L12,2H11V9.58L6.41,5L5,6.41L10.59,12L5,17.58L6.41,19L11,14.41V22H12L17.71,16.29L13.41,12L17.71,7.71Z')
        right_layout.addWidget(self.bluetooth_icon)

        self.battery_icon = self.create_icon_widget('M16,20H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z')
        right_layout.addWidget(self.battery_icon)

        self.battery_percentage = QLabel("100%")
        right_layout.addWidget(self.battery_percentage)

        self.addPermanentWidget(right_widget)

    def create_icon_widget(self, icon_path, color='black', size=24):
        view = QWebEngineView()
        view.setFixedSize(size, size)
        html = f"""
        <html>
        <body style="margin: 0; display: flex; justify-content: center; align-items: center;">
            <svg viewBox="0 0 24 24" width="{size}" height="{size}">
                <path fill="{color}" d="{icon_path}" />
            </svg>
        </body>
        </html>
        """
        view.setHtml(html)
        return view

    def update_console_log(self, message):
        self.console_log.setText(f"Console Log: {message[:256]}")

    def update_stream_status(self, status):
        self.stream_status.setText(f"Stream Status: {status}")

    def update_output_settings(self, settings):
        self.output_settings.setText(f"Output: {settings}")

    def update_device_status(self, device):
        self.device_status.setText(f"Device: {device}")

    def update_battery(self, percentage):
        battery_path = {
            0: 'M16,20H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z',
            10: 'M16,18H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z',
            20: 'M16,17H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z',
            30: 'M16,15H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z',
            40: 'M16,14H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z',
            50: 'M16,13H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z',
            60: 'M16,12H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z',
            70: 'M16,10H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z',
            80: 'M16,9H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z',
            90: 'M16,8H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z',
            100: 'M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z'
        }
        level = min(100, max(0, (percentage // 10) * 10))
        self.battery_icon.setHtml(self.create_icon_widget(battery_path[level]).page().toHtml())
        self.battery_percentage.setText(f"{percentage}%")

    def update_bluetooth_status(self, connected):
        bluetooth_path = 'M14.88,16.29L13,18.17V14.41M13,5.83L14.88,7.71L13,9.58M17.71,7.71L12,2H11V9.58L6.41,5L5,6.41L10.59,12L5,17.58L6.41,19L11,14.41V22H12L17.71,16.29L13.41,12L17.71,7.71Z' if connected else 'M13,5.83L14.88,7.71L13,9.58L14.88,11.46L13,13.33L14.88,15.21L13,17.08V22H12L7.41,17.41L8.83,16L12,19.17V15.89L9.41,13.31L8,14.72L7.29,14L12,9.29V5.59L9.41,3L8,4.41L7.29,3.71L12,1V5.83M16.59,6L15.17,7.41L17.17,9.41L18.58,8M17.17,14.59L15.17,16.59L16.59,18L18.58,16.09L17.17,14.59Z'
        self.bluetooth_icon.setHtml(self.create_icon_widget(bluetooth_path).page().toHtml())