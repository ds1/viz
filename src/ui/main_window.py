import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox, QPushButton, QAction
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QThread
from src.ui.visualizer import Visualizer
from src.ui.timeline import Timeline
from src.ui.status_bar import StatusBar
from src.data.lsl_receiver import LSLReceiver
from src.data.data_processor import DataProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Petal Viz 1.0")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #1E1E1E; color: white;")

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        icon_path = os.path.join(base_dir, 'src', 'resources', 'images', 'viz_logo.png')
        self.setWindowIcon(QIcon(icon_path))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.setup_menu()
        self.setup_ui()

    def setup_ui(self):

        # Control dropdowns
        controls = QHBoxLayout()
        for control, default in [("Data", "EEG"), ("Filter", "Mid"), ("Vertical Scale", "200 uHZ"), ("Window", "4 SEC")]:
            dropdown = QComboBox()
            dropdown.addItem(default)
            dropdown.setStyleSheet("""
                QComboBox {
                    background-color: #1E1E1E;
                    color: white;
                    padding: 5px;
                    border: #8E44AD;
                    border-radius: 5px;
                }
            """)
            controls.addWidget(dropdown)
        
        self.main_layout.addLayout(controls)

        # Visualizer
        self.visualizer = Visualizer()
        self.main_layout.addWidget(self.visualizer)

        # Timeline
        self.timeline = Timeline()
        self.main_layout.addWidget(self.timeline)

        # Pause button
        pause_button = QPushButton("II")
        pause_button.setStyleSheet("""
            QPushButton {
                background-color: #8E44AD;
                color: white;
                border-radius: 15px;
                padding: 5px;
                font-weight: bold;
            }
        """)
        pause_button.setFixedSize(30, 30)
        self.main_layout.addWidget(pause_button, alignment=Qt.AlignLeft)

        # Status bar
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)

        # Set up LSL receiver and data processor
        self.lsl_receiver = LSLReceiver()
        self.data_processor = DataProcessor()

        # Connect signals
        self.lsl_receiver.data_received.connect(self.data_processor.process_data)
        self.data_processor.processed_data.connect(self.visualizer.update_data)
        self.lsl_receiver.connection_changed.connect(self.update_connection_status)

        # Start LSL receiver
        self.lsl_receiver.connect_to_stream()

    def setup_menu(self):

        # View menu
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.create_action("Zoom In", self.zoom_in))
        view_menu.addAction(self.create_action("Zoom Out", self.zoom_out))
        view_menu.addAction(self.create_action("Monochrome", self.set_monochrome))
        view_menu.addAction(self.create_action("Multicolor", self.set_multicolor))

        # Window menu
        window_menu = self.menuBar().addMenu("Window")
        window_menu.addAction(self.create_action("Enter Fullscreen", self.enter_fullscreen))
        window_menu.addAction(self.create_action("Exit Fullscreen", self.exit_fullscreen))

        # Help menu
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction(self.create_action("User Guide", self.open_user_guide))
        help_menu.addAction(self.create_action("Support", self.open_support))

    def create_action(self, text, slot):
        action = QAction(text, self)
        action.triggered.connect(slot)
        return action

    def update_connection_status(self, is_connected):
        status = "Connected" if is_connected else "Disconnected"
        self.status_bar.update_stream_status(status)

    def closeEvent(self, event):
        self.lsl_receiver.disconnect()
        self.receiver_thread.quit()
        self.receiver_thread.wait()
        super().closeEvent(event)

    def zoom_in(self):
        self.visualizer.zoom_in()

    def zoom_out(self):
        self.visualizer.zoom_out()

    def set_monochrome(self):
        self.visualizer.set_color_mode('monochrome')

    def set_multicolor(self):
        self.visualizer.set_color_mode('multicolor')

    def enter_fullscreen(self):
        self.showFullScreen()

    def exit_fullscreen(self):
        self.showNormal()

    def open_user_guide(self):
        # Implement opening the user guide URL
        pass

    def open_support(self):
        # Implement opening the support URL
        pass

    def update_battery_status(self, percentage):
        self.status_bar.update_battery(percentage)