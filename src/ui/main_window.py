import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox, QPushButton, QAction
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QThread, QByteArray
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
        self.setup_data_processing()

    @staticmethod
    def create_svg_icon(icon_name, color):
        svg_content = {
            "menu_down": f'<path fill="{color}" d="M7,10L12,15L17,10H7Z" />',
            # Add other icons here as needed, for example:
            # "battery": f'<path fill="{color}" d="M16,20H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z" />',
            # "bluetooth": f'<path fill="{color}" d="M14.88,16.29L13,18.17V14.41M13,5.83L14.88,7.71L13,9.58M17.71,7.71L12,2H11V9.58L6.41,5L5,6.41L10.59,12L5,17.58L6.41,19L11,14.41V22H12L17.71,16.29L13.41,12L17.71,7.71Z" />',
        }
        return f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            {svg_content.get(icon_name, '')}
        </svg>
        """

    def setup_ui(self):
        controls_layout = QHBoxLayout()
        
        for control, default in [("Data", "EEG"), ("Filter", "Mid"), ("Vertical Scale", "200 uHZ"), ("Window", "4 SEC")]:
            control_layout = QVBoxLayout()
            
            label = QLabel(control)
            label.setStyleSheet("color: white; margin-bottom: 5px;")
            control_layout.addWidget(label)

            dropdown = QComboBox()
            dropdown.addItem(default)
            dropdown.setStyleSheet("""
                QComboBox {
                    background-color: #121212;
                    color: white;
                    border: 1px solid #C58BFF;
                    padding: 5px;
                    padding-right: 20px;
                    border-radius: 5px;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 20px;
                    border-left-width: 0px;
                }
            """)
            
            # Create SVG widget for dropdown arrow
            svg_content = self.create_svg_icon("menu_down", "#C58BFF")
            svg_widget = QSvgWidget()
            svg_widget.load(QByteArray(svg_content.encode('utf-8')))
            svg_widget.setFixedSize(20, 20)
            
            dropdown_layout = QHBoxLayout()
            dropdown_layout.addWidget(dropdown)
            dropdown_layout.addWidget(svg_widget)
            dropdown_layout.setSpacing(0)
            dropdown_layout.setContentsMargins(0, 0, 0, 0)
            
            control_layout.addLayout(dropdown_layout)
            controls_layout.addLayout(control_layout)
        
        self.main_layout.addLayout(controls_layout)

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

    def setup_data_processing(self):
        # Set up LSL receiver and data processor
        self.lsl_receiver = LSLReceiver()
        self.data_processor = DataProcessor()

        # Set up receiver thread
        self.receiver_thread = QThread()
        self.lsl_receiver.moveToThread(self.receiver_thread)
        self.receiver_thread.started.connect(self.lsl_receiver.connect_to_stream)

        # Connect signals
        self.lsl_receiver.data_received.connect(self.data_processor.process_data)
        self.data_processor.processed_data.connect(self.visualizer.update_data)
        self.lsl_receiver.connection_changed.connect(self.update_connection_status)

        # Start receiver thread
        self.receiver_thread.start()

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