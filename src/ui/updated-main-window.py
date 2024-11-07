import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QComboBox, QAction)
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QIcon

from src.ui.visualizer import Visualizer
from src.ui.status_bar import StatusBar
from src.ui.design_system import DesignSystem
from src.ui.stylesheet_manager import StylesheetManager
from src.data.lsl_receiver import LSLReceiver
from src.data.data_processor import DataProcessor, DataProcessorThread

class ControlBar(QWidget):
    """Control bar with data, filter, scale, and window controls"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ControlBar")
        
        # Create layout
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(DesignSystem.SPACING.lg)
        self.layout.setContentsMargins(
            DesignSystem.SPACING.md,
            DesignSystem.SPACING.sm,
            DesignSystem.SPACING.md,
            DesignSystem.SPACING.sm
        )
        
        # Add controls
        self.setup_controls()
        
    def setup_controls(self):
        """Create and configure control elements"""
        controls = [
            ("Data", ["EEG", "PPG", "ACC", "GYRO"]),
            ("Filter", ["Off", "Default", "Low", "High"]),
            ("Scale", ["200 µV", "100 µV", "50 µV", "500 µV"]),
            ("Window", ["4s", "2s", "8s"])
        ]
        
        for label, items in controls:
            control_layout = QVBoxLayout()
            control_layout.setSpacing(DesignSystem.SPACING.xs)
            
            # Add label
            label_widget = QLabel(label)
            control_layout.addWidget(label_widget)
            
            # Add combo box
            combo = QComboBox()
            combo.addItems(items)
            combo.setObjectName(f"{label.lower()}_combo")
            control_layout.addWidget(combo)
            
            self.layout.addLayout(control_layout)
        
        self.layout.addStretch()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.setWindowTitle("Petal Viz 1.0")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set up styling
        StylesheetManager.setup_fonts()
        self.setup_theme()
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Set up UI components
        self.setup_menu()
        self.setup_ui()
        self.setup_data_processing()
        
    def setup_theme(self):
        """Apply application theme"""
        self.setStyleSheet(DesignSystem.get_style_sheet())
        self.menuBar().setStyleSheet(StylesheetManager.get_menu_bar_style())
        
    def setup_menu(self):
        """Configure menu bar"""
        # View menu
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.create_action("Monochrome", self.set_monochrome))
        view_menu.addAction(self.create_action("Multicolor", self.set_multicolor))
        
        # Window menu
        window_menu = self.menuBar().addMenu("Window")
        window_menu.addAction(self.create_action("Enter Fullscreen", 
                                               self.showFullScreen))
        window_menu.addAction(self.create_action("Exit Fullscreen", 
                                               self.showNormal))
        
        # Help menu
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction(self.create_action("User Guide", self.open_user_guide))
        help_menu.addAction(self.create_action("Support", self.open_support))
        
    def setup_ui(self):
        """Set up main UI components"""
        # Add control bar
        self.control_bar = ControlBar()
        self.main_layout.addWidget(self.control_bar)
        
        # Add visualizer
        self.visualizer = Visualizer()
        self.main_layout.addWidget(self.visualizer)
        
        # Add status bar
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)
        
    def setup_data_processing(self):
        """Configure data processing pipeline"""
        # Set up LSL receiver
        self.lsl_receiver = LSLReceiver()
        
        # Set up data processor
        self.data_processor = DataProcessor(
            lsl_receiver=self.lsl_receiver,
            buffer_size=DesignSystem.PLOT_CONFIG['buffer_size']
        )
        
        # Set up processing thread
        self.processor_thread = DataProcessorThread(self.data_processor)
        self.processor_thread.processed_data.connect(self.visualizer.update_data)
        
        # Start processing
        self.processor_thread.start()
        
    def create_action(self, text, slot):
        """Create menu action"""
        action = QAction(text, self)
        action.triggered.connect(slot)
        return action
    
    def set_monochrome(self):
        self.visualizer.set_color_mode('monochrome')
        
    def set_multicolor(self):
        self.visualizer.set_color_mode('multicolor')
        
    def open_user_guide(self):
        # TODO: Implement user guide
        pass
        
    def open_support(self):
        # TODO: Implement support
        pass
        
    def closeEvent(self, event):
        """Clean up on window close"""
        self.processor_thread.stop()
        self.processor_thread.wait()
        event.accept()
