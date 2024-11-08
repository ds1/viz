from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QComboBox, QPushButton, QAction, QStatusBar)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

from ..ui.visualizer import Visualizer
from ..ui.status_bar import StatusBar
from src.ui.timeline import Timeline
from ..ui.design_system import DesignSystem
from ..data.lsl_receiver import LSLReceiver, StreamStatus
from ..data.data_processor import DataProcessor, DataProcessorThread
from ..constants import DataType, DisplayConfig, ProcessingConfig

class ControlBar(QWidget):
    """Control bar with data type, filter, scale, and window controls"""
    
    # Signals use simple types
    data_type_changed = pyqtSignal(str)
    filter_changed = pyqtSignal(str)
    scale_changed = pyqtSignal(float)
    window_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ControlBar")
        
        # Create main layout
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(DesignSystem.SPACING.lg)
        self.layout.setContentsMargins(
            DesignSystem.SPACING.lg,
            DesignSystem.SPACING.md,
            DesignSystem.SPACING.lg,
            DesignSystem.SPACING.md
        )
        
        # Create controls
        self.setupControls()
        
    def setupControls(self):
        controls = [
            ("Data", {
                "items": [dt.value for dt in DataType],
                "signal": self.data_type_changed,
                "transform": lambda x: x  # Pass string directly
            }),
            ("Filter", {
                "items": ProcessingConfig.FILTER_NAMES[DataType.EEG],
                "signal": self.filter_changed,
                "transform": lambda x: x
            }),
            ("Scale", {
                "items": [f"{s}x" for s in DisplayConfig.SCALE_FACTORS],
                "signal": self.scale_changed,
                "transform": lambda x: float(x.rstrip('x'))
            }),
            ("Window", {
                "items": [f"{w}s" for w in DisplayConfig.TIME_WINDOWS],
                "signal": self.window_changed,
                "transform": lambda x: float(x.rstrip('s'))
            })
        ]
        
        for label, config in controls:
            control_layout = QVBoxLayout()
            control_layout.setSpacing(DesignSystem.SPACING.xs)
            
            # Add label
            label_widget = QLabel(label)
            label_widget.setFont(QFont(
                DesignSystem.TYPOGRAPHY['controls'].family,
                DesignSystem.TYPOGRAPHY['controls'].size,
                DesignSystem.TYPOGRAPHY['controls'].weight
            ))
            control_layout.addWidget(label_widget)
            
            # Add combobox
            combo = QComboBox()
            combo.addItems(config["items"])
            combo.currentTextChanged.connect(
                lambda text, t=config["transform"], s=config["signal"]: 
                s.emit(t(text))
            )
            combo.setFixedHeight(30)
            control_layout.addWidget(combo)
            
            setattr(self, f"{label.lower()}_combo", combo)
            self.layout.addLayout(control_layout)
            
        self.layout.addStretch()

class MainWindow(QMainWindow):
    """Main application window with integrated design system"""
    
    def __init__(self, data_processor=None, lsl_receiver=None):
        super().__init__()
        self.setWindowTitle("Petal Viz 1.0")
        self.resize(1200, 800)
        
        # Store dependencies
        self.data_processor = data_processor
        self.lsl_receiver = lsl_receiver
        self.current_data_type = DataType.EEG
        self.monochrome_mode = False
        
        # Set up UI
        self.setupWindow()
        self.setupMenuBar()
        self.setupCentralWidget()
        
        # Initialize state
        self.current_data_type = DataType.EEG
        self.monochrome_mode = False
        
    def setupWindow(self):
        """Configure window appearance"""
        # Set window style
        self.setStyleSheet(DesignSystem.get_style_sheet())
        
        # Set minimum size
        self.setMinimumSize(800, 600)
        
        # Center window
        self.centerWindow()
        
    def setupMenuBar(self):
        """Create and configure menu bar"""
        menubar = self.menuBar()
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction(self.createAction(
            "Zoom In", 
            lambda: self.visualizer.scale_changed.emit(2.0)
        ))
        view_menu.addAction(self.createAction(
            "Zoom Out", 
            lambda: self.visualizer.scale_changed.emit(0.5)
        ))
        view_menu.addSeparator()
        view_menu.addAction(self.createAction(
            "Monochrome",
            lambda: self.setColorMode(True)
        ))
        view_menu.addAction(self.createAction(
            "Multicolor",
            lambda: self.setColorMode(False)
        ))
        
        # Window menu  
        window_menu = menubar.addMenu("Window")
        window_menu.addAction(self.createAction(
            "Enter Fullscreen",
            self.showFullScreen
        ))
        window_menu.addAction(self.createAction(
            "Exit Fullscreen",
            self.showNormal
        ))
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction(self.createAction(
            "User Guide",
            lambda: self.openUrl("https://docs.petal.tech")
        ))
        help_menu.addAction(self.createAction(
            "Support",
            lambda: self.openUrl("https://docs.petal.tech")
        ))
        
    def setupCentralWidget(self):
        """Create main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add control bar
        self.control_bar = ControlBar()
        main_layout.addWidget(self.control_bar)
        
        # Add visualizer
        self.visualizer = Visualizer()
        main_layout.addWidget(self.visualizer)
        
        # Add timeline
        self.timeline = Timeline()
        main_layout.addWidget(self.timeline)
        
        # Add status bar
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)
        
        # Connect control signals
        self.connectControls()
        
    def setupDataProcessing(self):
        """Initialize data processing pipeline - only if dependencies not provided"""
        if self.lsl_receiver is None:
            self.lsl_receiver = LSLReceiver()
            self.lsl_receiver.connection_changed.connect(self.updateConnectionStatus)
            self.lsl_receiver.error_occurred.connect(self.status_bar.showError)
        
        if self.data_processor is None:
            self.data_processor = DataProcessor(self.current_data_type)
            self.processor_thread = DataProcessorThread(self.data_processor)
            self.processor_thread.processed_data.connect(self.visualizer.updateData)
            self.processor_thread.processed_data.connect(self.updateQualityMetrics)
            
            self.lsl_receiver.connect_to_stream()
            self.processor_thread.start()
        
    def connectControls(self):
        """Connect control signals"""
        # Data processor connections
        if self.data_processor:
            self.control_bar.filter_changed.connect(self.data_processor.set_filter)
            
        # LSL receiver connections
        if self.lsl_receiver:
            self.lsl_receiver.status_changed.connect(
                self.status_bar.updateStreamStatus
            )
            self.lsl_receiver.error_occurred.connect(
                self.status_bar.showError
            )
        
        # UI connections that don't depend on external components
        self.control_bar.scale_changed.connect(
            self.visualizer.setScale  # Updated to match the method name
        )
        self.control_bar.window_changed.connect(
            self.visualizer.setTimeWindow  # Make sure this method exists too
        )
        
        # Visualizer signals
        self.visualizer.scale_changed.connect(self.updateScaleDisplay)
        self.visualizer.quality_updated.connect(self.updateQualityMetrics)
        
    def changeDataType(self, data_type: DataType):
        """Handle data type change"""
        self.current_data_type = data_type
        self.data_processor.set_data_type(data_type)
        self.visualizer.setDataType(data_type)
        self.status_bar.updateDataType(data_type)
        
    def setColorMode(self, monochrome: bool):
        """Toggle between monochrome and color modes"""
        self.monochrome_mode = monochrome
        self.visualizer.setColorMode(monochrome)
        
    def updateConnectionStatus(self, status: StreamStatus):
        """Update status bar with connection state"""
        self.status_bar.updateStreamStatus(status)
        
    def updateQualityMetrics(self, metrics: dict):
        """Update status bar with signal quality metrics"""
        self.status_bar.updateQualityMetrics(metrics)
        
    def updateScaleDisplay(self, scale: float):
        """Update scale display in control bar"""
        self.control_bar.scale_combo.setCurrentText(f"{scale:.1f}x")
        
    def createAction(self, text: str, slot) -> QAction:
        """Helper to create menu actions"""
        action = QAction(text, self)
        action.triggered.connect(slot)
        return action
        
    def centerWindow(self):
        """Center window on screen"""
        frame = self.frameGeometry()
        screen = self.screen().availableGeometry().center()
        frame.moveCenter(screen)
        self.move(frame.topLeft())
        
    def closeEvent(self, event):
        """Clean up resources on close"""
        self.processor_thread.stop()
        self.processor_thread.wait()
        self.lsl_receiver.disconnect()
        event.accept()