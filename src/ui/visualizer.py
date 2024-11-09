from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QSizePolicy, QFrame)
from PyQt5.QtCore import Qt, QTimer, QRectF, QSize, pyqtSignal, QLineF, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QPalette
import pyqtgraph as pg
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

from src.ui.design_system import DesignSystem
from src.constants import ProcessingConfig, DisplayConfig, DataType, StreamConfig, StreamChannelConfig
from src.data.data_processor import ProcessedData
from src.data.utils import SignalQualityMetrics

class PlotContainer(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(0, 0, 0, 0))
        self.setPalette(palette)
        
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {DesignSystem.DARK_THEME.background['secondary']};
                border-radius: {DesignSystem.SPACING.sm}px;
                border: none;
            }}
        """)

        self.grid_overlay = GridOverlay(self)
        self.layout.addWidget(self.grid_overlay)

class ChannelPlot(pg.PlotWidget):
    """Enhanced plot widget for individual channels"""
    
    def __init__(self, channel_name: str, channel_config: StreamChannelConfig, parent=None):
        super().__init__(parent)
        
        # Configuration
        self.channel_name = channel_name
        self.initial_y_range = channel_config.range  # This is now a tuple
        self.current_y_range = channel_config.range
        
        # Setup appearance
        self.setupPlot()
        
    def setupPlot(self):
        """Configure plot styling and behavior"""
        # Basic setup
        self.setBackground('transparent')
        self.setMenuEnabled(False)
        self.hideAxis('left')
        self.hideAxis('bottom')
        
        # Configure view box
        view = self.getViewBox()
        view.setBackgroundColor('transparent')
        view.setYRange(self.initial_y_range[0], self.initial_y_range[1], padding=0)
        view.setMouseEnabled(x=False, y=False)
        
        # Create plot line with antialiasing
        self.curve = self.plot(pen=self.createPen())
        
    def createPen(self) -> pg.mkPen:
        """Create styled pen for plot line"""
        return pg.mkPen({
            'color': DesignSystem.DARK_THEME.channels[self.channel_name.lower()],
            'width': DesignSystem.PLOT_CONFIG['line_width'],
            'cosmetic': True
        })
        
    def setMonochrome(self, enabled: bool):
        """Toggle between monochrome and color mode"""
        if enabled:
            color = DesignSystem.DARK_THEME.foreground['primary']
        else:
            color = DesignSystem.DARK_THEME.channels[self.channel_name.lower()]
            
        self.curve.setPen(pg.mkPen({
            'color': color,
            'width': DesignSystem.PLOT_CONFIG['line_width'],
            'cosmetic': True
        }))

class GridOverlay(QWidget):
    """Transparent overlay widget providing shared grid lines"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def paintEvent(self, event):
        """Draw grid lines and time labels"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Configure grid line pen
        grid_pen = QPen(QColor(DesignSystem.DARK_THEME.grid['major']))
        grid_pen.setWidthF(0.5)
        painter.setPen(grid_pen)
        
        # Draw vertical time grid lines
        width = self.width()
        height = self.height()
        time_interval = width / 4  # 4 major divisions
        
        try:
            for i in range(5):  # 0 to 4 seconds
                x = i * time_interval
                # Use QLineF for floating point coordinates
                line = QLineF(x, 0, x, height)
                painter.drawLine(line)
                
                # Draw time label
                if i < 4:  # Don't draw -0s
                    label = f"-{4-i}.000s"
                    painter.drawText(
                        int(x + 5),  # Convert to int for drawText
                        height - 20,
                        label
                    )
        finally:
            painter.end()

class Visualizer(QWidget):
    """Main visualization widget with enhanced UX and visual design"""
    
    # Signals
    scale_changed = pyqtSignal(float)
    quality_updated = pyqtSignal(object)
    
    def __init__(self, parent=None, data_type: DataType = DataType.EEG):
        super().__init__(parent)
        
        # Configure widget
        self.setObjectName("Visualizer")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(
            DesignSystem.PLOT_CONFIG['channel_height'] * 5 + 
            DesignSystem.PLOT_CONFIG['time_axis_height']
        )
        
        # Initialize state
        self.data_type = data_type
        self.channels = StreamConfig.CHANNELS[data_type]
        self.data_buffer = None
        self.time_data = None
        self.scale_factor = 1.0
        self.monochrome = False
        self.paused = False
        
        # Create UI elements
        self.setupUi()
        self.setupPlots()
        
        # Initialize update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.updatePlots)
        self.update_timer.start(DisplayConfig.MINIMUM_REFRESH_INTERVAL)

    def setupUi(self):
        """Create and configure UI layout"""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Plot container
        self.plot_container = PlotContainer()  # Removed data_type parameter
        self.layout.addWidget(self.plot_container)
        
        # Channel layout
        self.channel_layout = QVBoxLayout()
        self.channel_layout.setSpacing(DesignSystem.SPACING.sm)
        self.plot_container.layout.addLayout(self.channel_layout)
        
        # Create pause button
        self.createPauseButton()
        
    def setupPlots(self):
        """Create and configure plot widgets"""
        self.plots: List[ChannelPlot] = []
        
        for channel_name, channel_config in self.channels.items():
            # Create row with label and plot
            row = QHBoxLayout()
            row.setSpacing(DesignSystem.SPACING.md)
            
            # Channel label
            label = QLabel(channel_name)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setFixedWidth(DesignSystem.PLOT_CONFIG['channel_label_width'])
            label.setStyleSheet(self.getChannelLabelStyle(channel_name))
            row.addWidget(label)
            
            # Channel plot - pass the entire channel_config
            plot = ChannelPlot(channel_name, channel_config)
            plot.setFixedHeight(DesignSystem.PLOT_CONFIG['channel_height'])
            self.plots.append(plot)
            row.addWidget(plot)
            
            self.channel_layout.addLayout(row)
            
    def createPauseButton(self):
        """Create and style pause button with fixed icons"""
        button_layout = QHBoxLayout()
        
        self.pause_button = QPushButton()
        # Use icons from DesignSystem
        self.pause_button.setIcon(DesignSystem.ICONS.pause)
        self.pause_button.setIconSize(QSize(20, 20))
        self.pause_button.setFixedSize(30, 30)
        self.pause_button.setStyleSheet(DesignSystem.get_pause_button_style())
        self.pause_button.clicked.connect(self.togglePause)
        
        button_layout.addWidget(self.pause_button, alignment=Qt.AlignLeft)
        button_layout.addStretch()
        self.layout.addLayout(button_layout)
        
    def updateData(self, processed: ProcessedData):
        """Handle new processed data"""
        if self.paused:
            return

        logging.info(f"Visualizer received data: {processed.data.shape}")
        
        try:
            self.data_buffer = processed.data
            
            # Update time points
            n_samples = processed.data.shape[1]
            self.time_data = np.linspace(
                -DisplayConfig.DEFAULT_TIME_WINDOW, 
                0, 
                n_samples
            )

            # Update quality metrics
            if processed.quality:
                self.quality_updated.emit(processed.quality)
            
            # Trigger plot update
            self.updatePlots()
            
        except Exception as e:
            logging.error(f"Error updating visualizer data: {e}", exc_info=True)
            
    def updatePlots(self):
        """Update all plot visualizations"""
        if self.data_buffer is None or self.time_data is None:
            return
            
        try:
            logging.debug("Updating plots")
            for i, plot in enumerate(self.plots):
                if i < len(self.data_buffer):
                    plot.curve.setData(
                        x=self.time_data, 
                        y=self.data_buffer[i]
                    )
        except Exception as e:
            logging.error(f"Error updating plots: {e}", exc_info=True)
                
    def wheelEvent(self, event):
        """Handle synchronized vertical scaling"""
        if event.angleDelta().y() != 0:
            scale_change = 1.1 if event.angleDelta().y() > 0 else 0.9
            new_scale = self.scale_factor * scale_change
            
            if DisplayConfig.MIN_SCALE <= new_scale <= DisplayConfig.MAX_SCALE:
                self.setScale(new_scale)
                self.scale_changed.emit(new_scale)
                    
        event.accept()
        
    def setColorMode(self, monochrome: bool):
        """Update color mode for all plots"""
        self.monochrome = monochrome
        for plot in self.plots:
            plot.setMonochrome(monochrome)
            
    def setTimeWindow(self, seconds: float):
        """Update visualization time window"""
        if self.time_data is not None:
            self.time_data = np.linspace(-seconds, 0, len(self.time_data))
            
        for plot in self.plots:
            view_box = plot.getViewBox()
            view_box.setXRange(-seconds, 0, padding=0)
            
    def togglePause(self):
        """Toggle data update pause state"""
        self.paused = not self.paused
        self.pause_button.setIcon(
            DesignSystem.ICONS.play if self.paused else DesignSystem.ICONS.pause
        )

    def setScale(self, scale_factor: float):
        """Set the vertical scale factor"""
        if DisplayConfig.MIN_SCALE <= scale_factor <= DisplayConfig.MAX_SCALE:
            self.scale_factor = scale_factor
            
            for plot in self.plots:
                view_box = plot.getViewBox()
                current_range = view_box.viewRange()[1]
                center = sum(current_range) / 2
                half_height = (current_range[1] - current_range[0]) * scale_factor / 2
                view_box.setYRange(
                    center - half_height,
                    center + half_height,
                    padding=0
                )

    @staticmethod
    def getChannelLabelStyle(channel_name: str) -> str:
        """Get stylesheet for channel label"""
        return f"""
            QLabel {{
                color: {DesignSystem.DARK_THEME.channels[channel_name.lower()]};
                font-family: {DesignSystem.TYPOGRAPHY['channel'].family};
                font-size: {DesignSystem.TYPOGRAPHY['channel'].size}px;
                font-weight: {DesignSystem.TYPOGRAPHY['channel'].weight};
            }}
        """