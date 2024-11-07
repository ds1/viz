from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QSizePolicy, QFrame)
from PyQt5.QtCore import Qt, QTimer, QRectF, QSize
from PyQt5.QtGui import QPainter, QPen, QColor
import pyqtgraph as pg
import numpy as np
from typing import List, Dict, Optional

from src.ui.design_system import DesignSystem, ColorMode

class PlotContainer(QFrame):
    """Custom container for plots with consistent styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(DesignSystem.SPACING.md)
        self.layout.setContentsMargins(
            DesignSystem.SPACING.md,
            DesignSystem.SPACING.md,
            DesignSystem.SPACING.md,
            DesignSystem.SPACING.md
        )
        
        # Apply container styling
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            PlotContainer {{
                background-color: {DesignSystem.DARK_THEME.background['secondary']};
                border-radius: {DesignSystem.SPACING.sm}px;
            }}
        """)

class ChannelPlot(pg.PlotWidget):
    """Enhanced plot widget with consistent styling and behavior"""
    def __init__(self, channel_name: str, y_range: tuple, parent=None):
        super().__init__(parent)
        
        # Store configuration
        self.channel_name = channel_name
        self.y_range = y_range
        
        # Apply styling
        self.setup_appearance()
        self.setup_grid()
        self.setup_axes()
        
    def setup_appearance(self):
        """Configure plot appearance"""
        plot_style = DesignSystem.get_plot_style('plot_background')
        self.setBackground(plot_style['background'])
        self.setAntialiasing(True)
        self.setMenuEnabled(False)
        
    def setup_grid(self):
        """Configure grid appearance"""
        grid_style = DesignSystem.get_plot_style('grid')
        self.showGrid(x=True, y=True)
        
        # Apply styling to both axes
        for axis in [self.getAxis('left'), self.getAxis('bottom')]:
            axis.setGrid(DesignSystem.PLOT_CONFIG['grid_opacity'])
            axis.setPen(QPen(QColor(grid_style['major']['color'])))
            
    def setup_axes(self):
        """Configure axes appearance"""
        axis_style = DesignSystem.get_plot_style('axis')
        
        # Y-axis configuration
        y_axis = self.getAxis('left')
        y_axis.setWidth(DesignSystem.PLOT_CONFIG['y_axis_width'])
        y_axis.setLabel('µV')
        y_axis.setRange(*self.y_range)
        
        # X-axis configuration (hidden by default, shown only on bottom plot)
        x_axis = self.getAxis('bottom')
        x_axis.hide()
