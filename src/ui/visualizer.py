from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer, QByteArray, QSize
from PyQt5.QtGui import QIcon, QPainter, QImage, QPixmap
from PyQt5.QtSvg import QSvgRenderer
from src.ui.svg_icons import IconManager
import pyqtgraph as pg
import numpy as np

class Visualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet("background-color: #2D2D2D;")

        # Initialize class variables
        self.channel_names = ['Left Ear\nTP9', 'Left Forehead\nFP1', 
                            'Right Forehead\nFP2', 'Right Ear\nTP10', 'Aux']
        self.plot_widgets = []
        self.curves = []
        self.current_data = None

        # Channel configuration
        self.y_ranges = {
            'Left Ear\nTP9': (-500, 500),
            'Left Forehead\nFP1': (-500, 500),
            'Right Forehead\nFP2': (-500, 500),
            'Right Ear\nTP10': (-500, 500),
            'Aux': (-500, 500)
        }

        # Create update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(33)  # ~30 FPS

        # Main plot area
        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setSpacing(0)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Create plots with separate regions but shared x-axis
        for i, name in enumerate(self.channel_names):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(5)
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            label = QLabel(name)
            label.setStyleSheet("color: white; font-size: 12px;")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setFixedWidth(100)
            row_layout.addWidget(label)

            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('#2D2D2D')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # Set y range for this channel and disable mouse interaction
            y_min, y_max = self.y_ranges[name]
            plot_widget.setYRange(y_min, y_max)
            plot_widget.setMouseEnabled(x=False, y=False)
            
            # Link x-axis with first plot
            if i > 0 and self.plot_widgets:
                plot_widget.setXLink(self.plot_widgets[0])
            
            # Only show x-axis on bottom plot
            if i < len(self.channel_names) - 1:
                plot_widget.getAxis('bottom').hide()
            
            curve = plot_widget.plot(pen='w')
            self.curves.append(curve)
            
            row_layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)
            plot_layout.addLayout(row_layout)

        self.layout.addWidget(plot_area)

        # Create pause button
        button_layout = QHBoxLayout()
        self.pause_button = QPushButton()
        
        # Create SVG content for pause icon
        svg_content = IconManager.create_svg_icon("pause", "#FFFFFF")
        svg_bytes = QByteArray(svg_content.encode('utf-8'))
        renderer = QSvgRenderer(svg_bytes)
        
        # Create pixmap with the desired size
        pixmap = QIcon()
        size = QSize(20, 20)  # Icon size
        image = QImage(size, QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        pixmap = QPixmap.fromImage(image)
        
        # Set icon
        self.pause_button.setIcon(QIcon(pixmap))
        self.pause_button.setIconSize(size)
        self.pause_button.setFixedSize(30, 30)
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #8E44AD;
                border-radius: 15px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #9B59B6;
            }
        """)
        button_layout.addWidget(self.pause_button, alignment=Qt.AlignLeft)
        button_layout.addStretch()
        self.layout.addLayout(button_layout)

        self.setup_plots()

    def setup_plots(self):
        for i, plot_widget in enumerate(self.plot_widgets):
            plot_widget.setLabel('left', 'µV')
            if i == len(self.plot_widgets) - 1:
                plot_widget.setLabel('bottom', 'Time', 's')
            
            plot_widget.setAntialiasing(True)
            plot_widget.setDownsampling(auto=True, mode='peak')
            plot_widget.setClipToView(True)
            
            # Configure axis appearance
            plot_widget.getAxis('left').setWidth(40)
            if i == len(self.plot_widgets) - 1:
                plot_widget.getAxis('bottom').setHeight(20)
                time_ticks = [0, -1, -2, -3, -4]
                plot_widget.getAxis('bottom').setTicks([[(v, str(abs(v))) for v in time_ticks]])
            
            # Lock the view range with minimal padding
            plot_widget.getViewBox().setXRange(0, -4, padding=0.01)
            plot_widget.getViewBox().setMouseEnabled(x=False, y=False)

    def update_data(self, new_data):
        """Buffer the new data for the next plot update"""
        if new_data is not None:
            self.current_data = new_data

    def update_plots(self):
        """Update all plots with current data"""
        if self.current_data is not None:
            # Create x-data from right to left (0 to -4)
            x_data = np.linspace(0, -4, self.current_data.shape[1])
            for i, curve in enumerate(self.curves):
                if i < len(self.current_data):
                    curve.setData(x=x_data, y=self.current_data[i])

    def zoom_in(self):
        """Set to predefined zoom level"""
        for plot_widget in self.plot_widgets:
            name = self.channel_names[self.plot_widgets.index(plot_widget)]
            y_min, y_max = self.y_ranges[name]
            plot_widget.setYRange(y_min/2, y_max/2)

    def zoom_out(self):
        """Set to predefined zoom level"""
        for plot_widget in self.plot_widgets:
            name = self.channel_names[self.plot_widgets.index(plot_widget)]
            y_min, y_max = self.y_ranges[name]
            plot_widget.setYRange(y_min*2, y_max*2)

    def set_time_window(self, seconds):
        """Set the time window for visualization"""
        for plot_widget in self.plot_widgets:
            plot_widget.setXRange(0, -seconds, padding=0.01)

    def set_color_mode(self, mode):
        """Set the color mode for all plots"""
        colors = ['w', 'r', 'b', 'g', 'm']  # white, red, blue, green, magenta
        for i, curve in enumerate(self.curves):
            if mode == 'monochrome':
                curve.setPen('w')
            elif mode == 'multicolor':
                curve.setPen(colors[i % len(colors)])