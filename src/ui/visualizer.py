from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy
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

        # Configure plot container to expand
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Initialize scaling variables
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0

        # Initialize cached data
        self.x_data = None
        self.time_ticks = [-4.000, -3.000, -2.000, -1.000, 0.000]
        self.time_tick_labels = [(v, f"{v:.3f}") for v in self.time_ticks]

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

        # Setup UI Components
        self.setup_plot_area()
        self.setup_pause_button()
        self.setup_plots()
        self.configure_grids()

    def setup_plot_area(self):
        """Initialize the plot area and create all plot widgets"""
        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setSpacing(0)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setStretchFactor(plot_area, 1)  # Make plot area expand to fill space

        # Create plots with separate regions but shared x-axis
        for i, name in enumerate(self.channel_names):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(5)
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            # Channel label
            label = QLabel(name)
            label.setStyleSheet("color: white; font-size: 12px;")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setFixedWidth(100)
            row_layout.addWidget(label)

            # Plot widget
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('#2D2D2D')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # Set y range and disable mouse interaction
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

    def setup_pause_button(self):
        """Initialize and configure the pause button"""
        button_layout = QHBoxLayout()
        self.pause_button = QPushButton()
        
        # Create SVG content for pause icon
        svg_content = IconManager.create_svg_icon("pause", "#FFFFFF")
        svg_bytes = QByteArray(svg_content.encode('utf-8'))
        renderer = QSvgRenderer(svg_bytes)
        
        # Create pixmap with the desired size
        size = QSize(20, 20)
        image = QImage(size, QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        
        # Set button properties
        self.pause_button.setIcon(QIcon(QPixmap.fromImage(image)))
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

    def sync_x_ranges(self):
        """Synchronize x-axis ranges across all plots"""
        x_range = self.master_viewbox.viewRange()[0]
        for plot_widget in self.plot_widgets[1:]:
            plot_widget.setXRange(x_range[0], x_range[1], padding=0)

    def sync_y_ranges(self):
        """Synchronize y-axis ranges across all plots"""
        y_range = self.master_viewbox.viewRange()[1]
        for plot_widget in self.plot_widgets[1:]:
            plot_widget.setYRange(y_range[0], y_range[1], padding=0)

    def setup_plots(self):
        """Configure plot settings and synchronization"""
        # Set up master viewbox for synchronization
        self.master_viewbox = self.plot_widgets[0].getViewBox()
        self.master_viewbox.setMouseEnabled(x=False, y=True)  # Enable only vertical mouse interaction
        
        for i, plot_widget in enumerate(self.plot_widgets):
            # Basic setup
            plot_widget.setLabel('left', 'µV')
            plot_widget.setAntialiasing(True)
            plot_widget.setDownsampling(auto=True, mode='peak')
            plot_widget.setClipToView(True)
            
            # Configure grid
            plot_widget.showGrid(x=True, y=True)
            
            # Make grid lines span full height for all plots
            axis = plot_widget.getAxis('bottom')
            axis.setHeight(0)  # Remove the axis height restriction
            
            # Configure grid appearance
            grid_pen = pg.mkPen(color=(128, 128, 128), width=1, style=Qt.SolidLine)
            plot_widget.getAxis('bottom').setPen(grid_pen)
            plot_widget.getAxis('left').setPen(grid_pen)
            
            # Get viewbox for this plot
            view_box = plot_widget.getViewBox()
            
            # Set y range
            y_min, y_max = self.y_ranges[self.channel_names[i]]
            view_box.setYRange(y_min, y_max, padding=0)
            
            # Link view to master
            if i > 0:
                view_box.setXLink(self.plot_widgets[0])
                view_box.setYLink(self.plot_widgets[0])
            
            # Disable individual plot mouse control
            view_box.setMouseEnabled(x=False, y=False)
            
            # Only show time labels on bottom plot
            if i == len(self.plot_widgets) - 1:
                plot_widget.getAxis('bottom').setLabel('Time', 's')
                plot_widget.getAxis('bottom').setStyle(showValues=True)
                plot_widget.getAxis('bottom').setTicks([self.time_tick_labels])
            else:
                plot_widget.getAxis('bottom').setStyle(showValues=False)

        # Override the wheel event for the master plot widget
        self.plot_widgets[0].wheelEvent = self.handle_mouse_wheel

    def configure_grids(self):
        """Configure consistent grid appearance across all plots"""
        grid_alpha = 60
        grid_pen = pg.mkPen(color=(128, 128, 128, grid_alpha), width=1, style=Qt.SolidLine)
        
        for plot_widget in self.plot_widgets:
            # Configure grid lines
            plot_widget.showGrid(x=True, y=True, alpha=0.5)
            
            # Set grid pen
            plot_widget.getAxis('bottom').setGrid(grid_alpha)
            plot_widget.getAxis('left').setGrid(grid_alpha)
            
            # Ensure grid lines span full height
            plot_widget.getAxis('bottom').setStyle(tickLength=-plot_widget.height())
            
            # Set grid pen
            plot_widget.getAxis('bottom').setPen(grid_pen)
            plot_widget.getAxis('left').setPen(grid_pen)

    def update_data(self, new_data):
        """Buffer the new data for the next plot update"""
        if new_data is not None:
            self.current_data = new_data

    def update_plots(self):
        """Update all plots with current data"""
        if self.current_data is not None:
            # Create or update x_data if needed
            if self.x_data is None or len(self.x_data) != self.current_data.shape[1]:
                self.x_data = np.linspace(-4, 0, self.current_data.shape[1])
                
            # Update each curve
            for i, curve in enumerate(self.curves):
                if i < len(self.current_data):
                    # Reverse the data array for right-to-left motion
                    y_data = self.current_data[i][::-1]  # Reverse the data array
                    curve.setData(x=self.x_data, y=y_data)

    def handle_mouse_wheel(self, event):
        """Handle mouse wheel events for synchronized vertical scaling"""
        delta = event.angleDelta().y()
        if delta != 0:
            scale_change = 1.1 if delta > 0 else 0.9
            new_scale = self.scale_factor * scale_change
            
            if self.min_scale <= new_scale <= self.max_scale:
                self.scale_factor = new_scale
                for plot_widget in self.plot_widgets:
                    view_box = plot_widget.getViewBox()
                    y_range = view_box.viewRange()[1]
                    center = (y_range[0] + y_range[1]) / 2
                    height = (y_range[1] - y_range[0]) * scale_change
                    view_box.setYRange(center - height/2, center + height/2, padding=0)
        
        event.accept()

    def set_time_window(self, seconds):
        """Set the time window for visualization"""
        for plot_widget in self.plot_widgets:
            plot_widget.setXRange(-seconds, 0, padding=0.01)

    def set_color_mode(self, mode):
        """Set the color mode for all plots"""
        colors = ['w', 'r', 'b', 'g', 'm']  # white, red, blue, green, magenta
        for i, curve in enumerate(self.curves):
            if mode == 'monochrome':
                curve.setPen('w')
            elif mode == 'multicolor':
                curve.setPen(colors[i % len(colors)])