from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np

class Visualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        self.plot_items = []
        self.colors = ['r', 'g', 'b', 'y', 'p']  # Colors for each channel
        self.channel_names = ['TP9', 'FP1', 'FP2', 'TP10', 'aux']
        self.num_points = 1000  # Number of points to display
        self.data = np.zeros((5, self.num_points))  # 5 channels

        self.setup_plot()

    def setup_plot(self):
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("EEG Data Visualization")
        self.plot_widget.setLabel('left', 'Amplitude', 'µV')
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.showGrid(x=True, y=True)

        for i in range(4):
            plot_item = self.plot_widget.plot(pen=self.colors[i], name=self.channel_names[i])
            self.plot_items.append(plot_item)

    def update_data(self, new_data):
        print(f"Updating visualizer with data shape: {new_data.shape}")  # Debug print
        self.data = new_data
        self.update_plot()

    def update_plot(self):
        for i, plot_item in enumerate(self.plot_items):
            plot_item.setData(self.data[i])

    def zoom_in(self):
        self.plot_widget.getViewBox().scaleBy((0.5, 0.5))

    def zoom_out(self):
        self.plot_widget.getViewBox().scaleBy((2, 2))

    def set_color_mode(self, mode):
        if mode == 'monochrome':
            for plot_item in self.plot_items:
                plot_item.setPen('k')
        elif mode == 'multicolor':
            for i, plot_item in enumerate(self.plot_items):
                plot_item.setPen(self.colors[i])