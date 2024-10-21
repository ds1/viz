from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np

class Visualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.setStyleSheet("background-color: #2D2D2D;")

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#2D2D2D')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        self.layout.addWidget(self.plot_widget)

        self.channel_names = ['Left Ear\nTP9', 'Left Forehead\nFP1', 'Right Forehead\nFP2', 'Right Ear\nTP10', 'Aux']
        self.plot_items = []
        self.num_points = 1000
        self.data = np.zeros((5, self.num_points))

        for i, name in enumerate(self.channel_names):
            label = QLabel(name)
            label.setStyleSheet("color: white; font-size: 12px;")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.layout.addWidget(label)
            
            plot_item = self.plot_widget.plot(pen='w')
            self.plot_items.append(plot_item)

        self.setup_plot()

    def setup_plot(self):
        self.plot_widget.setLabel('left', 'Amplitude', 'µV')
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.setRange(xRange=[-4, 0], padding=0)

    def update_data(self, new_data):
        self.data = new_data
        self.update_plot()

    def update_plot(self):
        for i, plot_item in enumerate(self.plot_items):
            plot_item.setData(self.data[i])

    def zoom_in(self):
        self.plot_widget.getViewBox().scaleBy((0.5, 0.5))

    def zoom_out(self):
        self.plot_widget.getViewBox().scaleBy((2, 2))

    # def set_color_mode(self, mode):
    #     if mode == 'monochrome':
    #         for plot_item in self.plot_items:
    #             plot_item.setPen('k')
    #     elif mode == 'multicolor':
    #         for i, plot_item in enumerate(self.plot_items):
    #             plot_item.setPen(self.colors[i])