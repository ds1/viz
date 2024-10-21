from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np

class Visualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.setStyleSheet("background-color: #2D2D2D;")

        self.channel_names = ['Left Ear\nTP9', 'Left Forehead\nFP1', 'Right Forehead\nFP2', 'Right Ear\nTP10', 'Aux']
        self.plot_widgets = []

        for name in self.channel_names:
            row_layout = QHBoxLayout()
            
            label = QLabel(name)
            label.setStyleSheet("color: white; font-size: 12px;")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_layout.addWidget(label, 1)

            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('#2D2D2D')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            row_layout.addWidget(plot_widget, 4)

            self.layout.addLayout(row_layout)
            self.plot_widgets.append(plot_widget)

        self.setup_plots()

    def setup_plots(self):
        for plot_widget in self.plot_widgets:
            plot_widget.setLabel('left', 'µV')
            plot_widget.setLabel('bottom', 'Time', 's')
            plot_widget.setRange(xRange=[-4, 0], yRange=[-200, 200])
            plot_widget.getAxis('bottom').setStyle(showValues=False)
            plot_widget.setMouseEnabled(x=True, y=False)
            plot_widget.invertX(True)

    def update_data(self, new_data):
        for i, plot_widget in enumerate(self.plot_widgets):
            if i < len(new_data):
                plot_widget.plot(new_data[i], clear=True, pen='w')

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