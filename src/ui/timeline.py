from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox
from PyQt5.QtCore import Qt

class Timeline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setAlignment(Qt.AlignLeft)

        self.time_label = QLabel("-4.000 s")
        self.layout.addWidget(self.time_label)

        self.window_selector = QComboBox()
        self.window_selector.addItems(["2 seconds", "4 seconds", "8 seconds"])
        self.window_selector.setCurrentIndex(1)  # Default to 4 seconds
        self.window_selector.currentIndexChanged.connect(self.window_changed)
        self.layout.addWidget(self.window_selector)

    def update_time(self, time):
        self.time_label.setText(f"{time:.3f} s")

    def window_changed(self, index):
        window_size = [2, 4, 8][index]
        # Implement logic to update visualizer's time window
        print(f"Window size changed to {window_size} seconds")