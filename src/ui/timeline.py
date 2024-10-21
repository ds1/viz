from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QComboBox
from PyQt5.QtCore import Qt

class Timeline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        times = ["-4.000 s", "-3.000 s", "-2.000 s", "-1.000 s", "0"]
        for time in times:
            label = QLabel(time)
            label.setStyleSheet("color: #AAAAAA; font-size: 10px;")
            label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
            layout.addWidget(label)
            if time != "0":
                layout.addStretch(1)

        self.setFixedHeight(30)
        self.setStyleSheet("background-color: #1E1E1E;")

    def update_time(self, time):
        self.time_label.setText(f"{time:.3f} s")

    def window_changed(self, index):
        window_size = [2, 4, 8][index]
        # Implement logic to update visualizer's time window
        print(f"Window size changed to {window_size} seconds")