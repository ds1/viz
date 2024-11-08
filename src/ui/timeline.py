# Add to src/ui/timeline.py (new file)

from PyQt5.QtWidgets import QWidget, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor

from src.ui.design_system import DesignSystem

class Timeline(QWidget):
    """Timeline widget for displaying time markers"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)  # Fixed height for timeline
        self.setMinimumWidth(200)
        
        # Apply styling
        self.setStyleSheet(f"""
            Timeline {{
                background-color: {DesignSystem.DARK_THEME.background['secondary']};
                border-radius: {DesignSystem.SPACING.sm}px;
            }}
        """)
        
    def paintEvent(self, event):
        """Draw timeline markers"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Configure pen
        pen = QPen(QColor(DesignSystem.DARK_THEME.grid['major']))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        
        # Draw markers
        width = self.width()
        height = self.height()
        time_interval = width / 4  # 4 major divisions
        
        for i in range(5):  # 0 to 4 seconds
            x = i * time_interval
            painter.drawLine(x, 0, x, height / 2)
            
            # Draw time label
            if i < 4:  # Don't draw at the end
                label = f"{i}.000s"
                painter.drawText(
                    x + 5,
                    height - 10,
                    label
                )
