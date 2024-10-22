from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import QByteArray

class SvgIcon(QSvgWidget):
    def __init__(self, size=24, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def update_icon(self, svg_content):
        self.load(QByteArray(svg_content.encode('utf-8')))

class IconManager:
    _icons = {}

    @classmethod
    def get_icon(cls, name, size=24):
        if name not in cls._icons:
            cls._icons[name] = SvgIcon(size)
        return cls._icons[name]

    @staticmethod
    def create_svg_icon(icon_name, color):
        svg_content = {
            "menu_down": f'<path fill="{color}" d="M7,10L12,15L17,10H7Z" />',
            "battery_0": f'<path fill="{color}" d="M16,20H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z" />',
            "battery_25": f'<path fill="{color}" d="M16,18H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z" />',
            "battery_50": f'<path fill="{color}" d="M16,15H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z" />',
            "battery_75": f'<path fill="{color}" d="M16,10H8V6H16M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z" />',
            "battery_100": f'<path fill="{color}" d="M16.67,4H15V2H9V4H7.33A1.33,1.33 0 0,0 6,5.33V20.67C6,21.4 6.6,22 7.33,22H16.67A1.33,1.33 0 0,0 18,20.67V5.33C18,4.6 17.4,4 16.67,4Z" />',
            "bluetooth_connected": f'<path fill="{color}" d="M14.88,16.29L13,18.17V14.41M13,5.83L14.88,7.71L13,9.58M17.71,7.71L12,2H11V9.58L6.41,5L5,6.41L10.59,12L5,17.58L6.41,19L11,14.41V22H12L17.71,16.29L13.41,12L17.71,7.71Z" />',
            "bluetooth_disconnected": f'<path fill="{color}" d="M13,5.83L14.88,7.71L13,9.58L14.88,11.46L13,13.33L14.88,15.21L13,17.08V22H12L7.41,17.41L8.83,16L12,19.17V15.89L9.41,13.31L8,14.72L7.29,14L12,9.29V5.59L9.41,3L8,4.41L7.29,3.71L12,1V5.83M16.59,6L15.17,7.41L17.17,9.41L18.58,8M17.17,14.59L15.17,16.59L16.59,18L18.58,16.09L17.17,14.59Z" />',
            "pause": f'<path fill="{color}" d="M14,19H18V5H14M6,19H10V5H6V19Z" />',
        }
        return f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            {svg_content.get(icon_name, '')}
        </svg>
        """

    @classmethod
    def update_icon(cls, name, icon_name, color):
        icon = cls.get_icon(name)
        svg_content = cls.create_svg_icon(icon_name, color)
        icon.update_icon(svg_content)