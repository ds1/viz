from PyQt5.QtGui import QFontDatabase, QFont
from src.ui.design_system import DesignSystem, ColorMode

class StylesheetManager:
    """Manages application-wide styling and fonts"""
    
    @staticmethod
    def setup_fonts():
        """Load and register custom fonts"""
        font_db = QFontDatabase()
        
        # Load Inter font family
        font_paths = {
            'regular': 'src/resources/fonts/Inter-Regular.ttf',
            'medium': 'src/resources/fonts/Inter-Medium.ttf',
            'semibold': 'src/resources/fonts/Inter-SemiBold.ttf',
            'bold': 'src/resources/fonts/Inter-Bold.ttf'
        }
        
        for style, path in font_paths.items():
            font_id = font_db.addApplicationFont(path)
            if font_id < 0:
                print(f"Warning: Failed to load font: {path}")

    @staticmethod
    def get_control_bar_style():
        """Get stylesheet for control bar"""
        colors = DesignSystem.DARK_THEME
        typography = DesignSystem.TYPOGRAPHY
        spacing = DesignSystem.SPACING
        
        return f"""
            QWidget#ControlBar {{
                background-color: {colors.background['tertiary']};
                border-bottom: 1px solid {colors.grid['major']};
                min-height: {spacing.xxl}px;
                padding: {spacing.sm}px {spacing.md}px;
            }}
            
            QLabel {{
                color: {colors.foreground['primary']};
                font-family: {typography['controls'].family};
                font-size: {typography['controls'].size}px;
                font-weight: {typography['controls'].weight};
            }}
            
            QComboBox {{
                background-color: {colors.background['primary']};
                border: 1px solid {colors.accent['primary']};
                border-radius: {spacing.xs}px;
                padding: {spacing.sm}px {spacing.md}px;
                min-width: 120px;
                color: {colors.foreground['primary']};
            }}
            
            QComboBox::drop-down {{
                border: none;
                width: {spacing.lg}px;
            }}
            
            QComboBox::down-arrow {{
                image: url(src/resources/icons/chevron-down.svg);
                width: {spacing.md}px;
                height: {spacing.md}px;
            }}
            
            QComboBox:hover {{
                border-color: {colors.accent['hover']};
            }}
            
            QComboBox:focus {{
                border-color: {colors.accent['active']};
            }}
        """

    @staticmethod
    def get_menu_bar_style():
        """Get stylesheet for menu bar"""
        colors = DesignSystem.DARK_THEME
        typography = DesignSystem.TYPOGRAPHY
        spacing = DesignSystem.SPACING
        
        return f"""
            QMenuBar {{
                background-color: {colors.background['tertiary']};
                border-bottom: 1px solid {colors.grid['major']};
            }}
            
            QMenuBar::item {{
                background-color: transparent;
                padding: {spacing.sm}px {spacing.md}px;
                color: {colors.foreground['primary']};
                font-family: {typography['controls'].family};
                font-size: {typography['controls'].size}px;
            }}
            
            QMenuBar::item:selected {{
                background-color: {colors.background['overlay']};
            }}
            
            QMenu {{
                background-color: {colors.background['overlay']};
                border: 1px solid {colors.grid['major']};
                padding: {spacing.xs}px 0px;
            }}
            
            QMenu::item {{
                padding: {spacing.sm}px {spacing.lg}px;
                color: {colors.foreground['primary']};
            }}
            
            QMenu::item:selected {{
                background-color: {colors.accent['primary']};
                color: {colors.foreground['inverse']};
            }}
        """

    @staticmethod
    def get_status_bar_style():
        """Get stylesheet for status bar"""
        colors = DesignSystem.DARK_THEME
        typography = DesignSystem.TYPOGRAPHY
        spacing = DesignSystem.SPACING
        
        return f"""
            QStatusBar {{
                background-color: {colors.background['tertiary']};
                border-top: 1px solid {colors.grid['major']};
                min-height: {spacing.xl}px;
            }}
            
            QStatusBar QLabel {{
                color: {colors.foreground['secondary']};
                font-size: {typography['label'].size}px;
                padding: 0px {spacing.sm}px;
            }}
            
            QStatusBar::item {{
                border: none;
            }}
        """
