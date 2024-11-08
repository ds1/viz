from dataclasses import dataclass
from typing import Dict, Tuple
from enum import Enum
from PyQt5.QtGui import QIcon
import os

@dataclass
class Typography:
    family: str
    size: int
    weight: int
    line_height: float = 1.5

@dataclass
class Spacing:
    """Spacing scale in pixels"""
    xxs: int = 2
    xs: int = 4
    sm: int = 8
    md: int = 16
    lg: int = 24
    xl: int = 32
    xxl: int = 48

class ColorMode(Enum):
    LIGHT = "light"
    DARK = "dark"

@dataclass
class ColorTheme:
    background: Dict[str, str] = None
    foreground: Dict[str, str] = None
    accent: Dict[str, str] = None
    grid: Dict[str, str] = None
    channels: Dict[str, str] = None
    status: Dict[str, str] = None

class Icons:
    """Icon definitions"""
    def __init__(self):
        self.pause = QIcon("src/resources/icons/pause.svg")
        self.play = QIcon("src/resources/icons/play.svg")

class DesignSystem:
    """Central design system configuration"""
    
    ICONS = Icons()

    # Typography definitions
    TYPOGRAPHY = {
        'heading': Typography(
            family='Inter',
            size=20,
            weight=600
        ),
        'channel': Typography(
            family='Inter',
            size=14,
            weight=500
        ),
        'controls': Typography(
            family='Inter',
            size=13,
            weight=400
        ),
        'label': Typography(
            family='Inter',
            size=12,
            weight=400
        ),
        'status': Typography(
            family='Inter',
            size=12,
            weight=400
        )
    }
    
    # Spacing scale
    SPACING = Spacing()
    
    # Plot configuration
    PLOT_CONFIG = {
        'channel_height': 120,
        'channel_label_width': 80,
        'time_axis_height': 40,
        'y_axis_width': 60,
        'line_width': 1.5,
        'grid_opacity': 0.3,
        'buffer_size': 1024,
        'min_refresh_rate': 30,
        'quality_threshold': 0.7
    }
    
    # Dark theme colors
    DARK_THEME = ColorTheme(
        background={
            'primary': '#1A1A1A',    # Main background
            'secondary': '#242424',   # Plot background
            'tertiary': '#2A2A2A',   # Control bar background
            'overlay': '#333333'      # Dropdown/menu background
        },
        foreground={
            'primary': '#FFFFFF',     # Primary text
            'secondary': '#B3B3B3',   # Secondary text
            'tertiary': '#808080',    # Disabled text
            'inverse': '#000000'      # Text on accent colors
        },
        accent={
            'primary': '#6B4A8C',     # Primary accent
            'secondary': '#6CC197',   # Secondary accent
            'hover': '#8E44AD',       # Hover state
            'active': '#9B59B6'       # Active state
        },
        grid={
            'major': '#404040',       # Major grid lines
            'minor': '#333333'        # Minor grid lines
        },
        channels={
            'tp9': '#4CAF50',         # Left ear
            'fp1': '#2196F3',         # Left forehead
            'fp2': '#F44336',         # Right forehead
            'tp10': '#FFC107',        # Right ear
            'aux': '#9C27B0'          # Auxiliary
        },
        status={
            'success': '#4CAF50',     # Success state
            'warning': '#FFC107',     # Warning state
            'error': '#F44336',       # Error state
            'info': '#2196F3'         # Info state
        }
    )
    
    @classmethod
    def get_plot_style(cls, element: str) -> Dict:
        """Get styling for plot elements"""
        styles = {
            'plot_background': {
                'background': cls.DARK_THEME.background['secondary'],
                'border': 'none'
            },
            'grid': {
                'major': {
                    'color': cls.DARK_THEME.grid['major'],
                    'width': 1
                },
                'minor': {
                    'color': cls.DARK_THEME.grid['minor'],
                    'width': 0.5
                }
            },
            'axis': {
                'text': {
                    'color': cls.DARK_THEME.foreground['secondary'],
                    'font-family': cls.TYPOGRAPHY['label'].family,
                    'font-size': cls.TYPOGRAPHY['label'].size
                },
                'line': {
                    'color': cls.DARK_THEME.grid['major'],
                    'width': 1
                }
            }
        }
        return styles.get(element, {})
    
    @classmethod
    def get_pause_button_style(cls) -> str:
        """Get stylesheet for pause button"""
        colors = cls.DARK_THEME
        spacing = cls.SPACING
        
        return f"""
            QPushButton {{
                background-color: {colors.accent['primary']};
                border-radius: 15px;
                padding: {spacing.sm}px;
            }}
            
            QPushButton:hover {{
                background-color: {colors.accent['hover']};
            }}
            
            QPushButton:pressed {{
                background-color: {colors.accent['active']};
            }}
        """
    
    @classmethod
    def get_style_sheet(cls) -> str:
        """Get global application stylesheet"""
        colors = cls.DARK_THEME
        typography = cls.TYPOGRAPHY
        spacing = cls.SPACING
        
        return f"""
            QMainWindow {{
                background-color: {colors.background['primary']};
                color: {colors.foreground['primary']};
            }}
            
            QLabel {{
                color: {colors.foreground['primary']};
                font-family: {typography['label'].family};
                font-size: {typography['label'].size}px;
            }}
            
            QComboBox {{
                background-color: {colors.background['tertiary']};
                border: 1px solid {colors.accent['primary']};
                border-radius: {spacing.xs}px;
                padding: {spacing.sm}px {spacing.md}px;
                color: {colors.foreground['primary']};
                font-family: {typography['controls'].family};
                font-size: {typography['controls'].size}px;
                min-width: 120px;
                min-height: 30px;
            }}
            
            QComboBox:hover {{
                border-color: {colors.accent['hover']};
            }}
            
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            
            QMenuBar {{
                background-color: {colors.background['tertiary']};
                color: {colors.foreground['primary']};
                font-family: {typography['controls'].family};
                font-size: {typography['controls'].size}px;
                min-height: 30px;
            }}
            
            QMenuBar::item:selected {{
                background-color: {colors.background['overlay']};
            }}
            
            QMenu {{
                background-color: {colors.background['overlay']};
                color: {colors.foreground['primary']};
                border: 1px solid {colors.grid['major']};
            }}
            
            QMenu::item:selected {{
                background-color: {colors.accent['primary']};
            }}
            
            QStatusBar {{
                background-color: {colors.background['tertiary']};
                color: {colors.foreground['secondary']};
                font-family: {typography['status'].family};
                font-size: {typography['status'].size}px;
                min-height: 30px;
            }}
            
            QStatusBar QLabel {{
                color: {colors.foreground['secondary']};
                font-size: {typography['status'].size}px;
                padding: 0px {spacing.sm}px;
            }}
        """