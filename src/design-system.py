from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum

class ColorMode(Enum):
    DARK = "dark"
    LIGHT = "light"  # Reserved for future use

@dataclass
class Colors:
    background: Dict[str, str]
    foreground: Dict[str, str]
    accent: Dict[str, str]
    grid: Dict[str, str]
    channels: Dict[str, str]
    states: Dict[str, str]

@dataclass
class Typography:
    family: str
    size: int
    weight: int
    line_height: float
    letter_spacing: float = 0.0

@dataclass
class Spacing:
    xs: int = 4
    sm: int = 8
    md: int = 16
    lg: int = 24
    xl: int = 32
    xxl: int = 48

class DesignSystem:
    DARK_THEME = Colors(
        background={
            'primary': '#1A1B1E',    # Main app background
            'secondary': '#232429',   # Plot background
            'tertiary': '#2A2B30',   # Control elements background
            'overlay': '#2F3136'     # Modal/overlay background
        },
        foreground={
            'primary': '#FFFFFF',     # Primary text
            'secondary': '#A0A0A0',   # Secondary text
            'tertiary': '#666666',    # Disabled/subtle text
            'inverse': '#1A1B1E'      # Inverse text (on accent)
        },
        accent={
            'primary': '#C58BFF',     # Primary accent
            'secondary': '#9B6CC9',   # Secondary accent
            'hover': '#D7A5FF',       # Hover state
            'active': '#B371FF'       # Active state
        },
        grid={
            'major': '#3A3B40',       # Major grid lines
            'minor': '#2A2B30'        # Minor grid lines
        },
        channels={
            'tp9': '#4A9BFF',         # Left ear
            'fp1': '#FF6B6B',         # Left forehead
            'fp2': '#4ECB71',         # Right forehead
            'tp10': '#FFB86C',        # Right ear
            'aux': '#B48EAD'          # Auxiliary
        },
        states={
            'error': '#FF5555',
            'warning': '#FFB86C',
            'success': '#50FA7B',
            'info': '#8BE9FD'
        }
    )

    TYPOGRAPHY = {
        'heading': Typography(
            family='Inter',
            size=20,
            weight=600,
            line_height=1.2,
            letter_spacing=-0.2
        ),
        'channel': Typography(
            family='Inter',
            size=14,
            weight=600,
            line_height=1.2
        ),
        'axis': Typography(
            family='Inter',
            size=12,
            weight=500,
            line_height=1.0
        ),
        'controls': Typography(
            family='Inter',
            size=14,
            weight=600,
            line_height=1.0
        ),
        'label': Typography(
            family='Inter',
            size=13,
            weight=500,
            line_height=1.0
        )
    }

    SPACING = Spacing()

    PLOT_CONFIG = {
        'channel_height': 120,        # Height of each channel plot
        'time_axis_height': 40,       # Height of time axis
        'y_axis_width': 60,          # Width of y-axis
        'channel_label_width': 100,   # Width of channel labels
        'grid_opacity': 0.3,         # Opacity of grid lines
        'line_width': 1.5,           # Width of plot lines
        'padding': SPACING.md         # Standard plot padding
    }

    @classmethod
    def get_style_sheet(cls, color_mode: ColorMode = ColorMode.DARK) -> str:
        """Generate Qt stylesheet based on design system"""
        colors = cls.DARK_THEME if color_mode == ColorMode.DARK else cls.DARK_THEME  # Fallback to dark for now
        
        return f"""
            QWidget {{
                background-color: {colors.background['primary']};
                color: {colors.foreground['primary']};
                font-family: {cls.TYPOGRAPHY['label'].family};
            }}
            
            QLabel {{
                color: {colors.foreground['primary']};
                font-size: {cls.TYPOGRAPHY['label'].size}px;
                font-weight: {cls.TYPOGRAPHY['label'].weight};
            }}
            
            QPushButton {{
                background-color: {colors.accent['primary']};
                border: none;
                border-radius: {cls.SPACING.sm}px;
                padding: {cls.SPACING.sm}px {cls.SPACING.md}px;
                color: {colors.foreground['inverse']};
                font-weight: {cls.TYPOGRAPHY['controls'].weight};
                font-size: {cls.TYPOGRAPHY['controls'].size}px;
            }}
            
            QPushButton:hover {{
                background-color: {colors.accent['hover']};
            }}
            
            QPushButton:pressed {{
                background-color: {colors.accent['active']};
            }}
            
            QComboBox {{
                background-color: {colors.background['tertiary']};
                border: 1px solid {colors.accent['primary']};
                border-radius: {cls.SPACING.xs}px;
                padding: {cls.SPACING.sm}px;
                color: {colors.foreground['primary']};
                font-size: {cls.TYPOGRAPHY['controls'].size}px;
                min-width: 100px;
            }}
            
            QComboBox::drop-down {{
                border: none;
                padding-right: {cls.SPACING.sm}px;
            }}
        """

    @classmethod
    def get_plot_style(cls, element: str) -> dict:
        """Get plot-specific styling"""
        colors = cls.DARK_THEME
        
        styles = {
            'plot_background': {
                'background': colors.background['secondary'],
                'border': 'none',
            },
            'grid': {
                'major': {'color': colors.grid['major'], 'width': 1},
                'minor': {'color': colors.grid['minor'], 'width': 1},
            },
            'axis': {
                'color': colors.foreground['secondary'],
                'font': cls.TYPOGRAPHY['axis'],
            }
        }
        
        return styles.get(element, {})
