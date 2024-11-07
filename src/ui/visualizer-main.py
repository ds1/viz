class Visualizer(QWidget):
    """Main visualization widget with enhanced UX and visual design"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configure widget
        self.setObjectName("Visualizer")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(DesignSystem.PLOT_CONFIG['channel_height'] * 5 + 
                            DesignSystem.PLOT_CONFIG['time_axis_height'])
        
        # Initialize state
        self.channel_names = ['TP9', 'FP1', 'FP2', 'TP10', 'AUX']
        self.y_ranges = {
            'TP9': (-500, 500),
            'FP1': (-500, 500),
            'FP2': (-500, 500),
            'TP10': (-500, 500),
            'AUX': (-500, 500)
        }
        
        # Initialize data structures
        self.plots: List[ChannelPlot] = []
        self.curves: List[pg.PlotDataItem] = []
        self.current_data: Optional[np.ndarray] = None
        self.x_data: Optional[np.ndarray] = None
        
        # Scaling state
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0
        
        # Create and configure layouts
        self.setup_layout()
        self.setup_plots()
        self.setup_shared_time_axis()
        self.setup_update_timer()
        
    def setup_layout(self):
        """Initialize layout structure"""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Plot container
        self.plot_container = PlotContainer(self)
        self.layout.addWidget(self.plot_container)
        
        # Channel layout
        self.channel_layout = QVBoxLayout()
        self.channel_layout.setSpacing(DesignSystem.SPACING.sm)
        self.plot_container.layout.addLayout(self.channel_layout)
        
    def setup_plots(self):
        """Create and configure plot widgets"""
        # Create shared x-axis range for linking
        self.shared_x_range = pg.ViewBox()
        self.shared_x_range.setRange(xRange=(-4, 0))
        
        # Create plots for each channel
        for channel_name in self.channel_names:
            # Create row layout
            row_layout = QHBoxLayout()
            row_layout.setSpacing(DesignSystem.SPACING.md)
            
            # Add channel label
            label = QLabel(channel_name)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setFixedWidth(DesignSystem.PLOT_CONFIG['channel_label_width'])
            label.setStyleSheet(f"""
                QLabel {{
                    color: {DesignSystem.DARK_THEME.channels[channel_name.lower()]};
                    font-family: {DesignSystem.TYPOGRAPHY['channel'].family};
                    font-size: {DesignSystem.TYPOGRAPHY['channel'].size}px;
                    font-weight: {DesignSystem.TYPOGRAPHY['channel'].weight};
                }}
            """)
            row_layout.addWidget(label)
            
            # Create plot
            plot = ChannelPlot(
                channel_name=channel_name,
                y_range=self.y_ranges[channel_name]
            )
            
            # Configure plot specifics
            plot.setFixedHeight(DesignSystem.PLOT_CONFIG['channel_height'])
            view_box = plot.getViewBox()
            view_box.setMouseEnabled(x=False, y=False)
            view_box.linkView(view_box.XAxis, self.shared_x_range)
            
            # Create curve with channel-specific color
            pen = pg.mkPen({
                'color': DesignSystem.DARK_THEME.channels[channel_name.lower()],
                'width': DesignSystem.PLOT_CONFIG['line_width']
            })
            curve = plot.plot(pen=pen)
            
            # Store references
            self.plots.append(plot)
            self.curves.append(curve)
            
            # Add plot to layout
            row_layout.addWidget(plot)
            self.channel_layout.addLayout(row_layout)
            
    def setup_shared_time_axis(self):
        """Configure shared time axis and grid"""
        # Create time axis widget
        self.time_axis = pg.AxisItem('bottom')
        self.time_axis.setLabel('Time', units='s')
        self.time_axis.setHeight(DesignSystem.PLOT_CONFIG['time_axis_height'])
        
        # Configure ticks and grid
        major_ticks = [(x, f'{x:.3f}') for x in np.linspace(-4, 0, 5)]
        minor_ticks = [(x, '') for x in np.linspace(-4, 0, 17) if x not in [t[0] for t in major_ticks]]
        self.time_axis.setTicks([major_ticks, minor_ticks])
        
        # Add time axis to layout
        time_layout = QHBoxLayout()
        time_layout.addSpacing(DesignSystem.PLOT_CONFIG['channel_label_width'] + 
                             DesignSystem.SPACING.md)
        time_layout.addWidget(self.time_axis)
        self.plot_container.layout.addLayout(time_layout)
        
    def setup_update_timer(self):
        """Configure update timer for smooth visualization"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(33)  # ~30 FPS
        
    def update_data(self, new_data: np.ndarray):
        """Buffer new data for visualization"""
        if new_data is not None:
            # Store new data
            self.current_data = new_data
            
            # Initialize or update x_data if needed
            if self.x_data is None or len(self.x_data) != new_data.shape[1]:
                self.x_data = np.linspace(-4, 0, new_data.shape[1])
    
    def update_plots(self):
        """Update plot visualizations with current data"""
        if self.current_data is not None and self.x_data is not None:
            for i, curve in enumerate(self.curves):
                if i < len(self.current_data):
                    # Data flows right to left (newest data at x=0)
                    curve.setData(x=self.x_data, y=self.current_data[i])
                    
    def wheelEvent(self, event):
        """Handle mouse wheel events for synchronized vertical scaling"""
        if event.angleDelta().y() != 0:
            # Calculate scale change
            scale_change = 1.1 if event.angleDelta().y() > 0 else 0.9
            new_scale = self.scale_factor * scale_change
            
            if self.min_scale <= new_scale <= self.max_scale:
                self.scale_factor = new_scale
                
                # Calculate average center point
                centers = []
                for plot in self.plots:
                    view_box = plot.getViewBox()
                    y_range = view_box.viewRange()[1]
                    centers.append(sum(y_range) / 2)
                avg_center = sum(centers) / len(centers)
                
                # Update all plots with synchronized scaling
                for plot in self.plots:
                    view_box = plot.getViewBox()
                    y_range = view_box.viewRange()[1]
                    height = (y_range[1] - y_range[0]) * scale_change
                    view_box.setYRange(
                        avg_center - height/2,
                        avg_center + height/2,
                        padding=0
                    )
        
        event.accept()
        
    def set_color_mode(self, mode: str):
        """Update plot colors based on mode"""
        if mode == 'monochrome':
            pen = pg.mkPen({
                'color': DesignSystem.DARK_THEME.foreground['primary'],
                'width': DesignSystem.PLOT_CONFIG['line_width']
            })
            for curve in self.curves:
                curve.setPen(pen)
        else:
            for i, curve in enumerate(self.curves):
                channel_name = self.channel_names[i].lower()
                pen = pg.mkPen({
                    'color': DesignSystem.DARK_THEME.channels[channel_name],
                    'width': DesignSystem.PLOT_CONFIG['line_width']
                })
                curve.setPen(pen)

    def set_time_window(self, seconds: float):
        """Update the time window for visualization"""
        self.shared_x_range.setRange(xRange=(-seconds, 0))
