import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QThread

from .ui.main_window import MainWindow
from .data.lsl_receiver import LSLReceiver
from .data.data_processor import DataProcessor
from .constants import DataType, ProcessingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Application(QApplication):
    """Main application with integrated data pipeline"""
    
    def __init__(self, argv):
        super().__init__(argv)
        
        # Enable high DPI scaling
        self.setAttribute(Qt.AA_EnableHighDpiScaling)
        self.setAttribute(Qt.AA_UseHighDpiPixmaps)
        
        # Create processing threads
        self.receiver_thread = QThread()
        self.processor_thread = QThread()
        
        # Create components
        self.setup_components()
        
        # Move components to threads
        self.move_to_threads()
        
        # Connect signals
        self.connect_signals()
        
        # Start processing
        self.start_processing()
        
    def setup_components(self):
        """Create and initialize components"""
        # Create LSL receiver
        self.lsl_receiver = LSLReceiver(
            stream_type=DataType.EEG,
            buffer_size=ProcessingConfig.BUFFER_SIZES[DataType.EEG],
            auto_reconnect=True
        )
        
        # Create data processor
        self.data_processor = DataProcessor(DataType.EEG)
        
        # Create main window
        self.main_window = MainWindow()
        
    def move_to_threads(self):
        """Move components to their respective threads"""
        # Move receiver to thread
        self.lsl_receiver.moveToThread(self.receiver_thread)
        self.receiver_thread.started.connect(self.lsl_receiver.connect_to_stream)
        
        # Move processor to thread
        self.data_processor.moveToThread(self.processor_thread)
        
    def connect_signals(self):
        """Connect component signals"""
        # LSL Receiver -> Data Processor
        self.lsl_receiver.data_ready.connect(self.data_processor.process_data)
        
        # Data Processor -> Visualizer
        self.data_processor.processed_data.connect(
            self.main_window.visualizer.updateData
        )
        
        # Status Updates
        self.lsl_receiver.status_changed.connect(
            self.main_window.status_bar.update_stream_status
        )
        self.lsl_receiver.stream_info_updated.connect(
            self.main_window.update_stream_info
        )
        self.lsl_receiver.quality_updated.connect(
            self.main_window.status_bar.update_quality_metrics
        )
        
        # Error Handling
        self.lsl_receiver.error_occurred.connect(
            self.main_window.status_bar.show_error
        )
        self.data_processor.error_occurred.connect(
            self.main_window.status_bar.show_error
        )
        
        # Control Updates
        self.main_window.control_bar.data_type_changed.connect(
            self.change_data_type
        )
        self.main_window.control_bar.filter_changed.connect(
            self.data_processor.set_filter
        )
        self.main_window.control_bar.scale_changed.connect(
            self.main_window.visualizer.set_scale
        )
        self.main_window.control_bar.window_changed.connect(
            self.main_window.visualizer.set_time_window
        )
        
        # Cleanup
        self.main_window.closing.connect(self.cleanup)
        
    def start_processing(self):
        """Start processing threads"""
        self.receiver_thread.start()
        self.processor_thread.start()
        
        # Show main window
        self.main_window.show()
        
    def change_data_type(self, data_type: DataType):
        """Handle data type changes"""
        # Stop current processing
        self.lsl_receiver.disconnect()
        self.data_processor.enable_processing(False)
        
        # Update components
        self.lsl_receiver.stream_type = data_type
        self.data_processor.set_data_type(data_type)
        self.main_window.visualizer.set_data_type(data_type)
        
        # Restart processing
        self.lsl_receiver.connect_to_stream()
        self.data_processor.enable_processing(True)
        
    def cleanup(self):
        """Clean up resources"""
        # Stop processing
        self.lsl_receiver.disconnect()
        self.data_processor.enable_processing(False)
        
        # Stop threads
        self.receiver_thread.quit()
        self.processor_thread.quit()
        
        # Wait for threads to finish
        self.receiver_thread.wait()
        self.processor_thread.wait()
        
def main():
    """Application entry point"""
    try:
        app = Application(sys.argv)
        return app.exec_()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())