# main.py
import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QThread

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Temporarily set to DEBUG for troubleshooting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('petal_viz.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

# Set High DPI attributes before any Qt instantiation
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# Import after Qt setup
from src.constants import DataType, ProcessingConfig
from src.ui.main_window import MainWindow
from src.data.lsl_receiver import LSLReceiver
from src.data.data_processor import DataProcessor
from src.data.processor_thread import DataProcessorThread  # Add this import

class Application(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        
        # Create components first
        self.setup_components()
        
        # Create and setup threads
        self.setup_threads()
        
        # Connect signals
        self.connect_signals()
        
        # Start processing
        self.start_processing()
        
    def setup_components(self):
        try:
            initial_data_type = DataType.EEG
            
            # Create components
            self.lsl_receiver = LSLReceiver(
                data_type=initial_data_type,
                buffer_size=ProcessingConfig.BUFFER_SIZES[initial_data_type],
                auto_reconnect=True
            )
            
            self.data_processor = DataProcessor(initial_data_type)
            
            self.main_window = MainWindow(
                data_processor=self.data_processor,
                lsl_receiver=self.lsl_receiver
            )
            
        except Exception as e:
            logger.error(f"Error in setup_components: {str(e)}", exc_info=True)
            raise
            
    def setup_threads(self):
        """Create and configure processing threads"""
        # LSL Receiver thread
        self.receiver_thread = QThread()
        self.lsl_receiver.moveToThread(self.receiver_thread)
        self.receiver_thread.started.connect(self.lsl_receiver.connect_to_stream)
        
        # Data Processor thread
        self.processor_thread = DataProcessorThread(self.data_processor)
        
    def connect_signals(self):
        """Connect component signals"""
        try:
            # Data flow connections
            self.lsl_receiver.data_ready.connect(self.processor_thread.add_data)
            self.processor_thread.processed_data.connect(
                self.main_window.visualizer.updateData
            )
            self.processor_thread.error_occurred.connect(
                self.main_window.status_bar.showError
            )

            # LSL Receiver status connections
            self.lsl_receiver.status_changed.connect(
                self.main_window.status_bar.updateStreamStatus
            )
            self.lsl_receiver.error_occurred.connect(
                self.main_window.status_bar.showError
            )

            # Main window connections
            self.main_window.control_bar.data_type_changed.connect(
                self.change_data_type
            )
            
        except Exception as e:
            logging.error(f"Error in connect_signals: {str(e)}", exc_info=True)
            raise
            
    def start_processing(self):
        """Start all processing threads"""
        self.processor_thread.start()
        self.receiver_thread.start()
        self.main_window.show()
        
    def cleanup(self):
        """Clean shutdown of all threads"""
        # Stop processing
        self.processor_thread.stop()
        self.lsl_receiver.disconnect()
        
        # Stop threads
        self.receiver_thread.quit()
        self.processor_thread.wait()
        self.receiver_thread.wait()
        
    def change_data_type(self, data_type: DataType):
        """Handle data type changes"""
        try:
            # Stop processing temporarily
            self.processor_thread.stop()
            self.lsl_receiver.disconnect()
            
            # Update components
            self.lsl_receiver.set_stream_type(data_type)
            self.data_processor.set_data_type(data_type)
            self.main_window.visualizer.set_data_type(data_type)
            
            # Restart processing
            self.processor_thread.start()
            self.lsl_receiver.connect_to_stream()
            
        except Exception as e:
            logging.error(f"Error in change_data_type: {str(e)}", exc_info=True)
            raise

def main():
    try:
        app = Application(sys.argv)
        return app.exec_()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())