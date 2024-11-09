# main.py
import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QThread

# Configure logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

class Application(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        
        # Create processing threads
        self.receiver_thread = QThread()
        self.processor_thread = QThread()
        
        # Create components
        self.setup_components()
        self.move_to_threads()
        self.connect_signals()
        self.start_processing()
        
    def setup_components(self):
        try:
            initial_data_type = DataType.EEG
            
            # Create components
            self.lsl_receiver = LSLReceiver(
                stream_type=initial_data_type.value,
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
            
    def move_to_threads(self):
        self.lsl_receiver.moveToThread(self.receiver_thread)
        self.receiver_thread.started.connect(self.lsl_receiver.connect_to_stream)
        self.data_processor.moveToThread(self.processor_thread)
        
    def connect_signals(self):
        """Connect component signals"""
        try:
            # Handle data type changes
            def on_data_type_changed(value: str):
                try:
                    enum_value = DataType(value)
                    self.change_data_type(enum_value)
                except Exception as e:
                    logging.error(f"Error converting data type: {str(e)}")
                    raise

            # Connect data flow signals
            self.lsl_receiver.data_ready.connect(self.data_processor.process_data)
            self.data_processor.processed_data.connect(
                self.main_window.visualizer.updateData
            )

            # Connect control signals
            self.main_window.control_bar.data_type_changed.connect(on_data_type_changed)
            
            # Connect status and error signals
            self.lsl_receiver.status_changed.connect(
                self.main_window.status_bar.updateStreamStatus
            )
            self.lsl_receiver.error_occurred.connect(
                self.main_window.status_bar.showError
            )
            self.data_processor.error_occurred.connect(
                self.main_window.status_bar.showError
            )
                
        except Exception as e:
            logging.error(f"Error in connect_signals: {str(e)}", exc_info=True)
            raise

    def change_data_type(self, data_type: DataType):
        try:
            self.lsl_receiver.disconnect()
            self.data_processor.enable_processing(False)
            
            self.lsl_receiver.stream_type = data_type.value
            self.data_processor.set_data_type(data_type)
            self.main_window.visualizer.set_data_type(data_type)
            
            self.lsl_receiver.connect_to_stream()
            self.data_processor.enable_processing(True)
        except Exception as e:
            logging.error(f"Error in change_data_type: {str(e)}", exc_info=True)
            raise
            
    def start_processing(self):
        self.receiver_thread.start()
        self.processor_thread.start()
        self.main_window.show()
        
    def cleanup(self):
        self.lsl_receiver.disconnect()
        self.data_processor.enable_processing(False)
        self.receiver_thread.quit()
        self.processor_thread.quit()
        self.receiver_thread.wait()
        self.processor_thread.wait()

def main():
    try:
        app = Application(sys.argv)
        return app.exec_()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())