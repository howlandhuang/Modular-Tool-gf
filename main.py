from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt6 import uic
import sys, logging, time
from pathlib import Path
from func.excel_tools import ExcelTools
from func.ulti import setup_logger

sys.argv += ['-platform', 'windows:darkmode=1']


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("MainWindow")
        self.load_ui()
        self.setup_connections()
        self.excel_tools = None
        self.csv_viewer = None


    def load_ui(self):
        """Load the main UI file"""
        try:
            ui_path = Path(__file__).parent /'UI' / 'mainwindow.ui'
            if not ui_path.exists():
                self.logger.error(f"UI file not found: {ui_path}")
                raise FileNotFoundError(f"UI file not found: {ui_path}")

            uic.loadUi(ui_path, self)
            self.logger.info("Main UI loaded successfully")
            self.setWindowTitle("Data Processing Tool")

        except Exception as e:
            self.logger.error(f"Failed to load main UI: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load UI:\n{str(e)}"
            )
            sys.exit(1)

    def setup_connections(self):
        """Setup button connections"""
        self.excel_processing.clicked.connect(self.open_excel_tools)
        self.csv_processing.clicked.connect(self.open_csv_tools)

    def open_excel_tools(self):
        try:
            if self.excel_tools is None:
                self.logger.info("Creating Excel Tools window")
                self.excel_tools = ExcelTools()
                self.excel_tools.destroyed.connect(self.on_excel_tools_closed)

            self.excel_tools.show()
            self.excel_tools.raise_()  # Bring window to front
            self.excel_tools.activateWindow()

        except Exception as e:
            self.logger.error(f"Failed to open Excel Tools: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open Excel Tools:\n{str(e)}"
            )

    def open_csv_tools(self):
        """Placeholder for CSV viewer"""
        QMessageBox.information(
            self,
            "Info",
            "CSV viewer not implemented yet."
        )

    def on_excel_tools_closed(self):
        """Handle Excel viewer window closure"""
        self.logger.info("Excel Tools GUI closed.")
        self.excel_tools = None  # Remove reference to allow garbage collection

    def closeEvent(self, event):
        self.logger.info("Main GUI is closing.")  # Log when the window is being closed
        event.accept()  # Accept the event to proceed with closing

def create_log_path():
    try:
        data_dir = Path('logs')
        data_dir.mkdir(exist_ok=True)
    except Exception as e:
        print(f"Failed to create log directory:\n{str(e)}")
        sys.exit(1)

def handle_exception(exc_type, exc_value, exc_traceback):
    error_msg = f"An unexpected error occurred:\n{str(exc_value)}"
    print(error_msg)
    if QApplication.instance():
        QMessageBox.critical(None, "Error", error_msg)
    sys.exit(1)

def main():
    create_log_path()
    create_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    listener = setup_logger(f"./logs/app_{create_time}.txt")

    sys.excepthook = handle_exception
    # Initialize multiprocessing support before QApplication
    if sys.platform.startswith('win'):
        import multiprocessing
        multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    # app.setStyle('windowsvista')
    window = MainWindow()
    window.show()

    try:
        sys.exit(app.exec())
    finally:
        listener.stop()  # Stop the logger when the app exits


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()