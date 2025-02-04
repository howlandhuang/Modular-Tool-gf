"""
Main entry point for the Data Processing Tool application.
This is a PyQt6-based GUI application for processing and analyzing data.
"""

from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt6 import uic
import sys
import logging
import time
from pathlib import Path
from func.excel_tools import ExcelTools
from func.ulti import setup_logger

# Enable Windows dark mode
sys.argv += ['-platform', 'windows:darkmode=1']

class MainWindow(QWidget):
    """
    Main window of the application.
    Provides access to Excel and CSV processing tools.
    """

    def __init__(self):
        """Initialize the main window and setup logging."""
        super().__init__()
        self.logger = logging.getLogger("MainWindow")
        self.load_ui()
        self.setup_connections()

        # Initialize tool windows as None
        self.excel_tools = None
        self.csv_viewer = None

    def load_ui(self):
        """
        Load the main UI file from the UI directory.
        Raises FileNotFoundError if UI file is not found.
        """
        try:
            # Construct path to UI file
            ui_path = Path(__file__).parent / 'UI' / 'mainwindow.ui'
            if not ui_path.exists():
                self.logger.error(f"UI file not found: {ui_path}")
                raise FileNotFoundError(f"UI file not found: {ui_path}")

            # Load UI and set window title
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
        """Setup button click event handlers."""
        self.excel_processing.clicked.connect(self.open_excel_tools)
        self.csv_processing.clicked.connect(self.open_csv_tools)

    def open_excel_tools(self):
        """
        Open the Excel processing tools window.
        Creates a new window if none exists, otherwise brings existing window to front.
        """
        try:
            if self.excel_tools is None:
                self.logger.info("Creating Excel Tools window")
                self.excel_tools = ExcelTools()
                # Connect the destroyed signal to cleanup handler
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
        """Placeholder for future CSV processing tools."""
        QMessageBox.information(
            self,
            "Info",
            "CSV viewer not implemented yet."
        )

    def on_excel_tools_closed(self):
        """
        Handle cleanup when Excel tools window is closed.
        Removes reference to allow garbage collection.
        """
        self.logger.info("Excel Tools GUI closed.")
        self.excel_tools = None

    def closeEvent(self, event):
        """Handle application shutdown."""
        self.logger.info("Main GUI is closing.")
        event.accept()

def create_log_path():
    """Create logs directory if it doesn't exist."""
    try:
        data_dir = Path('logs')
        data_dir.mkdir(exist_ok=True)
    except Exception as e:
        print(f"Failed to create log directory:\n{str(e)}")
        sys.exit(1)

def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler to show error messages to user."""
    error_msg = f"An unexpected error occurred:\n{str(exc_value)}"
    print(error_msg)
    if QApplication.instance():
        QMessageBox.critical(None, "Error", error_msg)
    sys.exit(1)

def main():
    """
    Application entry point.
    Sets up logging, creates and shows main window.
    """
    # Initialize logging
    create_log_path()
    create_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    listener = setup_logger(f"./logs/app_{create_time}.txt")

    # Set global exception handler
    sys.excepthook = handle_exception

    # Initialize multiprocessing support for Windows
    if sys.platform.startswith('win'):
        import multiprocessing
        multiprocessing.freeze_support()

    # Create and configure application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for consistent look

    # Create and show main window
    window = MainWindow()
    window.show()

    try:
        sys.exit(app.exec())
    finally:
        listener.stop()  # Ensure logger is properly closed

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()