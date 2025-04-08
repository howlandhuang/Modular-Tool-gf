"""
Modular Tools module for data processing and visualization.
Provides a GUI interface for various Excel-related operations using a modular tab approach.
This module can be run directly as the main entry point for the application.
"""

from PyQt6 import QtWidgets, uic, QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QPixmap, QPalette, QBrush
import logging
import sys
import time
from pathlib import Path
from multiprocessing import freeze_support
from func.ulti import setup_logger
from func.NoiseTool.NoiseToolTab import NoiseToolTab

# Initialize module logger
logger = logging.getLogger(__name__)

# Set a higher logging level for PyQt6.uic
logging.getLogger('PyQt6.uic').setLevel(logging.ERROR)  # or logging.ERROR

class ModularTools(QtWidgets.QWidget):
    """
    Main widget for Excel processing tools.
    Uses a modular approach with separate tab widgets.
    """

    def __init__(self):
        """Initialize the Modular Tools widget and load UI."""
        super().__init__()
        logger.info("Initializing Modular Tools widget")
        self.version = "distribution version"
        self.setup_ui()

        # Initialize tab modules
        self.initialize_tabs()

        # Set background watermark
        watermark_path = Path(__file__).parent / 'ui' / 'watermark.jpg'
        self.set_background_watermark(str(watermark_path))

        logger.info("Modular Tools initialization complete")

    def setup_ui(self):
        """Load the UI file."""
        ui_path = Path(__file__).parent / 'UI' / 'modular_tools.ui'
        logger.info(f"Loading UI from: {ui_path}")
        try:
            uic.loadUi(ui_path, self)
            self.setWindowTitle(f"Modular Tools - {self.version}")
        except Exception as e:
            logger.error(f"Failed to load UI: {str(e)}")
            raise

    def set_background_watermark(self, image_path: str) -> None:
        """
        Set a watermark as the background of the application.

        Args:
            image_path: Path to the watermark image
        """
        try:
            logger.info(f"Setting background watermark from: {image_path}")
            # Create a pixmap from the image file
            pixmap = QPixmap(image_path)

            if pixmap.isNull():
                logger.error(f"Failed to load watermark image from {image_path}")
                return

            # Scale the pixmap to fit the widget
            scaled_pixmap = pixmap.scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                QtCore.Qt.TransformationMode.SmoothTransformation
            )

            # Create a palette
            palette = self.palette()
            brush = QBrush(scaled_pixmap)
            palette.setBrush(QPalette.ColorRole.Window, brush)
            self.setPalette(palette)

            # Make sure the background is visible
            self.setAutoFillBackground(True)

            # Connect resize event to update watermark size when window is resized
            self.resizeEvent = self.update_watermark_on_resize

            logger.info("Background watermark set successfully")
        except Exception as e:
            logger.error(f"Error setting background watermark: {str(e)}")

    def update_watermark_on_resize(self, event: QtGui.QResizeEvent) -> None:
        """
        Update the watermark when the window is resized.

        Args:
            event: The resize event
        """
        try:
            # Re-apply the watermark with new dimensions
            watermark_path = Path(__file__).parent / 'ui' / 'watermark.jpg'
            self.set_background_watermark(str(watermark_path))
            # Call the parent class's resize event handler
            super().resizeEvent(event)
        except Exception as e:
            logger.error(f"Error updating watermark on resize: {str(e)}")
            super().resizeEvent(event)

    def initialize_tabs(self):
        """Initialize tab module instances and add them to the tab widget."""
        logger.info("Initializing tab modules")

        # Create tab instances
        self.noise_tab = NoiseToolTab()

        # Clear existing tabs and add our custom tabs
        self.tabWidget.clear()

        # Add tabs to the tab widget
        self.tabWidget.addTab(self.noise_tab, "Noise Tool")

        logger.info("Tab modules initialized and added to tab widget")

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Modular Tools window closing - cleaning up tabs")
        try:
            # Clean up tabs explicitly

            if hasattr(self, 'noise_tab'):
                self.noise_tab = None
            # Clear tab widget
            self.tabWidget.clear()

            logger.info("All tabs cleaned up successfully")
            self.hide()
            self.deleteLater()
            event.accept()
        except Exception as e:
            logger.error(f"Error during window closure: {str(e)}")
            event.accept()  # Still close even if cleanup fails

    def __del__(self):
        """Cleanup when object is deleted."""
        logger.info("Modular Tools object being deleted")

def create_log_path():
    """Create logs directory if it doesn't exist."""
    logger = logging.getLogger("Setup")
    try:
        data_dir = Path('logs')
        logger.debug(f"Creating log directory at: {data_dir.absolute()}")
        data_dir.mkdir(exist_ok=True)
        logger.info("Log directory created/verified")
        return data_dir
    except Exception as e:
        logger.error(f"Failed to create log directory: {str(e)}")
        print(f"Failed to create log directory:\n{str(e)}")
        sys.exit(1)

def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler to show error messages to user."""
    logger = logging.getLogger("ExceptionHandler")
    error_msg = f"An unexpected error occurred:\n{str(exc_value)}"
    logger.error(f"Uncaught exception: {error_msg}", exc_info=(exc_type, exc_value, exc_traceback))
    print(error_msg)
    if QApplication.instance():
        QMessageBox.critical(None, "Error", error_msg)
    sys.exit(1)

def main():
    """
    Main function for standalone execution.
    Creates and shows the Modular Tools window for UI testing.
    """
    try:
        # Initialize logging
        create_log_path()
        create_time = time.strftime("%Y_%m_%d_%H_%M_%S")
        log_file = f"./logs/modular_tools_{create_time}.txt"
        listener = setup_logger(log_file)
        logger = logging.getLogger("ModularToolsStandalone")
        logger.info("Starting Modular Tools in standalone mode")

        # Set global exception handler
        sys.excepthook = handle_exception
        logger.debug("Global exception handler set")

        # Enable Windows dark mode
        sys.argv += ['-platform', 'windows:darkmode=1']

        # Create application
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  # Use Fusion style for consistent look

        # Create and show Modular Tools window
        window = ModularTools()
        window.show()

        # Start event loop
        logger.info("Starting application event loop")
        try:
            return_code = app.exec()
            logger.info(f"Application exiting with code: {return_code}")
            sys.exit(return_code)
        finally:
            logger.info("Shutting down logging")
            listener.stop()  # Ensure logger is properly closed

    except Exception as e:
        print(f"Critical error during startup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()