"""
CSV Tool Tab module for Excel Tools application.
Provides functionality for processing CSV files.
"""

from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt6.QtCore import QStringListModel
from pathlib import Path
import logging
from PyQt6 import uic
import os

# Initialize module logger
logger = logging.getLogger(__name__)

class CSVToolTab(QWidget):
    """Tab for CSV processing operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.info("Initializing CSV Tool Tab")
        self.setup_ui()
        self.current_mode = None  # Track current input selection mode
        self.setup_connections()
        self.input_files_model = QStringListModel()
        self.input_files_listview.setModel(self.input_files_model)
        self.input_selection_btn.setEnabled(False)  # Initially disabled
        self.output_selection_btn.setEnabled(False)

    def setup_ui(self):
        """Set up the UI components for the CSV Tool tab."""
        ui_path = Path(__file__).parent.parent.parent / 'UI' / 'csv_tool_tab.ui'
        logger.info(f"Loading UI from: {ui_path}")
        uic.loadUi(ui_path, self)

    def setup_connections(self):
        """Setup UI element connections and initial states."""
        logger.debug("Setting up UI connections")
        try:
            # Connect main buttons
            self.cascade_btn.clicked.connect(self.handle_cascade)
            self.tel_btn.clicked.connect(self.handle_tel)
            self.tohuku_btn.clicked.connect(self.handle_tohuku)
            self.inch12_btn.clicked.connect(self.handle_12inch)
            self.inch8_btn.clicked.connect(self.handle_8inch)
            self.vtgm_btn.clicked.connect(self.handle_vtgm)

            # Connect input/output selection buttons
            self.input_selection_btn.clicked.connect(self.select_input)
            self.output_selection_btn.clicked.connect(self.select_output)

            logger.debug("All UI connections established")
        except Exception as e:
            logger.error(f"Failed to setup UI connections: {str(e)}")
            raise

    def select_input(self):
        """
        Open file dialog to select input based on current mode.
        """
        def get_directory_structure(path: str, prefix: str = "") -> list:
            """
            Helper function to recursively get the directory structure with proper indentation.

            Args:
                path: The directory path to scan
                prefix: The prefix string for indentation

            Returns:
                List of strings representing the directory structure
            """
            items = []
            try:
                # Get all items in the current directory
                for item in os.listdir(path):
                    full_path = os.path.join(path, item)
                    if os.path.isdir(full_path):
                        # If it's a directory, add it with proper indentation
                        items.append(f"{prefix}{item}")
                        # Recursively get contents of subdirectories
                        items.extend(get_directory_structure(full_path, prefix + "--"))
                    else:
                        # If it's a file, add it with proper indentation
                        items.append(f"{prefix}{item}")
            except Exception as e:
                logger.error(f"Error scanning directory {path}: {str(e)}")
            return items

        selected_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Input Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if selected_folder:
            logger.info(f"Selected input folder: {selected_folder}")
            # Get hierarchical structure of the directory
            display_items = get_directory_structure(selected_folder, "-")
            self.input_files_model.setStringList(display_items)

    def select_output(self):
        """Open directory dialog to select output folder."""
        self.output_path_text.setText("")
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", ""
        )
        if folder:
            self.output_path_text.setText(folder)
            logger.info(f"Output folder set to: {folder}")

    def handle_cascade(self):
        """Handle Cascade button click."""
        logger.info("Cascade button clicked")
        self.current_mode = "folder"
        self.input_selection_btn.setEnabled(True)
        self.output_selection_btn.setEnabled(True)
        self.input_files_model.setStringList([])  # Clear previous selection

    def handle_tel(self):
        """Handle TEL button click."""
        logger.info("TEL button clicked")
        self.current_mode = "folder"
        self.input_selection_btn.setEnabled(True)
        self.output_selection_btn.setEnabled(True)
        self.input_files_model.setStringList([])  # Clear previous selection

    def handle_tohuku(self):
        """Handle Tohuku button click."""
        logger.info("Tohuku button clicked")
        self.process_files("tohuku")

    def handle_12inch(self):
        """Handle 12inch button click."""
        logger.info("12inch button clicked")
        self.process_files("12inch")

    def handle_8inch(self):
        """Handle 8inch button click."""
        logger.info("8inch button clicked")
        self.process_files("8inch")

    def handle_vtgm(self):
        """Handle Vtgm button click."""
        logger.info("Vtgm button clicked")
        self.process_files("vtgm")

    def process_files(self, processor_type: str):
        """
        Process the selected CSV files according to the specified processor type.

        Args:
            processor_type: The type of processing to apply ('cascade', 'tel', 'tohuku', etc.)
        """
        input_files = self.input_files_model.stringList()
        output_folder = self.output_path_text.toPlainText()

        if not input_files:
            logger.warning("No input files selected")
            QMessageBox.warning(self, "Warning", "Please select input files first.")
            return

        if not output_folder:
            logger.warning("No output folder selected")
            QMessageBox.warning(self, "Warning", "Please select output folder first.")
            return

        logger.info(f"Processing {len(input_files)} files with {processor_type} processor")
        # Implement the actual processing logic here based on processor_type
        # This is a placeholder for your implementation

    def closeEvent(self, event):
        """Handle tab close event - cleanup handled by parent."""
        logger.debug("CSV ToolTab close event triggered")
        event.accept()

    def __del__(self):
        """Cleanup when object is deleted."""
        try:
            if hasattr(self, 'model'):
                self.model.setStringList([])
                self.model = None

            logger.info("CSV Tool Tab object cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during object cleanup: {str(e)}")
