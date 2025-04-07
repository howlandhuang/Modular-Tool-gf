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
from func.ulti import (
    CsvProcessingConfig,
    ValidationError,
    validate_frequency_list,
    validate_single_number
)
from func.CsvTool import inch_8, inch_12, Cascade, TEL, Tohuku, Vtgm

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
            self.cascade_btn.clicked.connect(self.execute_cascade)
            self.tel_btn.clicked.connect(self.execute_tel)
            self.tohuku_btn.clicked.connect(self.execute_tohuku)
            self.inch12_btn.clicked.connect(self.execute_12inch)
            self.inch8_btn.clicked.connect(self.execute_8inch)
            self.vtgm_btn.clicked.connect(self.execute_vtgm)

            # Connect input/output selection buttons
            self.input_selection_btn.clicked.connect(self.select_input)
            self.output_selection_btn.clicked.connect(self.select_output)

            logger.debug("All UI connections established")
        except Exception as e:
            logger.error(f"Failed to setup UI connections: {str(e)}")
            raise

    def initialize_processors(self):
        """Initialize data processing configurations and processors."""
        logger.debug("Initializing data processors")
        try:
            # Clear any existing processors first
            if hasattr(self, 'extract_processor'):
                self.extract_processor = None
            if hasattr(self, 'stack_processor'):
                self.stack_processor = None
            if hasattr(self, 'plot_processor'):
                self.plot_processor = None

            # Create unified configuration with memory limits
            self.uni_config = CsvProcessingConfig(
                input_file_path = None,
                output_file_path = None,
            )
            logger.debug("Configuration initialized")

            # Initialize processors

            self.inch8_processor = extract_noise.DataProcessor(self.uni_config)
            self.inch12_processor = stack_table.StackProcessor(self.uni_config)
            self.Cascade_processor = noise_plot.PlotProcessor(self.uni_config)
            logger.debug("All processors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processors: {str(e)}")
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

    def check_io_path(self) -> bool:
        """
        Validate input and output paths based on mode.

        In debug mode: only checks input path
        In normal mode: checks both input and output paths

        Returns:
            bool: True if paths are valid for current mode, False otherwise
        """
        logger.debug("Checking paths based on mode")

        # Check input path first
        if not self.uni_config.base_path:
            logger.warning("Missing input path")
            QMessageBox.warning(self, "No Input Selected", "Please select input files.")
            return False

        # Additional output path check only for normal mode
        if not self.debug_mode_box.isChecked():
            if not self.uni_config.output_path:
                logger.warning("Missing output path in normal mode")
                QMessageBox.warning(self, "No Output Selected", "Please select output directory.")
                return False

        logger.debug(f"Path validation successful in {'debug' if self.debug_mode_box.isChecked() else 'normal'} mode")
        return True

    def execute_cascade(self):
        """Handle Cascade button click."""
        logger.info("Cascade button clicked")
        self.current_mode = "folder"
        self.input_selection_btn.setEnabled(True)
        self.output_selection_btn.setEnabled(True)
        self.input_files_model.setStringList([])  # Clear previous selection

        logger.info("Starting raw data extraction")
        if not self.check_io_path():
            return

        try:
            logger.debug("Configuring extraction parameters")
            # Get configuration from UI
            self.uni_config.debug_flag = self.debug_mode_box.isChecked()
            self.uni_config.auto_size = self.auto_size_box.isChecked()
            # Validate input parameters
            logger.debug("Validating input parameters")
            try:
                pred_range_lower = validate_single_number(self.range_low_edit.text())
                pred_range_upper = validate_single_number(self.range_high_edit.text())
                interest_freq = validate_frequency_list(self.interest_freq_edit.text())

                # Set validated parameters
                self.uni_config.pred_range_lower = pred_range_lower
                self.uni_config.pred_range_upper = pred_range_upper
                self.uni_config.interest_freq = interest_freq
                logger.info("Parameters validated, starting extraction")
                self.extract_processor.process_all_devices()
                logger.info("Raw data extraction completed successfully")

            except ValidationError as e:
                logger.warning(f"Parameter validation failed: {e.message}")
                QMessageBox.warning(self, "Input Validation Error", e.message)

        except Exception as e:
            logger.error(f"Error during raw data extraction: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred during extraction:\n{str(e)}")

    def execute_tel(self):
        """Handle TEL button click."""
        logger.info("TEL button clicked")
        self.current_mode = "folder"
        self.input_selection_btn.setEnabled(True)
        self.output_selection_btn.setEnabled(True)
        self.input_files_model.setStringList([])  # Clear previous selection

    def execute_tohuku(self):
        """Handle Tohuku button click."""
        logger.info("Tohuku button clicked")
        self.process_files("tohuku")

    def execute_12inch(self):
        """Handle 12inch button click."""
        logger.info("12inch button clicked")
        self.process_files("12inch")

    def execute_8inch(self):
        """Handle 8inch button click."""
        logger.info("8inch button clicked")
        self.process_files("8inch")

    def execute_vtgm(self):
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
