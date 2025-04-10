"""
CSV Tool Tab module for Excel Tools application.
Provides functionality for processing CSV files.
"""
import os
import logging

from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt6.QtCore import QStringListModel
from pathlib import Path
from PyQt6 import uic
from func.CsvTool import inch_8
from func.ulti import (
    CsvProcessingConfig, get_user_input
)

# Initialize module logger
logger = logging.getLogger(__name__)

class CSVToolTab(QWidget):
    """Tab for CSV processing operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.info("Initializing CSV Tool Tab")
        self.setup_ui()
        self.setup_connections()
        self.initialize_processors()

        # Since it changes only when script combo box is changed,
        # we need to set it to a default value which is '8 inch', the same one as the script combo box
        self.current_script = '8 inch'

        self.input_files_model = QStringListModel()
        self.input_files_listview.setModel(self.input_files_model)

    def setup_ui(self):
        """Load the UI file."""
        ui_path = Path(__file__).parent.parent.parent / 'ui' / 'csv_tool_tab.ui'
        logger.info(f"Loading UI from: {ui_path}")
        try:
            uic.loadUi(ui_path, self)
        except Exception as e:
            logger.error(f"Failed to load UI: {str(e)}")
            raise

    def setup_connections(self):
        """Setup UI element connections and initial states."""
        logger.debug("Setting up UI connections")
        try:
            # Connect scripts combo box to log changes
            self.scripts_combo.currentTextChanged.connect(self.on_script_changed)

            # Connect input/output selection buttons
            self.input_selection_btn.clicked.connect(self.select_input)
            self.output_selection_btn.clicked.connect(self.select_output)
            self.start_btn.clicked.connect(self.start_processing)

            logger.debug("All UI connections established")
        except Exception as e:
            logger.error(f"Failed to setup UI connections: {str(e)}")
            raise

    def initialize_processors(self):
        """Initialize data processing configurations and processors."""
        logger.debug("Initializing data processors")
        try:
            # Clear any existing processors first
            if hasattr(self, 'inch8_processor'):
                self.inch8_processor = None

            # Create unified configuration with memory limits
            self.uni_config = CsvProcessingConfig(
                input_path = None,
                output_path = None,
                magnetic_fields = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
            )
            logger.debug("Configuration initialized")

            # Initialize processors
            self.inch8_processor = inch_8.DataProcessor(self.uni_config)
            logger.debug("All processors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processors: {str(e)}")
            raise

    def on_script_changed(self, script_text: str):
        """
        Handle script combo box selection changes.

        Args:
            script_text: The text of the selected script
        """
        try:
            self.current_script = script_text
            self.uni_config.input_path = None
            self.input_path_text.setText("")
            self.input_files_model.setStringList([])
            logger.info(f"Script changed to: {script_text}")
        except Exception as e:
            logger.error(f"Failed to handle script change: {str(e)}")
            raise

    def analyze_directory_structure(self, base_path: str):
        """
        Processes the directory structure to identify whether the main folder contains subfolders or just files.

        Args:
            base_path (str): The base directory path to analyze.

        Returns:
            List[str]: If the main folder contains subfolders, returns a list of full paths to all subfolders.
                      If the main folder contains only files, returns a list containing only the base_path.

        Raises:
            FileNotFoundError: If the base path does not exist.
        """
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"The base path '{base_path}' does not exist.")

        # Check if there are any subfolders in the base path
        subfolders = [os.path.join(base_path, f) for f in os.listdir(base_path)
                     if os.path.isdir(os.path.join(base_path, f))]

        if subfolders:
            # Case 1: Main folder contains subfolders
            return subfolders
        else:
            # Case 2: Main folder contains only files
            return [base_path]

    def select_input(self):
        """
        Open file dialog to select input based on current script.
        """
        def update_input_display():
            # Handle displaying both files and directories in the list view
            if isinstance(self.uni_config.input_path, list):
                # We have a list of files
                upper_dir = [f"--Selected Files--", '']
                sub_dir = [os.path.basename(f) for f in self.uni_config.input_path]
            elif isinstance(self.uni_config.input_path, str) and os.path.isdir(self.uni_config.input_path):
                # We have a directory
                upper_dir = [f"--{self.uni_config.input_path}--", '']
                sub_dir = [f for f in os.listdir(self.uni_config.input_path)]
            else:
                # Single file or None
                upper_dir = ["--Selected File--", '']
                sub_dir = [self.uni_config.input_path] if self.uni_config.input_path else []

            self.input_files_model.setStringList(upper_dir + sub_dir)

        def _select_files():
            logger.debug("Opening file selection dialog")
            selected_files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select File",
                "",
                "All Files (*)",
            )
            if selected_files:
                logger.info(f"Selected {len(selected_files)} input files")
                for f in selected_files:
                    logger.debug(f"Selected file: {f}")
                self.uni_config.input_path = selected_files
                self.input_path_text.setText(selected_files[0] if len(selected_files) == 1 else
                                        f"{selected_files[0]} (+{len(selected_files)-1} more files)")
                update_input_display()

        def _select_folder():
            logger.debug("Opening folder selection dialog")
            selected_folder = QFileDialog.getExistingDirectory(
                self,
                "Select Input Directory",
                "",
                QFileDialog.Option.ShowDirsOnly
            )
            if selected_folder:
                logger.info(f"Selected input folder: {selected_folder}")
                self.uni_config.input_path = selected_folder
                self.input_path_text.setText(selected_folder)
                update_input_display()

        if self.current_script == None:
            logger.warning("No script selected")
            QMessageBox.warning(self, "No Script Selected", "Please select a script.")
            return

        if self.current_script == "8 inch":
            _select_files()
        else:
            _select_folder()

    def select_output(self):
        """Open directory dialog to select output folder."""
        self.output_path_text.setText("")
        selected_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if selected_folder:
            logger.info(f"Selected output folder: {selected_folder}")
            self.uni_config.output_path = selected_folder
            self.output_path_text.setText(selected_folder)

    def check_io_path(self) -> bool:
        """
        Validate both input and output paths before proceeding.

        Returns:
            bool: True if both paths are valid, False otherwise
        """
        logger.debug("Checking input and output paths")

        # Check input path
        if not self.uni_config.input_path:
            logger.warning("Missing input path")
            QMessageBox.warning(self, "No Input Selected", "Please select input.")
            return False

        # Check output path
        if not self.uni_config.output_path:
            logger.warning("Missing output path")
            QMessageBox.warning(self, "No Output Selected", "Please select output directory.")
            return False

        logger.debug("Path validation successful")
        return True

    def execute_8_inch_processing(self):
        """Execute 8 inch processing."""
        self.inch8_processor.run()

    def execute_12_inch_processing(self):
        """Execute 12 inch processing."""
        pass

    def start_processing(self):
        """Start the processing of the selected input files."""
        if not self.check_io_path():
            return
        if self.current_script == "8 inch":
            self.execute_8_inch_processing()
        else:
            pass

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

