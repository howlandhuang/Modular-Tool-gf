"""
Noise Tool Tab module for Excel Tools application.
Provides functionality for noise analysis with sub-tabs for data extraction, stack tables, and plotting.
"""

import logging
import os

from pathlib import Path
from PyQt6 import uic
from PyQt6.QtCore import QStringListModel
from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox
from func.NoiseTool import extract_noise, stack_table, noise_plot
from func.ulti import (
    get_user_input, validate_filename, validate_single_number,
    validate_frequency_list, validate_range, ValidationError, ProcessingConfig
)

# Initialize module logger
logger = logging.getLogger(__name__)

class NoiseToolTab(QWidget):
    """Tab for noise analysis operations with sub-tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)

        logger.info("Initializing Noise Tools Tab")
        # Load UI file
        self.setup_ui()

        # Setup UI and processors
        self.setup_connections()
        self.initialize_processors()
        self.stack_widget.setCurrentIndex(0)
        logger.info("Excel Tools initialization complete")

    def setup_ui(self):
        ui_path = Path(__file__).parent.parent.parent / 'ui' / 'noise_tool_tab.ui'
        logger.info(f"Loading UI from: {ui_path}")
        uic.loadUi(ui_path, self)

    def setup_connections(self):
        """Setup UI element connections and initial states."""
        logger.debug("Setting up UI connections")
        try:
            # Setup list view model
            self.model = QStringListModel()
            self.input_files_listview.setModel(self.model)
            self.output_path_text.setReadOnly(True)

            # Disable buttons initially
            self.input_selection_btn.setEnabled(False)
            self.output_selection_btn.setEnabled(False)

            # Connect tab switching buttons with the generic method
            self.data_extraction_tab_btn.clicked.connect(
                lambda: self.switch_tab(1, self.select_extraction_input))
            self.stack_table_tab_btn.clicked.connect(
                lambda: self.switch_tab(2, self.select_input))
            self.plot_tab_btn.clicked.connect(
                lambda: self.switch_tab(3, self.select_input))

            # Connect action buttons
            self.extract_btn.clicked.connect(self.execute_raw_data_extraction)
            self.noise_stack_btn.clicked.connect(self.execute_stack_noise_table)
            self.prediction_stack_btn.clicked.connect(self.execute_stack_prediction_table)
            self.by_site_btn.clicked.connect(self.execute_plot)
            self.med_only_btn.clicked.connect(self.execute_plot)
            self.min_only_btn.clicked.connect(self.execute_plot)
            self.max_only_btn.clicked.connect(self.execute_plot)
            self.save_filtered_btn.clicked.connect(self.save_filtered_result)

            # Set up mutually exclusive checkboxes
            self.setup_mutually_exclusive_checkboxes(self.f_box, self.reciprocal_f_box)

            logger.debug("All UI connections established")
        except Exception as e:
            logger.error(f"Failed to setup UI connections: {str(e)}")
            raise

    def setup_mutually_exclusive_checkboxes(self, box1, box2, default_first=True):
        """
        Setup two checkboxes to be mutually exclusive (like radio buttons).

        Args:
            box1: First checkbox
            box2: Second checkbox
            default_first: Whether to check the first box by default
        """
        # Set default states
        box1.setChecked(default_first)
        box2.setChecked(not default_first)

        # Connect click handlers
        box1.clicked.connect(lambda: self.handle_exclusive_checkboxes(box1, box2))
        box2.clicked.connect(lambda: self.handle_exclusive_checkboxes(box2, box1))

        logger.debug(f"Setup mutually exclusive checkboxes: {box1.objectName()} and {box2.objectName()}")

    def handle_exclusive_checkboxes(self, clicked_box, other_box):
        """
        Handle mutually exclusive checkbox behavior.

        Args:
            clicked_box: The checkbox that was clicked
            other_box: The other checkbox in the pair
        """
        if clicked_box.isChecked():
            # If checking the clicked box, uncheck the other
            other_box.setChecked(False)
        else:
            # If unchecking the clicked box, ensure the other is checked
            other_box.setChecked(True)

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
            self.uni_config = ProcessingConfig(
                base_path = None,
                output_path = None,
                basic_info_line_num = 8, # Line number where frequency table starts (parameters + header), here we have 7 lines of parameters and 1 line of header
                pred_range_lower = None,
                pred_range_upper = None,
                interest_freq = None,
                debug_flag = False,
                filter_outliers_flag = False,
                filter_threshold = 0.1,
                filter_tolerance = 1.0,
                auto_size = False
            )
            logger.debug("Configuration initialized")

            # Initialize processors
            self.extract_processor = extract_noise.DataProcessor(self.uni_config)
            self.stack_processor = stack_table.StackProcessor(self.uni_config)
            self.plot_processor = noise_plot.PlotProcessor(self.uni_config)
            logger.debug("All processors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processors: {str(e)}")
            raise

    def reset_button_connection(self, btn):
        """Safely disconnect all signals from a button."""
        logger.debug(f"Resetting connections for button: {btn.objectName()}")
        try:
            # Store original state
            was_blocked = btn.signalsBlocked()
            btn.blockSignals(True)

            # Disconnect all signals
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass  # No connections to disconnect

            # Restore original state
            btn.blockSignals(was_blocked)
            logger.debug("Button connections reset successfully")
        except Exception as e:
            logger.error(f"Error resetting button connections: {str(e)}")

    def switch_tab(self, tab_index, input_handler=None):
        """
        Generic method to switch between tabs.

        Args:
            tab_index: Index of the tab to switch to
            input_handler: Function to handle input selection for this tab
        """
        tab_names = {
            1: "data extraction",
            2: "stack table",
            3: "plot"
        }
        tab_name = tab_names.get(tab_index, f"index {tab_index}")

        logger.info(f"Switching to {tab_name} tab")
        try:
            self.stack_widget.setCurrentIndex(tab_index)
            self.input_selection_btn.setEnabled(True)
            self.output_selection_btn.setEnabled(True)

            # Reset and reconnect buttons
            self.reset_button_connection(self.input_selection_btn)
            self.reset_button_connection(self.output_selection_btn)

            if input_handler:
                self.input_selection_btn.clicked.connect(input_handler)
            self.output_selection_btn.clicked.connect(self.select_output)

        except Exception as e:
            logger.error(f"Error switching to {tab_name} tab: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error switching tab:\n{str(e)}")

    def update_input_display(self):
        # Handle displaying both files and directories in the list view
        if isinstance(self.uni_config.base_path, list):
            # We have a list of files
            upper_dir = [f"--Selected Files--", '']
            sub_dir = [os.path.basename(f) for f in self.uni_config.base_path]
        elif isinstance(self.uni_config.base_path, str) and os.path.isdir(self.uni_config.base_path):
            # We have a directory
            upper_dir = [f"--{self.uni_config.base_path}--", '']
            sub_dir = [f for f in os.listdir(self.uni_config.base_path)]
        else:
            # Single file or None
            upper_dir = ["--Selected File--", '']
            sub_dir = [self.uni_config.base_path] if self.uni_config.base_path else []

        self.model.setStringList(upper_dir + sub_dir)

    def select_extraction_input(self):
        """Select input directory for data extraction."""
        logger.debug("Opening directory selection dialog for extraction input")
        selected_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Raw Data Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if selected_folder:
            logger.info(f"Selected input folder: {selected_folder}")
            self.uni_config.base_path = selected_folder
            self.update_input_display()

    def select_input(self):
        """Select input files for processing."""
        logger.debug("Opening file selection dialog")
        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .xlsx File(s)",
            "",
            "Excel Files (*.xlsx)",
        )
        if selected_files:
            logger.info(f"Selected {len(selected_files)} input files")
            for f in selected_files:
                logger.info(f"Selected file: {f}")
            self.uni_config.base_path = selected_files
            self.update_input_display()

    def select_output(self):
        """Select output directory."""
        logger.debug("Opening directory selection dialog for output")
        selected_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if selected_folder:
            logger.info(f"Selected output folder: {selected_folder}")
            self.uni_config.output_path = selected_folder
            self.output_path_text.setPlainText(selected_folder)
            logger.debug("Output path updated")

    def check_io_path(self) -> bool:
        """
        Validate input and output paths based on current tab and mode.

        Requirements:
        - Input path is always required
        - Output path requirements depend on:
          - Current tab (plot tab has special debug mode handling)
          - Debug mode status (only in plot tab)

        Returns:
            bool: True if paths are valid, False otherwise
        """
        logger.debug("Checking input/output paths")

        # Always check input path first regardless of mode or tab
        if not self.uni_config.base_path:
            logger.warning("Missing input path")
            QMessageBox.warning(self, "No Input Selected", "Please select input files.")
            return False

        # Check output path based on current tab and mode
        is_plot_tab = self.stack_widget.currentIndex() == 3
        is_debug_mode = self.debug_mode_box.isChecked()

        # Special case: Plot tab in debug mode doesn't require output path
        if is_plot_tab and is_debug_mode:
            logger.debug("Plot tab in debug mode - output path not required")
            return True

        # All other cases require output path
        if not self.uni_config.output_path:
            logger.warning("Missing output path")
            QMessageBox.warning(self, "No Output Selected", "Please select output directory.")
            return False

        logger.debug("Input/output paths validated successfully")
        return True

    def update_noise_types(self):
        """Update noise types based on UI selections."""
        self.noise_types = []
        if self.sid_box.isChecked():
            self.noise_types.append('Sid')
        if self.sid_id2_box.isChecked():
            self.noise_types.append('Sid/id^2')
        if self.sid_f_box.isChecked():
            self.noise_types.append('Sid*f')
        if self.svg_box.isChecked():
            self.noise_types.append('Svg')
        if self.svg_norm_box.isChecked():
            self.noise_types.append('Svg_norm')
        if self.svg_f_box.isChecked():
            self.noise_types.append('Svg*f')

        logger.debug(f"Selected noise types: {self.noise_types}")
        if not self.noise_types:
            logger.warning("No plot types selected")
            QMessageBox.warning(self, "No Plot Type Selected", "Please select at least one plot type.")
            return

    def execute_raw_data_extraction(self):
        """Execute raw data extraction process."""
        logger.info("Starting raw data extraction")
        if not self.check_io_path():
            return

        try:
            logger.debug("Configuring extraction parameters")
            # Get configuration from UI
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
                self.extract_processor.run()
                logger.info("Raw data extraction completed successfully")

            except ValidationError as e:
                logger.warning(f"Parameter validation failed: {e.message}")
                QMessageBox.warning(self, "Input Validation Error", e.message)

        except Exception as e:
            logger.error(f"Error during raw data extraction: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred during extraction:\n{str(e)}")

    def execute_stack_noise_table(self):
        """Execute table stacking process."""
        logger.info("Starting table stacking process")
        if not self.check_io_path():
            return

        try:
            logger.debug("Using file name from user input")
            save_name = get_user_input(
                'Input File Name',
                'Enter the file name:',
                validate_filename
            )
            if save_name is None:
                logger.warning("User cancelled or invalid file name input")
                return

            logger.info(f"Starting stacking with save name: {save_name}")
            self.stack_processor.run_noise_table_stacking(save_name)
            logger.info("Table stacking completed successfully")

        except Exception as e:
            logger.error(f"Error during table stacking: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred during stacking:\n{str(e)}")

    def execute_stack_prediction_table(self):
        """Execute table stacking process."""
        logger.info("Starting table stacking process")
        if not self.check_io_path():
            return

        try:
            logger.debug("Using file name from user input")
            save_name = get_user_input(
                'Input File Name',
                'Enter the file name:',
                validate_filename
            )
            if save_name is None:
                logger.warning("User cancelled or invalid file name input")
                return

            logger.info(f"Starting stacking with save name: {save_name}")
            self.stack_processor.run_prediction_table_stacking(save_name)
            logger.info("Table stacking completed successfully")

        except Exception as e:
            logger.error(f"Error during table stacking: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred during stacking:\n{str(e)}")

    def execute_plot(self):
        """Execute plotting process."""
        logger.info("Starting plot generation")
        if not self.check_io_path():
            return

        try:
            # Check if at least one plot type is selected
            self.update_noise_types()

            # Check debug mode and save file name
            if self.debug_mode_box.isChecked():
                logger.debug("Debug mode enabled")
                self.uni_config.debug_flag = True
                save_name = 'DEBUGGING'
            else:
                logger.debug("Getting save name from user")
                self.uni_config.debug_flag = False
                save_name = get_user_input(
                    'Input File Name',
                    'Enter the file name:',
                    validate_filename
                )
                if save_name is None:
                    logger.warning("User cancelled or invalid file name input")
                    return

            # Check filter threshold and tolerance setting
            if self.filter_outliers_box.isChecked():
                logger.debug("Configuring outlier filtering")
                self.uni_config.filter_outliers_flag = True
                try:
                    self.uni_config.filter_threshold = validate_range(
                        self.filter_threshold_edit.text(), 0.0, 1.0)
                    self.uni_config.filter_tolerance = validate_range(
                        self.filter_tolerance_edit.text(), 0.0, 3.0)
                except ValidationError as e:
                    logger.warning(f"Invalid filter parameters: {e.message}")
                    QMessageBox.warning(self, "Input Validation Error", e.message)
                    return
            else:
                self.uni_config.filter_outliers_flag = False

            freq_type = None
            if self.f_box.isChecked() and not self.reciprocal_f_box.isChecked():
                freq_type = 'f'
            elif self.reciprocal_f_box.isChecked() and not self.f_box.isChecked():
                freq_type = '1/f'
            else:
                raise ValueError("Invalid frequency type")

            # Execute plot based on button clicked
            sender = self.sender()
            logger.info(f"Generating plot from {sender.objectName()}")
            if sender == self.by_site_btn:
                self.plot_processor.run_plots(self.noise_types, freq_type, 'by_site', save_name)
            elif sender == self.med_only_btn:
                self.plot_processor.run_plots(self.noise_types, freq_type, 'median_only', save_name)
            elif sender == self.min_only_btn:
                self.plot_processor.run_plots(self.noise_types, freq_type, 'min_only', save_name)
            elif sender == self.max_only_btn:
                self.plot_processor.run_plots(self.noise_types, freq_type, 'max_only', save_name)
            else:
                logger.error("Invalid button clicked")
                raise ValueError("Invalid button clicked")

            self.uni_config.debug_flag = False # Reset debug flag after plotting, so other tabs will check input&output paths correctly
            logger.info("Plot generation completed successfully")

        except Exception as e:
            logger.error(f"Error during plot generation: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred during plotting:\n{str(e)}")

    def save_filtered_result(self):
        """Save filtered data results."""
        logger.info("Starting filtered result export")
        if not self.check_io_path():
            return
        try:
            self.update_noise_types()
            self.plot_processor.save_filtered_result(self.noise_types)
            logger.info("Filtered results saved successfully")
        except Exception as e:
            logger.error(f"Error saving filtered results: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred when saving filtered result:\n{str(e)}")

    def closeEvent(self, event):
        """Handle tab close event - cleanup handled by parent."""
        logger.debug("NoiseToolTab close event triggered")
        event.accept()

    def __del__(self):
        """Cleanup when object is deleted."""
        try:
            # Clean up processors
            self.extract_processor = None
            self.stack_processor = None
            self.plot_processor = None

            # Clean up configuration
            self.uni_config = None

            # Clean up model
            if hasattr(self, 'model'):
                self.model.setStringList([])
                self.model = None

            logger.info("Excel Tools object cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during object cleanup: {str(e)}")
