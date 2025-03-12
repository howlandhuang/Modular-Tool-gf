"""
Noise Tool Tab module for Excel Tools application.
Provides functionality for noise analysis with sub-tabs for data extraction, stack tables, and plotting.
"""

from PyQt6.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt6.QtCore import QStringListModel
import logging, os
from pathlib import Path
from PyQt6 import uic
from func import ulti
from func.ulti import (
    get_user_input, validate_filename, validate_single_number,
    validate_frequency_list, validate_range, ValidationError
)
from func.NoiseTool import extract_noise
from func.NoiseTool import stack_table
from func.NoiseTool import noise_plot

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
        ui_path = Path(__file__).parent.parent.parent / 'UI' / 'noise_tool_tab.ui'
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

            # Connect tab switching buttons
            self.data_extraction_tab_btn.clicked.connect(self.switch_data_extraction_tab)
            self.stack_table_tab_btn.clicked.connect(self.switch_stack_table_tab)
            self.plot_tab_btn.clicked.connect(self.switch_plot_tab)

            # Connect action buttons
            self.extract_btn.clicked.connect(self.execute_raw_data_extraction)
            self.noise_stack_btn.clicked.connect(self.execute_stack_noise_table)
            self.prediction_stack_btn.clicked.connect(self.execute_stack_prediction_table)
            self.by_site_btn.clicked.connect(self.execute_plot)
            self.med_only_btn.clicked.connect(self.execute_plot)
            self.min_only_btn.clicked.connect(self.execute_plot)
            self.max_only_btn.clicked.connect(self.execute_plot)
            self.save_filtered_btn.clicked.connect(self.save_filtered_result)
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
            self.uni_config = ulti.ProcessingConfig(
                base_path = None,
                output_path = None,
                basic_info_line_num = 5, # Line number where frequency table starts
                pred_range_lower = None,
                pred_range_upper = None,
                interest_freq = None,
                debug_flag = False,
                filter_outliers_flag = False,
                filter_threshold = 0.1,
                filter_tolerance = 1.0
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

    def switch_data_extraction_tab(self):
        """Switch to data extraction tab."""
        logger.info("Switching to data extraction tab")
        try:
            self.stack_widget.setCurrentIndex(1)
            self.input_selection_btn.setEnabled(True)
            self.output_selection_btn.setEnabled(True)
            self.reset_button_connection(self.input_selection_btn)
            self.reset_button_connection(self.output_selection_btn)
            self.input_selection_btn.clicked.connect(self.select_extraction_input)
            self.output_selection_btn.clicked.connect(self.select_output)
            logger.debug("Data extraction tab setup complete")
        except Exception as e:
            logger.error(f"Error switching to data extraction tab: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error switching tab:\n{str(e)}")

    def switch_stack_table_tab(self):
        """Switch to stack table tab."""
        logger.info("Switching to stack table tab")
        try:
            self.stack_widget.setCurrentIndex(2)
            self.input_selection_btn.setEnabled(True)
            self.output_selection_btn.setEnabled(True)
            self.reset_button_connection(self.input_selection_btn)
            self.reset_button_connection(self.output_selection_btn)
            self.input_selection_btn.clicked.connect(self.select_input)
            self.output_selection_btn.clicked.connect(self.select_output)
            logger.debug("Stack table tab setup complete")
        except Exception as e:
            logger.error(f"Error switching to stack table tab: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error switching tab:\n{str(e)}")

    def switch_plot_tab(self):
        """Switch to plot tab."""
        logger.info("Switching to plot tab")
        try:
            self.stack_widget.setCurrentIndex(3)
            self.input_selection_btn.setEnabled(True)
            self.output_selection_btn.setEnabled(True)
            self.reset_button_connection(self.input_selection_btn)
            self.reset_button_connection(self.output_selection_btn)
            self.input_selection_btn.clicked.connect(self.select_input)
            self.output_selection_btn.clicked.connect(self.select_output)
            logger.debug("Plot tab setup complete")
        except Exception as e:
            logger.error(f"Error switching to plot tab: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error switching tab:\n{str(e)}")

    def select_extraction_input(self):
        """Select input directory for data extraction."""
        logger.debug("Opening directory selection dialog for extraction input")
        selected_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Raw Data Directory",
            r"C:\Users\hhuang10\Downloads\negative_debugging\extracted_data\new",
            QFileDialog.Option.ShowDirsOnly
        )
        if selected_folder:
            logger.info(f"Selected input folder: {selected_folder}")
            self.uni_config.base_path = selected_folder
            die_folders = [f for f in os.listdir(self.uni_config.base_path)
                          if f.startswith('Die') and
                          os.path.isdir(os.path.join(self.uni_config.base_path, f))]
            self.model.setStringList(die_folders)
            logger.info(f"Found {len(die_folders)} die folders")

    def select_input(self):
        """Select input files for processing."""
        logger.debug("Opening file selection dialog")
        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .xlsx File(s)",
            r"C:\Users\hhuang10\Documents\test_data\results",
            "Excel Files (*.xlsx)",
        )
        if selected_files:
            logger.info(f"Selected {len(selected_files)} input files")
            for f in selected_files:
                logger.info(f"Selected file: {f}")
            self.uni_config.base_path = selected_files
            self.model.setStringList([os.path.basename(file_path) for file_path in self.uni_config.base_path])
            logger.debug("Input file list updated")

    def select_output(self):
        """Select output directory."""
        logger.debug("Opening directory selection dialog for output")
        selected_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            r"C:\Users\hhuang10\Documents\test_data",
            QFileDialog.Option.ShowDirsOnly
        )
        if selected_folder:
            logger.info(f"Selected output folder: {selected_folder}")
            self.uni_config.output_path = selected_folder
            self.output_path_text.setPlainText(selected_folder)
            logger.debug("Output path updated")

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

    def execute_raw_data_extraction(self):
        """Execute raw data extraction process."""
        logger.info("Starting raw data extraction")
        if not self.check_io_path():
            return

        try:
            logger.debug("Configuring extraction parameters")
            # Get configuration from UI
            self.uni_config.debug_flag = self.debug_mode_box.isChecked()

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
            noise_types = []
            if self.sid_box.isChecked():
                noise_types.append('Sid')
            if self.sid_id2_box.isChecked():
                noise_types.append('Sid/id^2')
            if self.svg_box.isChecked():
                noise_types.append('Svg')
            if self.sid_f_box.isChecked():
                noise_types.append('Sid*f')
            if self.svg_norm_box.isChecked():
                noise_types.append('Svg_norm')

            logger.debug(f"Selected noise types: {noise_types}")
            if not noise_types:
                logger.warning("No plot types selected")
                QMessageBox.warning(self, "No Plot Type Selected", "Please select at least one plot type.")
                return

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

            # Execute plot based on button clicked
            sender = self.sender()
            logger.info(f"Generating plot from {sender.objectName()}")
            if sender == self.by_site_btn:
                self.plot_processor.run_plots(noise_types, 0, save_name)
            elif sender == self.med_only_btn:
                self.plot_processor.run_plots(noise_types, 1, save_name)
            elif sender == self.min_only_btn:
                self.plot_processor.run_plots(noise_types, 2, save_name)
            elif sender == self.max_only_btn:
                self.plot_processor.run_plots(noise_types, 3, save_name)
            else:
                logger.error("Invalid button clicked")
                raise ValueError("Invalid button clicked")

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
            self.plot_processor.save_filtered_result()
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
