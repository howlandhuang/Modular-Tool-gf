"""
Excel Tools module for data processing and visualization.
Provides a GUI interface for various Excel-related operations including:
- Data extraction
- Table stacking
- Data plotting
"""

from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem, QInputDialog
from PyQt6.QtCore import QStringListModel
import os, logging
from pathlib import Path
from multiprocessing import freeze_support
from func import noise_plot
from func import stacked_table
from func import extract_noise
from func import ulti

# Initialize module logger
logger = logging.getLogger(__name__)
freeze_support()

class ExcelTools(QtWidgets.QWidget):
    """
    Main widget for Excel processing tools.
    Provides interface for data extraction, stacking and plotting operations.
    """

    def __init__(self):
        """Initialize the Excel Tools widget and load UI."""
        super().__init__()
        logger.info("Initializing Excel Tools widget")

        # Load UI file
        ui_path = Path(__file__).parent.parent / 'UI' / 'exceltools.ui'
        logger.debug(f"Loading UI from: {ui_path}")
        uic.loadUi(ui_path, self)
        self.setWindowTitle("Excel Tools")

        # Initialize state variables
        logger.debug("Initializing state variables")
        self.sheets = None
        self.current_excel = None
        self.current_df = None

        # Setup UI and processors
        self.setup_connections()
        self.initialize_processors()
        self.stack_widget.setCurrentIndex(0)
        logger.info("Excel Tools initialization complete")

    def initialize_processors(self):
        """Initialize data processing configurations and processors."""
        logger.debug("Initializing data processors")
        try:
            # Create unified configuration
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
                filter_tolerance = 1.0,
                prediction_only_flag = False
            )
            logger.debug("Configuration initialized")

            # Initialize processors
            self.validator_processor = ulti.InputValidator()
            self.extract_processor = extract_noise.DataProcessor(self.uni_config)
            self.stack_processor = stacked_table.StackProcessor(self.uni_config)
            self.plot_processor = noise_plot.PlotProcessor(self.uni_config)
            logger.debug("All processors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processors: {str(e)}")
            raise

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
            self.execute_stack_btn.clicked.connect(self.execute_stack)
            self.by_site_btn.clicked.connect(self.execute_plot)
            self.med_only_btn.clicked.connect(self.execute_plot)
            self.min_only_btn.clicked.connect(self.execute_plot)
            self.max_only_btn.clicked.connect(self.execute_plot)
            self.save_filtered_btn.clicked.connect(self.save_filtered_result)
            logger.debug("All UI connections established")
        except Exception as e:
            logger.error(f"Failed to setup UI connections: {str(e)}")
            raise

    def reset_button_connection(self, btn):
        """Safely disconnect all signals from a button."""
        logger.debug(f"Resetting connections for button: {btn.objectName()}")
        try:
            btn.clicked.disconnect()
            logger.debug("Button connections reset successfully")
        except TypeError:
            logger.debug("No connections to reset")
            pass

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
        logger.info("Opening directory selection dialog for extraction input")
        selected_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Raw Data Directory",
            r"C:\Users\hhuang10\Downloads\negative_debugging\extracted_data\new",
            QFileDialog.Option.ShowDirsOnly
        )
        if selected_folder:
            logger.debug(f"Selected input folder: {selected_folder}")
            self.uni_config.base_path = selected_folder
            die_folders = [f for f in os.listdir(self.uni_config.base_path)
                          if f.startswith('Die') and
                          os.path.isdir(os.path.join(self.uni_config.base_path, f))]
            self.model.setStringList(die_folders)
            logger.info(f"Found {len(die_folders)} die folders")

    def select_input(self):
        """Select input files for processing."""
        logger.info("Opening file selection dialog")
        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .xlsx File(s)",
            r"C:\Users\hhuang10\Documents\test_data\results",
            "Excel Files (*.xlsx)",
        )
        if selected_files:
            logger.info(f"Selected {len(selected_files)} input files")
            self.uni_config.base_path = selected_files
            self.model.setStringList([os.path.basename(file_path) for file_path in self.uni_config.base_path])
            logger.debug("Input file list updated")

    def select_output(self):
        """Select output directory."""
        logger.info("Opening directory selection dialog for output")
        selected_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            r"C:\Users\hhuang10\Documents\test_data",
            QFileDialog.Option.ShowDirsOnly
        )
        if selected_folder:
            logger.info(f"Selected output folder: {selected_folder}")
            self.uni_config.output_path = selected_folder
            self.output_path_text.setPlainText(self.uni_config.output_path)
            logger.debug("Output path updated")

    def check_input_output(self):
        """
        Validate input and output paths.
        Returns True if valid, False otherwise.
        """
        logger.debug("Checking input/output paths")
        if not self.uni_config.base_path or (not self.debug_mode_box.isChecked() and not self.uni_config.output_path):
            logger.warning("Missing input or output path")
            QMessageBox.warning(self, "No Folder Selected", "Please select both input files and output directory.")
            return False
        logger.debug("Input/output paths validated")
        return True

    def execute_raw_data_extraction(self):
        """Execute raw data extraction process."""
        logger.info("Starting raw data extraction")
        if not self.check_input_output():
            return

        try:
            logger.debug("Configuring extraction parameters")
            # Get configuration from UI
            self.uni_config.debug_flag = self.debug_mode_box.isChecked()
            self.uni_config.prediction_only_flag = self.prediction_only_box.isChecked()

            # Validate input parameters
            logger.debug("Validating input parameters")
            is_valid_low, err_msg_low, pred_range_lower = self.validator_processor.validate_single_number(self.range_low_edit.text())
            is_valid_high, err_msg_high, pred_range_upper = self.validator_processor.validate_single_number(self.range_high_edit.text())
            is_valid_foi, err_msg_foi, interest_freq = self.validator_processor.validate_frequency_list(self.interest_freq_edit.text())

            if is_valid_low and is_valid_high and is_valid_foi:
                # Set validated parameters
                self.uni_config.pred_range_lower = pred_range_lower
                self.uni_config.pred_range_upper = pred_range_upper
                self.uni_config.interest_freq = interest_freq
                logger.info("Parameters validated, starting extraction")
                self.extract_processor.process_all_devices()
                logger.info("Raw data extraction completed successfully")
            else:
                # Show validation errors
                error_messages = []
                if not is_valid_low:
                    error_messages.append(f"Prediction Lower Range: {err_msg_low}")
                if not is_valid_high:
                    error_messages.append(f"Prediction Higher Range: {err_msg_high}")
                if not is_valid_foi:
                    error_messages.append(f"Interest Prediction Frequency: {err_msg_foi}")
                logger.warning(f"Parameter validation failed: {', '.join(error_messages)}")
                QMessageBox.warning(self, "Input Validation Errors", "\n".join(error_messages))

        except Exception as e:
            logger.error(f"Error during raw data extraction: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred during extraction:\n{str(e)}")

    def execute_stack(self):
        """Execute table stacking process."""
        logger.info("Starting table stacking process")
        if not self.uni_config.base_path or not self.uni_config.output_path:
            logger.warning("Missing input or output path")
            QMessageBox.warning(self, "No Folder Selected", "Please select both input files and output directory.")
            return

        try:
            if self.from_selection_box.isChecked():
                logger.debug("Using file name from user input")
                while True:  # Loop until valid input is received
                    text, ok = QInputDialog.getText(self, 'Input File Name', 'Enter the file name:')
                    if not ok:  # User clicked Cancel or closed the dialog
                        logger.debug("User cancelled file name input")
                        return  # Exit the method entirely

                    is_valid, err_msg, save_name = self.validator_processor.validate_path(text)
                    if is_valid:
                        logger.info(f"Starting stacking with save name: {save_name}")
                        self.stack_processor.run_stacking(save_name)
                        logger.info("Table stacking completed successfully")
                        break  # Exit the loop if input is valid
                    else:
                        logger.warning(f"Invalid file name: {err_msg}")
                        QMessageBox.warning(self, "Error File Name Input", err_msg)

        except Exception as e:
            logger.error(f"Error during table stacking: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred during stacking:\n{str(e)}")

    def execute_plot(self, plot_median_flag):
        """Execute plotting process."""
        logger.info("Starting plot generation")
        if not self.check_input_output():
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
                while True:  # Loop until valid input is received
                    text, _ = QInputDialog.getText(self, 'Input File Name', 'Enter the file name:')
                    is_valid, err_msg, save_name = self.validator_processor.validate_path(text)
                    if is_valid:
                        break  # Exit the loop if input is valid
                    else:
                        logger.warning(f"Invalid file name: {err_msg}")
                        QMessageBox.warning(self, "File Name Invalid, Please try another one", err_msg)

            # Check filter threshold and tolerance setting
            if self.filter_outliers_box.isChecked():
                logger.debug("Configuring outlier filtering")
                self.uni_config.filter_outliers_flag = True
                is_valid_threshold, err_msg_threshold, self.uni_config.filter_threshold = self.validator_processor.validate_range(
                    self.filter_threshold_edit.text(), 0.0, 1.0)
                is_valid_tolerance, err_msg_tolerance, self.uni_config.filter_tolerance = self.validator_processor.validate_range(
                    self.filter_tolerance_edit.text(), 0.0, 3.0)

                if not is_valid_threshold or not is_valid_tolerance:
                    error_messages = []
                    if not is_valid_threshold:
                        error_messages.append(f"Invalid Filter Threshold Input: {err_msg_threshold}")
                    if not is_valid_tolerance:
                        error_messages.append(f"Invalid Filter Tolerance Input: {err_msg_tolerance}")
                    logger.warning(f"Invalid filter parameters: {', '.join(error_messages)}")
                    QMessageBox.warning(self, "Input Validation Errors", "\n".join(error_messages))
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
        if not self.check_input_output():
            return
        try:
            self.plot_processor.save_filtered_result()
            logger.info("Filtered results saved successfully")
        except Exception as e:
            logger.error(f"Error saving filtered results: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred when saving filtered result:\n{str(e)}")

    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Excel Tools window closing")
        self.hide()
        self.deleteLater()
        event.accept()

    def __del__(self):
        """Cleanup when object is deleted."""
        logger.info("Excel Tools object being deleted")