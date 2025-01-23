from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem, QInputDialog
from PyQt6.QtCore import QStringListModel
import pandas as pd
from pathlib import Path
from func import noise_plot_v2 as noise_plot
from func import stacked_table
from func import extract_noise_v3 as extract_noise
from func import ulti
import os
import logging
from multiprocessing import freeze_support
logger = logging.getLogger(__name__)
freeze_support()

class ExcelTools(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        ui_path = Path(__file__).parent.parent / 'UI' / 'exceltools.ui'
        uic.loadUi(ui_path, self)
        self.setWindowTitle("Excel Tools")
        self.sheets = None
        self.current_excel = None
        self.current_df = None
        self.setup_connections()
        self.initialize_processors()
        self.stack_widget.setCurrentIndex(0)
        logger.info("Excel Tools Initializing Complete")

    def initialize_processors(self):
        self.uni_config = ulti.ProcessingConfig(
            base_path = None,
            output_path = None,
            pred_range_lower = None,
            pred_range_upper = None,
            interest_freq = None,
            debug_flag = False,
            filter_outliers_flag = False,
            filter_threshold = 0.1,
            filter_tolerance = 1.0,
            prediction_only_flag = False
        )
        self.validator_processor = ulti.InputValidator()
        self.extract_processor = extract_noise.DataProcessor(self.uni_config)
        self.stack_processor = stacked_table.StackProcessor(self.uni_config)
        self.plot_processor = noise_plot.PlotProcessor(self.uni_config)

    def setup_connections(self):
        self.model = QStringListModel()
        self.input_files_listview.setModel(self.model)
        self.output_path_text.setReadOnly(True)

        self.input_selection_btn.setEnabled(False)
        self.output_selection_btn.setEnabled(False)

        self.data_extraction_tab_btn.clicked.connect(self.switch_data_extraction_tab)
        self.stack_table_tab_btn.clicked.connect(self.switch_stack_table_tab)
        self.plot_tab_btn.clicked.connect(self.switch_plot_tab)

        self.extract_btn.clicked.connect(self.execute_raw_data_extraction)
        self.execute_stack_btn.clicked.connect(self.execute_stack)
        self.by_site_btn.clicked.connect(self.execute_plot_by_site)
        self.med_only_btn.clicked.connect(self.execute_plot_med_only)
        self.save_filtered_btn.clicked.connect(self.save_filtered_result)

    def reset_button_connection(self, btn):
        try:
            btn.clicked.disconnect()
        except TypeError:
            pass

    def switch_data_extraction_tab(self):
        try:
            self.stack_widget.setCurrentIndex(1)
            self.input_selection_btn.setEnabled(True)
            self.output_selection_btn.setEnabled(True)
            self.reset_button_connection(self.input_selection_btn)
            self.reset_button_connection(self.output_selection_btn)
            self.input_selection_btn.clicked.connect(self.select_extraction_input)
            self.output_selection_btn.clicked.connect(self.select_output)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error switching tab:\n{str(e)}")

    def switch_stack_table_tab(self):
        try:
            self.stack_widget.setCurrentIndex(2)
            self.input_selection_btn.setEnabled(True)
            self.output_selection_btn.setEnabled(True)
            self.reset_button_connection(self.input_selection_btn)
            self.reset_button_connection(self.output_selection_btn)
            self.input_selection_btn.clicked.connect(self.select_input)
            self.output_selection_btn.clicked.connect(self.select_output)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error switching tab:\n{str(e)}")

    def switch_plot_tab(self):
        try:
            self.stack_widget.setCurrentIndex(3)
            self.input_selection_btn.setEnabled(True)
            self.output_selection_btn.setEnabled(True)
            self.reset_button_connection(self.input_selection_btn)
            self.reset_button_connection(self.output_selection_btn)
            self.input_selection_btn.clicked.connect(self.select_input)
            self.output_selection_btn.clicked.connect(self.select_output)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error switching tab:\n{str(e)}")

    def open_excel_file(self):
        """Open Excel file dialog and load worksheets"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Excel File",
                "",
                "Excel Files (*.xlsx *.xls)",
            )

            if file_path:
                # Load Excel file
                self.current_excel = pd.ExcelFile(file_path)

                # Update worksheet list
                self.worksheet_list.clear()
                self.worksheet_list.addItems(self.current_excel.sheet_names)

                # Enable buttons
                self.worksheet_list.setEnabled(True)
                self.load_ws_button.setEnabled(True)


        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error opening Excel file:\n{str(e)}")

    def on_worksheet_changed(self, index):
        """Handle worksheet selection change"""
        if index >= 1:
            self.load_ws_button.setEnabled(True)
            self.display_ws_button.setEnabled(True)

    def load_worksheet(self):
        """Load selected worksheet into memory"""
        try:
            if self.current_excel and self.worksheet_list.currentText():
                sheet_name = self.worksheet_list.currentText()
                self.current_df = pd.read_excel(self.current_excel, sheet_name=sheet_name)
                QMessageBox.information(self, "Success", f"Worksheet '{sheet_name}' loaded successfully!")

                # Enable buttons
                self.display_ws_button.setEnabled(False)
                self.plot_button.setEnabled(False)
                self.by_bias_button.setEnabled(True)
                self.by_site_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading worksheet:\n{str(e)}")

    def display_worksheet(self):
        """Display current worksheet in table widget"""
        try:
            if self.current_df is not None:
                # Clear existing items
                self.excel_table_widget.clear()

                # Set table dimensions
                rows, cols = self.current_df.shape
                self.excel_table_widget.setRowCount(30)
                self.excel_table_widget.setColumnCount(cols)

                # Set headers
                self.excel_table_widget.setHorizontalHeaderLabels(self.current_df.columns)

                # Populate table
                for row in range(rows):
                    for col in range(cols):
                        value = str(self.current_df.iloc[row, col])
                        item = QTableWidgetItem(value)
                        self.excel_table_widget.setItem(row, col, item)

                # Adjust column widths
                self.excel_table_widget.resizeColumnsToContents()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error displaying worksheet:\n{str(e)}")

    def select_extraction_input(self):
        selected_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Raw Data Directory",
            r"C:\Users\hhuang10\Downloads\negative_debugging\extracted_data\new",
            QFileDialog.Option.ShowDirsOnly
        )
        if selected_folder:
            self.uni_config.base_path = selected_folder
            self.model.setStringList([f for f in os.listdir(self.uni_config.base_path)
                          if f.startswith('Die') and
                          os.path.isdir(os.path.join(self.uni_config.base_path, f))])

    def select_input(self):
        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .xlsx File(s)",
            r"C:\Users\hhuang10\Documents\test_data\results",
            "Excel Files (*.xlsx)",
        )
        if selected_files:
            logger.info(f"Selected input files: {selected_files}")
            self.uni_config.base_path = selected_files
            self.model.setStringList([os.path.basename(file_path) for file_path in self.uni_config.base_path])

    def select_output(self):
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

    def check_input_output(self):
        if not self.uni_config.base_path or (not self.debug_mode_box.isChecked() and not self.uni_config.output_path):
            QMessageBox.warning(self, "No Folder Selected", "Please select both input files and output directory.")
            return False
        return True

    def execute_raw_data_extraction(self):

        if not self.uni_config.base_path or not self.uni_config.output_path:
            QMessageBox.warning(self, "No Folder Selected", "Please select both input files and output directory.")
            return

        try:
            logger.info("Button clicked, start extracting raw data")
            self.uni_config.debug_flag = True if self.debug_mode_box.isChecked() else False
            self.uni_config.prediction_only_flag = True if self.prediction_only_box.isChecked() else False
            is_valid_low, err_msg_low, pred_range_lower = self.validator_processor.validate_single_number(self.range_low_edit.text())
            is_valid_high, err_msg_high, pred_range_upper = self.validator_processor.validate_single_number(self.range_high_edit.text())
            is_valid_foi, err_msg_foi, interest_freq = self.validator_processor.validate_frequency_list(self.interest_freq_edit.text())

            if is_valid_low and is_valid_high and is_valid_foi:
                self.uni_config.pred_range_lower = pred_range_lower
                self.uni_config.pred_range_upper = pred_range_upper
                self.uni_config.interest_freq = interest_freq
                logger.info("Parameters are valid, start processing")
                self.extract_processor.process_all_devices()
                # extract_noise.main(self.uni_config)
            else:
                error_messages = []
                if not is_valid_low:
                    error_messages.append(f"Prediction Lower Range: {err_msg_low}")
                if not is_valid_high:
                    error_messages.append(f"Prediction Higher Range: {err_msg_high}")
                if not is_valid_foi:
                    error_messages.append(f"Interest Prediction Frequency: {err_msg_foi}")
                QMessageBox.warning(self, "Input Validation Errors", "\n".join(error_messages))

            print("Processing done")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred when stack table: {str(e)}")

    def execute_stack(self):

        if not self.uni_config.base_path or not self.uni_config.output_path:
            QMessageBox.warning(self, "No Folder Selected", "Please select both input files and output directory.")
            return

        try:
            if self.from_selection_box.isChecked():
                text, _ = QInputDialog.getText(self, 'Input File Name', 'Enter the file name:')
                is_valid, err_msg, save_name = self.validator_processor.validate_path(text)
                if is_valid:
                    self.stack_processor.from_selection(save_name)
                else:
                    QMessageBox.warning(self, "Error File Name Input", err_msg)
                    return
            print("Processing done")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred when stack table: {str(e)}")

    def execute_plot_by_site(self):
        # If no input/output folder selected, return early
        if not self.check_input_output():
            return

        try:
            self.uni_config.debug_flag = True if self.debug_mode_box.isChecked() else False

            # Check if at least one plot type is selected
            plot_type = []
            if self.sid_box.isChecked():
                plot_type.append('Sid')
            if self.sid_id2_box.isChecked():
                plot_type.append('Sid/id^2')
            if self.svg_box.isChecked():
                plot_type.append('Svg')
            if self.sid_f_box.isChecked():
                plot_type.append('Sid*f')
            if plot_type == []:
                QMessageBox.warning(self, "No Plot Type Selected", "Please select at least one plot type.")
                return

            # Validate filter threshold and tolerance
            if self.filter_outliers_box.isChecked():
                self.uni_config.filter_outliers_flag = True
                is_valid_threshold, err_msg_threshold, self.uni_config.filter_threshold = self.validator_processor.validate_range(self.filter_threshold_edit.text(), 0.0, 1.0)
                is_valid_tolerance, err_msg_tolerance, self.uni_config.filter_tolerance = self.validator_processor.validate_range(self.filter_tolerance_edit.text(), 0.0, 3.0)
                if not is_valid_threshold or not is_valid_tolerance:
                    error_messages = []
                    if not is_valid_threshold:
                        error_messages.append(f"Invalid Filter Threshold Input: {err_msg_threshold}")
                    if not is_valid_tolerance:
                        error_messages.append(f"Invalid Filter Tolerance Input: {err_msg_tolerance}")
                    QMessageBox.warning(self, "Input Validation Errors", "\n".join(error_messages))
                    return
            else:
                self.uni_config.filter_outliers_flag = False

            # Check debug mode and save file name
            if self.debug_mode_box.isChecked():
                self.plot_processor.run_by_site(plot_type, 'DEBUGGING')
            else:
                text, _ = QInputDialog.getText(self, 'Input File Name', 'Enter the file name:')
                is_valid, err_msg, save_name = self.validator_processor.validate_path(text)
                if is_valid:
                    self.plot_processor.run_by_site(plot_type, save_name)
                else:
                    QMessageBox.warning(self, "Error File Name Input", err_msg)
                    return

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred when plot by site: {str(e)}")

    def execute_plot_med_only(self):
        # If no input/output folder selected, return early
        if not self.check_input_output():
            return

        try:
            self.uni_config.debug_flag = True if self.debug_mode_box.isChecked() else False

            # Check if at least one plot type is selected
            plot_type = []
            if self.sid_box.isChecked():
                plot_type.append('Sid')
            if self.sid_id2_box.isChecked():
                plot_type.append('Sid/id^2')
            if self.svg_box.isChecked():
                plot_type.append('Svg')
            if self.sid_f_box.isChecked():
                plot_type.append('Sid*f')
            if plot_type == []:
                QMessageBox.warning(self, "No Plot Type Selected", "Please select at least one plot type.")
                return

            # Check debug mode and save file name
            if self.debug_mode_box.isChecked():
                self.plot_processor.run_med_only(plot_type, 'DEBUGGING')
            else:
                text, _ = QInputDialog.getText(self, 'Input File Name', 'Enter the file name:')
                is_valid, err_msg, save_name = self.validator_processor.validate_path(text)
                if is_valid:
                    self.plot_processor.run_med_only(plot_type, save_name)
                else:
                    QMessageBox.warning(self, "No File Name Input", err_msg)
                    return

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred when plot median only: {str(e)}")

    def save_filtered_result(self):
        # If no input/output folder selected, return early
        if not self.check_input_output():
            return
        try:
            self.plot_processor.save_filtered_result()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred when saving filtered result: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event (X button click)"""
        logger.info("Excel Tools is closing via X button")
        # Hide the window first
        self.hide()
        # Mark it for deletion when we return to event loop
        self.deleteLater()
        event.accept()

    def __del__(self):
        """Destructor"""
        logger.info("Excel Tools object is being deleted")