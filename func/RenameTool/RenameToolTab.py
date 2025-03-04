"""
Rename Tool Tab module for Excel Tools application.
Provides functionality for file renaming operations with various methods.
"""

import os
import re
import logging, os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QLineEdit, QPushButton, QFileDialog, QTabWidget,
                            QFormLayout, QTextEdit, QListWidget, QSplitter, QMessageBox, QDialog)
from PyQt6.QtCore import Qt
from pathlib import Path
from PyQt6 import uic
from func import ulti

def add_prefix_postfix(folder_path, prefix="", postfix="", file_filter=None):
    """
    Add prefix and/or postfix to all files in a folder.

    Args:
        folder_path (str): Path to the folder containing files
        prefix (str): Prefix to add to filenames
        postfix (str): Postfix to add before file extension
        file_filter (str, optional): Only process files matching this pattern
    """
    results = []
    os.chdir(folder_path)
    results.append(f"Working in: {os.getcwd()}")

    for filename in os.listdir(folder_path):
        if not os.path.isfile(os.path.join(folder_path, filename)):
            continue

        if file_filter and not re.search(file_filter, filename):
            continue

        # Split filename and extension
        name_parts = os.path.splitext(filename)
        base_name = name_parts[0]
        extension = name_parts[1] if len(name_parts) > 1 else ""

        # Create new filename
        new_name = f"{prefix}{base_name}{postfix}{extension}"

        if new_name != filename:
            results.append(f"Renaming: {filename} -> {new_name}")
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))

    return results

def replace_regex(folder_path, pattern, replacement, file_filter=None):
    """
    Replace a string using regex in all filenames in a folder.

    Args:
        folder_path (str): Path to the folder containing files
        pattern (str): Regex pattern to match
        replacement (str): Replacement string (can include regex groups)
        file_filter (str, optional): Only process files matching this pattern
    """
    results = []
    os.chdir(folder_path)
    results.append(f"Working in: {os.getcwd()}")

    # Compile the regex pattern
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return [f"Error in regex pattern: {str(e)}"]

    for filename in os.listdir(folder_path):
        if not os.path.isfile(os.path.join(folder_path, filename)):
            continue

        if file_filter and not re.search(file_filter, filename):
            continue

        if re.search(pattern, filename):
            try:
                new_name = re.sub(pattern, replacement, filename)
                results.append(f"Renaming: {filename} -> {new_name}")
                os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
            except Exception as e:
                results.append(f"Error renaming {filename}: {str(e)}")

    return results

def custom_transform(folder_path, pattern, transform_expr, file_filter=None):
    """
    Apply a custom transformation to parts of filenames matched by regex.

    Args:
        folder_path (str): Path to the folder containing files
        pattern (str): Regex pattern to match (must include capture groups)
        transform_expr (str): Python expression to transform the captured groups
                             Use g1, g2, etc. to refer to capture groups
        file_filter (str, optional): Only process files matching this pattern
    """
    results = []
    os.chdir(folder_path)
    results.append(f"Working in: {os.getcwd()}")

    # Compile the regex pattern
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return [f"Error in regex pattern: {str(e)}"]

    def transform_match(match):
        try:
            # Create a dictionary of capture groups
            groups = {f"g{i+1}": group for i, group in enumerate(match.groups())}

            # Add the full match as g0
            groups["g0"] = match.group(0)

            # Evaluate the transformation expression
            # This uses eval() which can be dangerous, but we're limiting it to filename transformations
            result = eval(transform_expr, {"__builtins__": {}}, groups)

            # Convert result to string
            return str(result)
        except Exception as e:
            # If transformation fails, return the original match
            results.append(f"Error in transformation: {str(e)}")
            return match.group(0)

    for filename in os.listdir(folder_path):
        if not os.path.isfile(os.path.join(folder_path, filename)):
            continue

        if file_filter and not re.search(file_filter, filename):
            continue

        if re.search(pattern, filename):
            try:
                new_name = re.sub(pattern, transform_match, filename)
                results.append(f"Renaming: {filename} -> {new_name}")
                os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
            except Exception as e:
                results.append(f"Error renaming {filename}: {str(e)}")

    return results

# Initialize module logger
logger = logging.getLogger(__name__)

class HelpDialog(QDialog):
    """Dialog for displaying help information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        logger.debug("HelpDialog initialized")

    def setup_ui(self):
        """Load and setup the help UI."""
        ui_path = Path(__file__).parent.parent.parent / 'UI' / 'rename_help.ui'
        uic.loadUi(ui_path, self)
        self.setWindowTitle("Rename Tool Help")
        self.setModal(True)

    def __del__(self):
        """Cleanup when dialog object is deleted."""
        try:
            # Clean up any resources
            if hasattr(self, 'help_text'):
                self.help_text.clear()
            logger.debug("HelpDialog cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during HelpDialog cleanup: {str(e)}")

class RenameToolTab(QWidget):
    """Tab for file renaming operations."""

    def __init__(self, parent=None):
        super().__init__(parent)

        logger.info("Initializing Rename Tools Tab")
        self.setup_ui()
        self.setup_connections()
        logger.info("Rename Tools initialization complete")

    def setup_ui(self):
        """Load and setup the UI."""
        ui_path = Path(__file__).parent.parent.parent / 'UI' / 'rename_tool_tab.ui'
        logger.info(f"Loading UI from: {ui_path}")
        uic.loadUi(ui_path, self)

    def setup_connections(self):
        """Setup UI element connections and initial states."""
        logger.debug("Setting up UI connections")
        try:
            # Connect signals
            self.folder_edit.textChanged.connect(self.update_file_list)
            self.browse_button.clicked.connect(self.browse_folder)
            self.filter_edit.textChanged.connect(self.update_file_list)

            # Connect operation buttons
            self.prefix_button.clicked.connect(self.run_prefix_postfix)
            self.regex_button.clicked.connect(self.run_regex_replace)
            self.custom_button.clicked.connect(self.run_custom_transform)
            self.help_button.clicked.connect(self.show_help)

            logger.debug("All UI connections established")
        except Exception as e:
            logger.error(f"Failed to setup UI connections: {str(e)}")
            raise

    def browse_folder(self):
        """Open folder browser dialog."""
        logger.debug("Opening folder browser dialog")
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_edit.setText(folder)
            self.update_file_list()

    def update_file_list(self):
        """Update the list of files based on current folder and filter."""
        logger.debug("Updating file list")
        self.file_list.clear()
        folder = self.folder_edit.text()
        if not folder or not os.path.isdir(folder):
            return

        file_filter = self.filter_edit.text() if self.filter_edit.text() else None

        try:
            files = []
            for filename in os.listdir(folder):
                full_path = os.path.join(folder, filename)
                if os.path.isfile(full_path):
                    if file_filter and not re.search(file_filter, filename):
                        continue
                    files.append(filename)

            # Sort files alphabetically
            files.sort()
            # Add items directly to QListWidget
            self.file_list.addItems(files)
        except Exception as e:
            logger.error(f"Error updating file list: {str(e)}")

    def check_folder_path(self) -> bool:
        """
        Validate folder path exists.

        Returns:
            bool: True if path is valid, False otherwise
        """
        logger.debug("Checking folder path")
        folder = self.folder_edit.text()
        if not folder or not os.path.isdir(folder):
            logger.warning("Missing or invalid folder path")
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder first.")
            return False
        return True

    def run_prefix_postfix(self):
        """Execute prefix/postfix renaming operation."""
        logger.info("Starting prefix/postfix operation")
        if not self.check_folder_path():
            return

        try:
            folder = self.folder_edit.text()
            prefix = self.prefix_edit.text()
            postfix = self.postfix_edit.text()
            file_filter = self.filter_edit.text() if self.filter_edit.text() else None

            if not prefix and not postfix:
                logger.warning("No prefix or postfix provided")
                return

            results = add_prefix_postfix(folder, prefix, postfix, file_filter)
            self.update_file_list()
            logger.info("Prefix/postfix operation completed")
        except Exception as e:
            logger.error(f"Error in prefix/postfix operation: {str(e)}")

    def run_regex_replace(self):
        """Execute regex replacement operation."""
        logger.info("Starting regex replacement operation")
        if not self.check_folder_path():
            return

        try:
            folder = self.folder_edit.text()
            pattern = self.pattern_edit.text()
            replacement = self.replacement_edit.text()
            file_filter = self.filter_edit.text() if self.filter_edit.text() else None

            if not pattern:
                logger.warning("No pattern provided")
                return

            # Escape pattern if it doesn't contain any regex special characters
            if not any(c in pattern for c in r'.^$*+?{}[]\|()'):
                pattern = re.escape(pattern)

            results = replace_regex(folder, pattern, replacement, file_filter)
            self.update_file_list()
            logger.info("Regex replacement operation completed")
        except Exception as e:
            logger.error(f"Error in regex replacement: {str(e)}")

    def run_custom_transform(self):
        """Execute custom transform operation."""
        logger.info("Starting custom transform operation")
        if not self.check_folder_path():
            return

        try:
            folder = self.folder_edit.text()
            pattern = self.pattern_edit.text()
            transform_expr = self.transform_edit.text()
            file_filter = self.filter_edit.text() if self.filter_edit.text() else None

            if not pattern:
                logger.warning("No pattern provided")
                return

            if not transform_expr:
                logger.warning("No transform expression provided")
                return

            results = custom_transform(folder, pattern, transform_expr, file_filter)
            self.update_file_list()
            logger.info("Custom transform operation completed")
        except Exception as e:
            logger.error(f"Error in custom transform: {str(e)}")

    def show_help(self):
        """Show the help dialog."""
        logger.debug("Showing help dialog")
        try:
            help_dialog = HelpDialog(self)
            help_dialog.exec()
        except Exception as e:
            logger.error(f"Error showing help dialog: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error showing help:\n{str(e)}")

    def closeEvent(self, event):
        """Handle tab close event - cleanup handled by parent."""
        logger.debug("RenameToolTab close event triggered")
        event.accept()

    def __del__(self):
        """Cleanup when object is deleted."""
        try:
            # Clean up configuration
            self.config = None

            logger.info("Rename Tools object cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during object cleanup: {str(e)}")

# def main():
#     app = QApplication(sys.argv)
#     window = FileRenamerApp()
#     window.show()
#     sys.exit(app.exec())

# if __name__ == "__main__":
#     main()

# Example usage:
# 1. Add prefix "TEST_" to all files:
#    python rename_file.py prefix "C:\path\to\folder" --prefix "TEST_"
#
# 2. Add postfix "_BACKUP" to all .txt files:
#    python rename_file.py prefix "C:\path\to\folder" --postfix "_BACKUP" --filter "\.txt$"
#
# 3. Replace "old" with "new" in all filenames:
#    python rename_file.py exact "C:\path\to\folder" "old" "new"
#
# 4. Replace numbers in parentheses with incremented value using regex:
#    python rename_file.py regex "C:\path\to\folder" "\((\d+)\)" "($1_new)"