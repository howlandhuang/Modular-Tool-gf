# Modular Tools - Semiconductor Analysis Suite

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-GPLv3-green)

A comprehensive GUI application for semiconductor data processing, analysis, and visualization with modular architecture.

## TODO/Future Work :-)

### Unit Testing Framework
- Create comprehensive unit tests for PyQt6 GUI components:
  - Implement test fixtures for main application windows
  - Add widget interaction tests (button clicks, raw data selection)
  - Develop mock objects for file system operations
  - Set up CI pipeline for automated test execution
- Test coverage targets:
  - Core processing algorithms (priority)
  - UI event handlers and signals
  - File I/O operations
  - Error handling and edge cases
- Tools/frameworks to consider:
  - pytest-qt for PyQt testing
  - unittest.mock for mocking dependencies
  - GitHub Actions for CI/CD integration

### Documentation
- Improve in-code documentation
- Create user manual with examples

## 🚀 Key Features

### 📊 CSV Processing Tool
- Multi-format support:
  - 8-inch/12-inch wafer data
  - Cascade/TEL/Tohuku/Vtgm measurement systems
  - Automated metadata extraction
- Advanced analysis:
  - Linear regression calculations
  - Outlier detection & filtering
  - Temperature-dependent organization
  - Custom column transformations

### 📈 Noise Analysis Tool
- Comprehensive visualization:
  - Site-specific plots
  - Median/Min/Max trends
  - Frequency-domain analysis
- Advanced processing:
  - Noise prediction models
  - Statistical filtering
  - Multi-device comparison
  - Excel report generation

### 🔄 File Management Tool
- Batch renaming with regex
- Context-aware transformations
  - Coordinate standardization
  - Timestamp normalization
  - Metadata preservation
- Preview system with undo/redo

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- pip package manager

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/Modular-Tool-gf.git
cd Modular-Tool-gf

# Create virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/MacOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🖥️ Usage

### Launch Application
```bash
python main.py [--debug] [--log-level LEVEL]
```
**Options:**
- `--debug`: Enable debug mode
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### Application Workflow
1. Select input files/folders through GUI
2. Configure processing parameters:
   - Filter thresholds
   - Visualization preferences
   - Output formatting
3. Preview transformations
4. Execute processing
5. Review generated reports/logs

## 🛠 Project Structure

```
modular-tools/
├── main.py              # Main application entry point
├── requirements.txt     # Project dependencies
├── LICENSE             # GPLv3 License
├── UI/                 # UI files
│   ├── modular_tools.ui
│   ├── csv_tool_tab.ui
│   ├── noise_tool_tab.ui
│   └── rename_tool_tab.ui
├── func/               # Functional modules
│   ├── CsvTool/       # CSV processing tools
│   ├── NoiseTool/     # Noise analysis tools
│   └── RenameTool/    # File renaming tools
└── logs/              # Application logs
```

## Development

### Code Style

This project follows PEP 8 guidelines and uses type hints for better code clarity.

### Logging

The application uses Python's logging module for comprehensive error tracking and debugging.

### Error Handling

Robust error handling is implemented throughout the application with user-friendly error messages.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Huang, Haoyang
- Email: howlandhuang@gmail.com

## Acknowledgments

- PyQt6 for the GUI framework
- Matplotlib for visualization capabilities
- Pandas for data processing
- NumPy for numerical computations
- XlsxWriter for Excel file handling