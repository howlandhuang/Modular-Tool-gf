# Modular Tools

A comprehensive data processing and visualization tool for semiconductor testing and analysis.

## Overview

Modular Tools is a Python-based application that provides a user-friendly interface for processing, analyzing, and visualizing semiconductor test data. The application features a modular architecture with separate tools for different data processing needs.

### Features

- **CSV Tool**: Process and analyze CSV data from various measurement systems
  - Support for multiple data formats (8-inch, 12-inch, Cascade, TEL, Tohuku, Vtgm)
  - Automated data extraction and processing
  - Temperature-based data organization
  - Custom column extraction
  - Statistical analysis capabilities

- **Noise Tool**: Advanced noise analysis and visualization
  - Multiple visualization options
  - Statistical processing
  - Prediction functionality
  - Customizable data filtering
  - Site-specific analysis

- **Rename Tool**: Batch file management
  - Batch file renaming capabilities
  - Regular expression support
  - Custom transformation rules
  - File filtering options
  - Preview functionality

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- PyQt6
- Matplotlib
- Pandas
- NumPy
- XlsxWriter

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/modular-tools.git
cd modular-tools
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Select the appropriate tool tab based on your needs:
   - CSV Tool for data processing
   - Noise Tool for noise analysis
   - Rename Tool for file management

3. Follow the on-screen instructions for each tool.

## Project Structure

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