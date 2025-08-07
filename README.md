# GEM STRATEGIA - PROFESJONALNY DASHBOARD

## Overview

**GEM-Invest-Calc** is a professional dashboard and calculator for implementing the Global Equities Momentum (GEM) strategy. This tool is designed to help users analyze and apply momentum-based investment strategies using Python.

## Features

- Interactive dashboard for GEM strategy analysis
- Portfolio visualization and calculation tools
- Customizable input for different asset classes and timeframes
- Easy-to-use interface for both beginners and professionals

## Installation

1. **Clone the repository:**
   ```bash
git clone https://github.com/ajwaj600-king/GEM-Invest-Calc.git
cd GEM-Invest-Calc
   ```
2. **Install dependencies:**
   ```bash
pip install -r requirements.txt
   ```

## Usage

Run the main Python script to start the dashboard:
```bash
streamlit run main.py
```
Follow the on-screen instructions to input your portfolio parameters and view analytical results.

## File Structure

- `main.py` – Entry point for the dashboard application
- `data_fetcher.py` – Fetches data from Yahoo Finance
- `gem_strategy.py` – Core logic of the GEM strategy
- `backtester.py` – Backtesting and performance evaluation
- `visualizations.py` – Charting and visualization utilities
- `config.py` – Configuration (ETFs, parameters, etc.)

## Example

![Dashboard Example](dashboard_example.png)
*Sample dashboard output*


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or feature suggestions.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
