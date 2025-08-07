#Global Equities Momentum Dashboard and calculator
#pip install yfinance pandas numpy matplotlib plotly dash streamlit scikit-learn seaborn
#HOW TO USE
#
# Open Browser
#RUN: streamlit run main.py

GEM_Calculator/
├── main.py              # Główna aplikacja web
├── data_fetcher.py      # Pobieranie danych z Yahoo
├── gem_strategy.py      # Logika strategii GEM
├── backtester.py        # Backtesting
├── visualizations.py    # Wykresy i wizualizacje
└── config.py           # Konfiguracja (ETF-y, parametry)
