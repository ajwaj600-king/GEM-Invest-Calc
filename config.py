# Konfiguracja ETF-ów i parametrów strategii

UK_ETFS = {
    'EIMI.L': 'iShares Core MSCI Emerging Markets',
    'IWDA.L': 'iShares Core MSCI World',
    'CBU0.L': 'iShares Core € Corp Bond',
    'IB01.L': 'iShares Core € Gov Bond 1-3yr',
    'CNDX.L': 'iShares Core S&P 500'
}

US_REFERENCE_ETFS = {
    'SPY': 'SPDR S&P 500 ETF',
    'VEA': 'Vanguard FTSE Developed Markets',
    'BND': 'Vanguard Total Bond Market',
    'QQQ': 'Invesco QQQ Trust',
    'VTI': 'Vanguard Total Stock Market'
}

# NOWE BENCHMARKI
BENCHMARK_INDICES = {
    '^GSPC': 'S&P 500 Index',
    'WIG20.WA': 'WIG20 Index',
    '^VIX': 'CBOE Volatility Index',
    '^TNX': '10-Year Treasury Yield',
    'GC=F': 'Gold Futures',
    'EURUSD=X': 'EUR/USD'
}

# POPULARNE ETF-y DO WYBORU
POPULAR_ETFS = {
    # US Equity
    'VTI': 'Vanguard Total Stock Market ETF',
    'VOO': 'Vanguard S&P 500 ETF',
    'VEA': 'Vanguard FTSE Developed Markets ETF',
    'VWO': 'Vanguard FTSE Emerging Markets ETF',
    'QQQ': 'Invesco QQQ Trust ETF',
    'IWM': 'iShares Russell 2000 ETF',

    # EU ETFs
    'VXUS': 'Vanguard Total International Stock ETF',
    'VGK': 'Vanguard FTSE Europe ETF',
    'VPL': 'Vanguard FTSE Pacific ETF',

    # Bonds
    'BND': 'Vanguard Total Bond Market ETF',
    'VXUS': 'Vanguard Total International Bond ETF',
    'TLT': 'iShares 20+ Year Treasury Bond ETF',

    # Sector ETFs
    'XLK': 'Technology Select Sector SPDR Fund',
    'XLF': 'Financial Select Sector SPDR Fund',
    'XLE': 'Energy Select Sector SPDR Fund',
    'XLV': 'Health Care Select Sector SPDR Fund',

    # Commodities
    'GLD': 'SPDR Gold Shares',
    'SLV': 'iShares Silver Trust',
    'DBC': 'Invesco DB Commodity Index Tracking Fund',

    # REITs
    'VNQ': 'Vanguard Real Estate Index Fund ETF',
    'SCHH': 'Schwab US REIT ETF'
}

# Parametry strategii
LOOKBACK_PERIOD = 252
REBALANCE_FREQUENCY = 'M'
RISK_FREE_RATE = 0.02

# INVESTMENT PLANNING PARAMETERS
INVESTMENT_SCENARIOS = {
    'conservative': {
        'expected_return': 0.06,
        'volatility': 0.10,
        'max_drawdown': 0.15,
        'description': 'Conservative growth with capital preservation focus'
    },
    'moderate': {
        'expected_return': 0.08,
        'volatility': 0.15,
        'max_drawdown': 0.25,
        'description': 'Balanced growth with moderate risk'
    },
    'aggressive': {
        'expected_return': 0.12,
        'volatility': 0.20,
        'max_drawdown': 0.35,
        'description': 'Maximum growth with high risk tolerance'
    }
}

# Kolory brandowe
BRAND_COLORS = {
    'primary': '#1E88E5',
    'secondary': '#FFC107',
    'success': '#4CAF50',
    'danger': '#F44336',
    'warning': '#FF9800',
    'info': '#00BCD4',
    'dark': '#212121',
    'light': '#F5F5F5',
    'purple': '#9C27B0',
    'teal': '#009688',
    'indigo': '#3F51B5'
}

# Style CSS
CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }

    .metric-positive {
        border-left-color: #4CAF50 !important;
    }

    .metric-negative {
        border-left-color: #F44336 !important;
    }

    .strategy-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }

    .investment-plan-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }

    .performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }

    .kpi-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-top: 3px solid #1E88E5;
    }

    .etf-selector {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
    }

    .sidebar-style {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }

    .investment-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }

    .scenario-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #1E88E5;
    }
</style>
"""