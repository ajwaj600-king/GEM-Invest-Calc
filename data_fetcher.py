import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

from config import UK_ETFS, US_REFERENCE_ETFS, BENCHMARK_INDICES


class AdvancedDataFetcher:
    def __init__(self):
        self.all_etfs = {**UK_ETFS, **US_REFERENCE_ETFS}
        self.benchmarks = BENCHMARK_INDICES
        self.cache = {}

    def fetch_comprehensive_data(self, symbols, period='5y', interval='1d'):
        """
        Pobiera kompletne dane: ceny, wolumeny, wskaźniki techniczne
        """
        try:
            # Format symbols
            formatted_symbols = self._format_symbols(symbols)

            # Download data with additional metrics
            data = yf.download(formatted_symbols, period=period, interval=interval, auto_adjust=True, group_by='ticker')

            if len(formatted_symbols) == 1:
                return self._process_single_symbol_data(data, formatted_symbols[0])
            else:
                return self._process_multi_symbol_data(data, formatted_symbols)

        except Exception as e:
            print(f"Error fetching comprehensive data: {e}")
            return None

    def fetch_benchmark_data(self, period='5y'):
        """
        Pobiera dane benchmarków (S&P500, WIG20, etc.)
        """
        try:
            benchmark_symbols = list(self.benchmarks.keys())
            data = yf.download(benchmark_symbols, period=period, auto_adjust=True)

            if 'Close' in data.columns:
                return data['Close']
            else:
                return data

        except Exception as e:
            print(f"Error fetching benchmark data: {e}")
            return None

    def calculate_advanced_metrics(self, price_data, period_days=252):
        """
        Oblicza zaawansowane metryki: Sharpe, Sortino, Calmar, etc.
        """
        if price_data is None or price_data.empty:
            return None

        metrics = {}

        for column in price_data.columns:
            try:
                series = price_data[column].dropna()
                if len(series) < period_days:
                    continue

                # Podstawowe zwroty
                returns = series.pct_change().dropna()
                recent_data = series.tail(period_days + 1)
                total_return = (recent_data.iloc[-1] / recent_data.iloc[0]) - 1

                # Zaawansowane metryki
                annual_return = (1 + total_return) ** (252 / period_days) - 1
                volatility = returns.std() * np.sqrt(252)

                # Sharpe Ratio
                excess_returns = returns - 0.02 / 252  # Daily risk-free rate
                sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

                # Sortino Ratio
                downside_returns = returns[returns < 0]
                sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(
                    downside_returns) > 0 and downside_returns.std() > 0 else 0

                # Maximum Drawdown
                rolling_max = series.expanding().max()
                drawdowns = (series - rolling_max) / rolling_max
                max_drawdown = drawdowns.min()

                # Calmar Ratio
                calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

                # VaR (Value at Risk) 95%
                var_95 = np.percentile(returns, 5)

                # Skewness and Kurtosis
                skewness = returns.skew()
                kurtosis = returns.kurtosis()

                # Win Rate
                positive_returns = returns[returns > 0]
                win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0

                metrics[column] = {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'sortino_ratio': sortino,
                    'max_drawdown': max_drawdown,
                    'calmar_ratio': calmar,
                    'var_95': var_95,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'win_rate': win_rate,
                    'current_price': series.iloc[-1],
                    'momentum_score': total_return - 0.02
                }

            except Exception as e:
                print(f"Error calculating metrics for {column}: {e}")
                continue

        return metrics

    def get_correlation_matrix(self, symbols, period='2y'):
        """
        Oblicza macierz korelacji między ETF-ami i benchmarkami
        """
        try:
            # Pobierz dane ETF-ów
            etf_data = self.fetch_comprehensive_data(symbols, period)
            benchmark_data = self.fetch_benchmark_data(period)

            if etf_data is None or benchmark_data is None:
                return None

            # Łącz dane
            if hasattr(etf_data, 'columns'):
                etf_returns = etf_data.pct_change().dropna()
            else:
                etf_returns = pd.DataFrame({symbols[0]: etf_data.pct_change().dropna()})

            benchmark_returns = benchmark_data.pct_change().dropna()

            # Wyrównaj indeksy
            common_dates = etf_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) == 0:
                return None

            etf_aligned = etf_returns.loc[common_dates]
            benchmark_aligned = benchmark_returns.loc[common_dates]

            # Połącz w jedną macierz
            combined_data = pd.concat([etf_aligned, benchmark_aligned], axis=1)
            correlation_matrix = combined_data.corr()

            return correlation_matrix

        except Exception as e:
            print(f"Error calculating correlation matrix: {e}")
            return None

    def get_sector_allocation(self, symbols):
        """
        Pobiera informacje o alokacji sektorowej ETF-ów
        """
        sector_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(self._format_single_symbol(symbol))
                info = ticker.info

                sector_data[symbol] = {
                    'name': info.get('longName', symbol),
                    'category': info.get('category', 'N/A'),
                    'total_assets': info.get('totalAssets', 0),
                    'expense_ratio': info.get('annualReportExpenseRatio', 0),
                    'yield': info.get('yield', 0),
                    'inception_date': info.get('fundInceptionDate', 'N/A')
                }

            except Exception as e:
                print(f"Error fetching sector data for {symbol}: {e}")
                sector_data[symbol] = {'name': symbol, 'category': 'Unknown'}

        return sector_data

    def _format_symbols(self, symbols):
        """Format symbols for Yahoo Finance"""
        formatted = []
        for symbol in symbols:
            if symbol in UK_ETFS and not symbol.endswith('.L'):
                formatted.append(f"{symbol}.L")
            else:
                formatted.append(symbol)
        return formatted

    def _format_single_symbol(self, symbol):
        """Format single symbol"""
        if symbol in UK_ETFS and not symbol.endswith('.L'):
            return f"{symbol}.L"
        return symbol

    def _process_single_symbol_data(self, data, symbol):
        """Process data for single symbol"""
        return pd.DataFrame({'Close': data['Close']})

    def _process_multi_symbol_data(self, data, symbols):
        """Process data for multiple symbols"""
        if 'Close' in data.columns:
            return data['Close']
        else:
            # Handle grouped data
            close_data = {}
            for symbol in symbols:
                if (symbol, 'Close') in data.columns:
                    close_data[symbol] = data[symbol]['Close']
                elif symbol in data.columns:
                    close_data[symbol] = data[symbol]
            return pd.DataFrame(close_data)


# Test the advanced fetcher
if __name__ == "__main__":
    fetcher = AdvancedDataFetcher()

    # Test comprehensive data
    symbols = ['EIMI', 'IWDA', 'CNDX']
    print("Testing advanced data fetcher...")

    data = fetcher.fetch_comprehensive_data(symbols, period='1y')
    if data is not None:
        print("✓ Comprehensive data fetched")

        metrics = fetcher.calculate_advanced_metrics(data)
        if metrics:
            print("✓ Advanced metrics calculated")
            for symbol, metric in metrics.items():
                print(f"{symbol}: Sharpe={metric['sharpe_ratio']:.2f}, Return={metric['annual_return']:.2%}")

    # Test benchmarks
    benchmark_data = fetcher.fetch_benchmark_data('1y')
    if benchmark_data is not None:
        print("✓ Benchmark data fetched")
        print(f"Benchmarks available: {list(benchmark_data.columns)}")