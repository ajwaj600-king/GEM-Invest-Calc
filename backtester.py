import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_fetcher import AdvancedDataFetcher
from gem_strategy import GEMStrategy
from config import UK_ETFS, US_REFERENCE_ETFS, LOOKBACK_PERIOD


class GEMBacktester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.fetcher = AdvancedDataFetcher()
        self.strategy = GEMStrategy()

    def run_backtest(self, symbols, start_date='2019-01-01', end_date=None, rebalance_freq='M'):
        """
        Wykonuje backtest strategii GEM z pełną obsługą błędów
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            print(f"Starting backtest for symbols: {symbols}")
            print(f"Date range: {start_date} to {end_date}")

            # Pobierz dane historyczne z rozszerzonym okresem
            price_data = self.fetcher.fetch_comprehensive_data(symbols, period='max')
            if price_data is None or price_data.empty:
                print("Error: No price data received")
                return None

            print(f"Raw data shape: {price_data.shape}")
            print(f"Raw data columns: {list(price_data.columns)}")
            print(f"Raw data date range: {price_data.index[0]} to {price_data.index[-1]}")

            # Sprawdź czy wszystkie symbole są w danych
            missing_symbols = set(symbols) - set(price_data.columns)
            if missing_symbols:
                print(f"Warning: Missing data for symbols: {missing_symbols}")
                # Usuń brakujące symbole
                symbols = [s for s in symbols if s in price_data.columns]
                if not symbols:
                    print("Error: No valid symbols found in data")
                    return None
                price_data = price_data[symbols]

            # Konwertuj daty na datetime jeśli to string
            try:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
            except Exception as e:
                print(f"Error parsing dates: {e}")
                return None

            # Filtruj dane do okresu backtestingu z buforem
            buffer_start = start_dt - timedelta(days=LOOKBACK_PERIOD + 100)  # Dodaj bufor

            # Sprawdź czy mamy wystarczająco danych historycznych
            if price_data.index[0] > buffer_start:
                print(
                    f"Warning: Insufficient historical data. Available from {price_data.index[0]}, needed from {buffer_start}")
                buffer_start = price_data.index[0]

            # Filtruj dane
            mask = (price_data.index >= buffer_start) & (price_data.index <= end_dt)
            price_data = price_data.loc[mask]

            if price_data.empty:
                print("Error: No data in specified date range")
                return None

            print(f"Filtered data shape: {price_data.shape}")
            print(f"Filtered data date range: {price_data.index[0]} to {price_data.index[-1]}")

            # Usuń kolumny z wszystkimi NaN
            price_data = price_data.dropna(axis=1, how='all')

            # Forward fill brakujące wartości
            price_data = price_data.fillna(method='ffill')

            # Usuń wiersze gdzie wszystkie wartości to NaN
            price_data = price_data.dropna(how='all')

            if price_data.empty:
                print("Error: No valid data after cleaning")
                return None

            print(f"Clean data shape: {price_data.shape}")

            # Przygotuj strukturę danych dla wyników
            backtest_results = []
            portfolio_value = self.initial_capital
            current_position = None
            transaction_costs = 0.001  # 0.1% koszty transakcji

            # Generuj daty rebalansowania
            rebalance_dates = self._generate_rebalance_dates(price_data.index, rebalance_freq, start_dt)

            if not rebalance_dates:
                print("Error: No rebalance dates generated")
                return None

            print(f"Generated {len(rebalance_dates)} rebalance dates")
            print(f"First rebalance: {rebalance_dates[0]}")
            print(f"Last rebalance: {rebalance_dates[-1]}")

            successful_rebalances = 0

            for i, date in enumerate(rebalance_dates):
                try:
                    # Znajdź najbliższą dostępną datę w danych
                    available_date = self._find_nearest_date(price_data.index, date)
                    if available_date is None:
                        print(f"Warning: No data available for rebalance date {date}")
                        continue

                    # Sprawdź czy mamy wystarczająco danych historycznych
                    historical_data = price_data.loc[:available_date]

                    if len(historical_data) < LOOKBACK_PERIOD + 30:
                        print(
                            f"Warning: Insufficient historical data at {available_date} (need {LOOKBACK_PERIOD + 30}, have {len(historical_data)})")
                        continue

                    # Oblicz zwroty na lookback period
                    lookback_data = historical_data.tail(LOOKBACK_PERIOD + 1)

                    if lookback_data.empty or len(lookback_data) < 2:
                        print(f"Warning: Insufficient lookback data at {available_date}")
                        continue

                    returns = self._calculate_period_returns(lookback_data)

                    if returns is None or not returns:
                        print(f"Warning: Could not calculate returns at {available_date}")
                        continue

                    # Zastosuj strategię GEM
                    try:
                        momentum_scores = self.strategy.calculate_momentum_scores(returns)
                        if not momentum_scores:
                            print(f"Warning: No momentum scores at {available_date}")
                            continue

                        sorted_etfs = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)

                        # Wybierz najlepszy ETF lub obligacje
                        best_etf = sorted_etfs[0]
                        if best_etf[1] > 0:
                            new_position = best_etf[0]
                        else:
                            # Wybierz obligacje
                            bond_etfs = [etf for etf in symbols if 'BU0' in etf or 'IB01' in etf]
                            new_position = bond_etfs[0] if bond_etfs else sorted_etfs[0][0]  # Fallback na najlepszy ETF

                    except Exception as strategy_error:
                        print(f"Strategy error at {available_date}: {strategy_error}")
                        continue

                    # Sprawdź czy trzeba zmienić pozycję
                    if new_position != current_position and new_position is not None:
                        if current_position is not None:
                            # Sprzedaj poprzednią pozycję (z kosztami)
                            portfolio_value *= (1 - transaction_costs)

                        current_position = new_position
                        # Kup nową pozycję (z kosztami)
                        portfolio_value *= (1 - transaction_costs)

                    # Oblicz wartość portfela na koniec okresu
                    portfolio_value = self._calculate_portfolio_value(
                        portfolio_value, current_position, available_date,
                        rebalance_dates, i, price_data
                    )

                    # Zapisz wyniki
                    backtest_results.append({
                        'date': available_date,
                        'position': current_position,
                        'portfolio_value': portfolio_value,
                        'momentum_scores': momentum_scores.copy(),
                        'best_etf': best_etf[0],
                        'best_momentum': best_etf[1]
                    })

                    successful_rebalances += 1

                    if successful_rebalances % 10 == 0:
                        print(f"Processed {successful_rebalances} rebalances, current value: ${portfolio_value:,.0f}")

                except Exception as e:
                    print(f"Error processing rebalance at {date}: {e}")
                    continue

            print(f"Backtest completed. Successful rebalances: {successful_rebalances}")

            if not backtest_results:
                print("Error: No successful backtest results")
                return None

            return self._calculate_performance_metrics(backtest_results, price_data, symbols)

        except Exception as e:
            print(f"Critical error in run_backtest: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_nearest_date(self, date_index, target_date):
        """Znajdź najbliższą dostępną datę w indeksie"""
        try:
            # Sprawdź czy target_date jest już w indeksie
            if target_date in date_index:
                return target_date

            # Znajdź najbliższą datę >= target_date
            future_dates = date_index[date_index >= target_date]
            if len(future_dates) > 0:
                return future_dates[0]

            # Jeśli nie ma przyszłych dat, weź ostatnią dostępną
            if len(date_index) > 0:
                return date_index[-1]

            return None

        except Exception as e:
            print(f"Error finding nearest date: {e}")
            return None

    def _calculate_portfolio_value(self, current_value, position, current_date, rebalance_dates, current_idx,
                                   price_data):
        """Oblicz wartość portfela na koniec okresu"""
        try:
            if position is None or position not in price_data.columns:
                return current_value

            current_price = price_data.loc[current_date, position]
            if pd.isna(current_price) or current_price <= 0:
                return current_value

            # Znajdź następną datę rebalansowania lub koniec danych
            if current_idx + 1 < len(rebalance_dates):
                next_rebalance = rebalance_dates[current_idx + 1]
                next_date = self._find_nearest_date(price_data.index, next_rebalance)
            else:
                next_date = price_data.index[-1]

            if next_date is None or next_date not in price_data.index:
                return current_value

            next_price = price_data.loc[next_date, position]
            if pd.isna(next_price) or next_price <= 0:
                return current_value

            period_return = (next_price / current_price) - 1
            return current_value * (1 + period_return)

        except Exception as e:
            print(f"Error calculating portfolio value: {e}")
            return current_value

    def _generate_rebalance_dates(self, date_index, freq, start_date):
        """Generuje daty rebalansowania z lepszą obsługą błędów"""
        try:
            if len(date_index) == 0:
                return []

            # Znajdź rzeczywistą datę rozpoczęcia (z buforem na lookback)
            actual_start = start_date
            if actual_start < date_index[0]:
                actual_start = date_index[0]

            # Dodaj dodatkowy bufor dla lookback
            buffer_date = actual_start + timedelta(days=LOOKBACK_PERIOD + 50)
            if buffer_date > date_index[-1]:
                print("Warning: Not enough data for proper backtest")
                return []

            end_date = date_index[-1]

            if freq == 'M':  # Miesięcznie
                dates = pd.date_range(start=buffer_date, end=end_date, freq='MS')  # MS = Month Start
            elif freq == 'Q':  # Kwartalnie
                dates = pd.date_range(start=buffer_date, end=end_date, freq='QS')  # QS = Quarter Start
            else:
                dates = pd.date_range(start=buffer_date, end=end_date, freq='MS')

            # Filtruj daty które są w zakresie naszych danych
            valid_dates = [d for d in dates if d >= date_index[0] and d <= date_index[-1]]

            print(f"Generated {len(valid_dates)} rebalance dates from {buffer_date} to {end_date}")
            return valid_dates

        except Exception as e:
            print(f"Error generating rebalance dates: {e}")
            return []

    def _calculate_period_returns(self, price_data):
        """Oblicza zwroty dla ostatniego okresu lookback z lepszą obsługą błędów"""
        try:
            if price_data is None or price_data.empty:
                return None

            if len(price_data) < 2:
                print(f"Warning: Insufficient data for return calculation (need 2, have {len(price_data)})")
                return None

            returns = {}
            for column in price_data.columns:
                try:
                    clean_data = price_data[column].dropna()
                    if len(clean_data) >= 2:
                        start_price = clean_data.iloc[0]
                        end_price = clean_data.iloc[-1]

                        if start_price > 0:  # Avoid division by zero
                            period_return = (end_price / start_price) - 1
                            returns[column] = period_return
                        else:
                            returns[column] = 0
                    else:
                        returns[column] = 0

                except Exception as e:
                    print(f"Error calculating return for {column}: {e}")
                    returns[column] = 0

            return returns if returns else None

        except Exception as e:
            print(f"Error in _calculate_period_returns: {e}")
            return None

    def _calculate_performance_metrics(self, backtest_results, price_data, symbols):
        """Oblicza metryki wydajności strategii z lepszą obsługą błędów"""
        try:
            if not backtest_results:
                return None

            df_results = pd.DataFrame(backtest_results)
            df_results.set_index('date', inplace=True)

            if df_results.empty or 'portfolio_value' not in df_results.columns:
                print("Error: No valid results to analyze")
                return None

            # Sprawdź czy mamy wystarczająco danych
            if len(df_results) < 2:
                print("Error: Insufficient data points for analysis")
                return None

            # Oblicz podstawowe metryki
            initial_value = self.initial_capital
            final_value = df_results['portfolio_value'].iloc[-1]
            total_return = (final_value / initial_value) - 1

            # Oblicz zwroty okresowe
            df_results['period_returns'] = df_results['portfolio_value'].pct_change()
            df_results['period_returns'] = df_results['period_returns'].fillna(0)

            # Annualized return
            years = (df_results.index[-1] - df_results.index[0]).days / 365.25
            if years <= 0:
                years = 1  # Minimum 1 year for calculation

            annualized_return = (1 + total_return) ** (1 / years) - 1

            # Volatility (annualized)
            period_returns = df_results['period_returns'].dropna()
            if len(period_returns) > 1:
                volatility = period_returns.std() * np.sqrt(252)  # Assume daily data, annualize
            else:
                volatility = 0

            # Sharpe ratio
            excess_return = annualized_return - 0.02  # Assume 2% risk-free rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0

            # Maximum drawdown
            portfolio_values = df_results['portfolio_value']
            rolling_max = portfolio_values.expanding().max()
            drawdowns = (portfolio_values - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()

            # Porównaj z buy & hold najlepszego ETF
            benchmark_returns = self._calculate_benchmark_performance(
                price_data, symbols, df_results.index[0], df_results.index[-1]
            )

            return {
                'backtest_data': df_results,
                'performance_metrics': {
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'final_value': final_value,
                    'years': years,
                    'num_trades': len(df_results),
                    'win_rate': len(period_returns[period_returns > 0]) / len(period_returns) if len(
                        period_returns) > 0 else 0
                },
                'benchmark_comparison': benchmark_returns
            }

        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_benchmark_performance(self, price_data, symbols, start_date, end_date):
        """Oblicza wydajność buy & hold dla każdego ETF jako benchmark"""
        try:
            if price_data is None or price_data.empty:
                return {}

            benchmark_data = price_data.loc[start_date:end_date]
            benchmarks = {}

            for symbol in symbols:
                try:
                    if symbol in benchmark_data.columns:
                        symbol_data = benchmark_data[symbol].dropna()
                        if len(symbol_data) >= 2:
                            total_return = (symbol_data.iloc[-1] / symbol_data.iloc[0]) - 1
                            years = (symbol_data.index[-1] - symbol_data.index[0]).days / 365.25
                            if years > 0:
                                annualized_return = (1 + total_return) ** (1 / years) - 1
                            else:
                                annualized_return = total_return

                            benchmarks[symbol] = {
                                'total_return': total_return,
                                'annualized_return': annualized_return
                            }
                except Exception as e:
                    print(f"Error calculating benchmark for {symbol}: {e}")
                    continue

            return benchmarks

        except Exception as e:
            print(f"Error in benchmark calculation: {e}")
            return {}


# Test backtestingu
if __name__ == "__main__":
    backtester = GEMBacktester(initial_capital=10000)
    uk_symbols = list(UK_ETFS.keys())

    print("Uruchamianie backtestingu...")
    results = backtester.run_backtest(uk_symbols, start_date='2020-01-01')

    if results:
        metrics = results['performance_metrics']
        print("✓ Backtest ukończony pomyślnie")
        print(f"Całkowity zwrot: {metrics['total_return']:.2%}")
        print(f"Zwrot roczny: {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maksymalny drawdown: {metrics['max_drawdown']:.2%}")
    else:
        print("✗ Błąd w backtestingu")