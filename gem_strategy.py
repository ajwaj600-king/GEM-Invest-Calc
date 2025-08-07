import pandas as pd
import numpy as np
from data_fetcher import AdvancedDataFetcher  # ← ZMIANA: AdvancedDataFetcher zamiast DataFetcher
from config import LOOKBACK_PERIOD, RISK_FREE_RATE


class GEMStrategy:
    def __init__(self):
        self.fetcher = AdvancedDataFetcher()  # ← ZMIANA

    def calculate_momentum_scores(self, returns_dict):
        """
        Oblicza wyniki momentum dla każdego ETF
        Momentum = zwrot - stopa wolna od ryzyka
        """
        momentum_scores = {}
        annual_risk_free = RISK_FREE_RATE

        for etf, total_return in returns_dict.items():
            # Przelicz stopę wolną od ryzyka na okres 12 miesięcy
            momentum_score = total_return - annual_risk_free
            momentum_scores[etf] = momentum_score

        return momentum_scores

    def get_gem_signal(self, uk_etfs, us_reference_etfs=None):
        """
        Główna logika strategii GEM
        Zwraca rekomendację alokacji
        """
        # Pobierz dane dla UK ETFs
        uk_data = self.fetcher.fetch_comprehensive_data(uk_etfs, period='2y')  # ← ZMIANA: użyj nowej metody
        if uk_data is None:
            return None

        # Oblicz zwroty
        uk_metrics = self.fetcher.calculate_advanced_metrics(uk_data, LOOKBACK_PERIOD)  # ← ZMIANA
        if uk_metrics is None:
            return None

        # Przygotuj zwroty w starym formacie dla kompatybilności
        uk_returns = {etf: metrics['total_return'] for etf, metrics in uk_metrics.items()}

        # Oblicz momentum scores
        momentum_scores = self.calculate_momentum_scores(uk_returns)

        # Sortuj ETF-y według momentum (malejąco)
        sorted_etfs = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)

        # Strategia GEM: wybierz ETF z najwyższym momentum
        best_etf = sorted_etfs[0]

        # Sprawdź czy najlepszy ETF ma pozytywne momentum
        if best_etf[1] > 0:
            recommendation = {
                'action': 'BUY_EQUITY',
                'etf': best_etf[0],
                'momentum_score': best_etf[1],
                'allocation': 1.0  # 100% w najlepszy ETF
            }
        else:
            # Jeśli wszystkie mają negatywne momentum, idź w obligacje
            bond_etfs = [etf for etf in uk_etfs if 'bond' in etf.lower() or 'BU0' in etf or 'IB01' in etf]
            if bond_etfs:
                recommendation = {
                    'action': 'BUY_BONDS',
                    'etf': bond_etfs[0],  # Wybierz pierwszy ETF obligacyjny
                    'momentum_score': 0,
                    'allocation': 1.0
                }
            else:
                recommendation = {
                    'action': 'CASH',
                    'etf': None,
                    'momentum_score': 0,
                    'allocation': 0.0
                }

        # Dodaj szczegóły wszystkich ETF-ów do analizy
        recommendation['all_scores'] = momentum_scores
        recommendation['ranking'] = sorted_etfs

        return recommendation


# Test strategii
if __name__ == "__main__":
    from config import UK_ETFS

    strategy = GEMStrategy()
    uk_symbols = list(UK_ETFS.keys())

    print("Testowanie strategii GEM...")
    result = strategy.get_gem_signal(uk_symbols)

    if result:
        print("✓ Strategia wykonana pomyślnie")
        print(f"Rekomendacja: {result['action']}")
        print(f"ETF: {result['etf']}")
        print(f"Momentum Score: {result['momentum_score']:.4f}")
    else:
        print("✗ Błąd wykonania strategii")