import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Optional
import math


@dataclass
class InvestmentPlan:
    name: str
    initial_investment: float
    monthly_contribution: float
    investment_horizon_years: int
    selected_etfs: List[str]
    allocation_weights: Dict[str, float]
    risk_profile: str
    target_amount: Optional[float] = None
    contribution_increases: Optional[Dict[int, float]] = None  # Year -> increase %


class AdvancedInvestmentPlanner:
    def __init__(self):
        self.risk_free_rate = 0.02
        self.inflation_rate = 0.025
        self.tax_rate = 0.19  # Capital gains tax

    def create_investment_projection(self, plan: InvestmentPlan, etf_metrics: Dict,
                                     scenarios: List[str] = None) -> Dict:
        """
        Tworzy kompleksową projekcję inwestycyjną z Monte Carlo
        """
        if scenarios is None:
            scenarios = ['conservative', 'moderate', 'aggressive']

        results = {}

        for scenario in scenarios:
            results[scenario] = self._run_scenario_simulation(plan, etf_metrics, scenario)

        # Dodaj portfolio-specific projection
        if etf_metrics:
            results['portfolio_based'] = self._run_portfolio_based_simulation(plan, etf_metrics)

        return results

    def _run_scenario_simulation(self, plan: InvestmentPlan, etf_metrics: Dict, scenario: str) -> Dict:
        """
        Uruchamia symulację dla konkretnego scenariusza
        """
        from config import INVESTMENT_SCENARIOS

        scenario_params = INVESTMENT_SCENARIOS.get(scenario, INVESTMENT_SCENARIOS['moderate'])

        # Monte Carlo simulation parameters
        num_simulations = 1000
        monthly_return = scenario_params['expected_return'] / 12
        monthly_volatility = scenario_params['volatility'] / np.sqrt(12)

        # Setup timeline
        months = plan.investment_horizon_years * 12
        timeline = pd.date_range(start=datetime.now(), periods=months, freq='M')

        # Run Monte Carlo simulations
        simulations = []

        for sim in range(num_simulations):
            portfolio_values = []
            contributions = []
            total_contributions = plan.initial_investment
            portfolio_value = plan.initial_investment

            for month in range(months):
                # Monthly contribution (with potential increases)
                year = month // 12
                monthly_contrib = plan.monthly_contribution

                if plan.contribution_increases and year in plan.contribution_increases:
                    increase_factor = 1 + (plan.contribution_increases[year] / 100)
                    monthly_contrib *= increase_factor

                # Add contribution
                portfolio_value += monthly_contrib
                total_contributions += monthly_contrib

                # Generate random return
                random_return = np.random.normal(monthly_return, monthly_volatility)
                portfolio_value *= (1 + random_return)

                portfolio_values.append(portfolio_value)
                contributions.append(total_contributions)

            simulations.append({
                'portfolio_values': portfolio_values,
                'contributions': contributions,
                'final_value': portfolio_values[-1],
                'total_contributions': contributions[-1]
            })

        # Calculate statistics
        final_values = [sim['final_value'] for sim in simulations]

        # Percentile analysis
        percentiles = {
            'p10': np.percentile(final_values, 10),
            'p25': np.percentile(final_values, 25),
            'p50': np.percentile(final_values, 50),
            'p75': np.percentile(final_values, 75),
            'p90': np.percentile(final_values, 90)
        }

        # Calculate success probability (if target is set)
        success_probability = 0
        if plan.target_amount:
            successful_sims = len([fv for fv in final_values if fv >= plan.target_amount])
            success_probability = successful_sims / num_simulations

        # Average timeline data for visualization
        avg_portfolio_values = np.mean([sim['portfolio_values'] for sim in simulations], axis=0)
        avg_contributions = np.mean([sim['contributions'] for sim in simulations], axis=0)

        return {
            'scenario': scenario,
            'timeline': timeline,
            'avg_portfolio_values': avg_portfolio_values,
            'avg_contributions': avg_contributions,
            'percentiles': percentiles,
            'success_probability': success_probability,
            'expected_return': scenario_params['expected_return'],
            'volatility': scenario_params['volatility'],
            'simulations': simulations[:50],  # Keep only 50 for memory efficiency
            'metrics': self._calculate_investment_metrics(avg_portfolio_values, avg_contributions, plan)
        }

    def _run_portfolio_based_simulation(self, plan: InvestmentPlan, etf_metrics: Dict) -> Dict:
        """
        Uruchamia symulację bazując na rzeczywistych metrykach portfela ETF
        """
        # Calculate portfolio expected return and volatility
        total_weight = sum(plan.allocation_weights.values())

        portfolio_return = 0
        portfolio_variance = 0

        for etf, weight in plan.allocation_weights.items():
            normalized_weight = weight / total_weight

            if etf in etf_metrics and etf_metrics[etf]:
                annual_return = etf_metrics[etf].get('annual_return', 0.08)
                volatility = etf_metrics[etf].get('volatility', 0.15)

                portfolio_return += normalized_weight * annual_return
                portfolio_variance += (normalized_weight ** 2) * (volatility ** 2)

        portfolio_volatility = np.sqrt(portfolio_variance)

        # Run simulation similar to scenario but with portfolio-specific parameters
        num_simulations = 1000
        monthly_return = portfolio_return / 12
        monthly_volatility = portfolio_volatility / np.sqrt(12)

        months = plan.investment_horizon_years * 12
        timeline = pd.date_range(start=datetime.now(), periods=months, freq='M')

        simulations = []

        for sim in range(num_simulations):
            portfolio_values = []
            contributions = []
            total_contributions = plan.initial_investment
            portfolio_value = plan.initial_investment

            for month in range(months):
                year = month // 12
                monthly_contrib = plan.monthly_contribution

                if plan.contribution_increases and year in plan.contribution_increases:
                    increase_factor = 1 + (plan.contribution_increases[year] / 100)
                    monthly_contrib *= increase_factor

                portfolio_value += monthly_contrib
                total_contributions += monthly_contrib

                # Generate correlated returns for realism
                base_return = np.random.normal(monthly_return, monthly_volatility)
                # Add some autocorrelation (momentum effect)
                if month > 0:
                    momentum_factor = 0.1 * (
                                portfolio_values[-1] / portfolio_values[max(0, month - 12)] - 1) if month >= 12 else 0
                    base_return += momentum_factor * 0.1

                portfolio_value *= (1 + base_return)

                portfolio_values.append(portfolio_value)
                contributions.append(total_contributions)

            simulations.append({
                'portfolio_values': portfolio_values,
                'contributions': contributions,
                'final_value': portfolio_values[-1],
                'total_contributions': contributions[-1]
            })

        # Calculate statistics
        final_values = [sim['final_value'] for sim in simulations]

        percentiles = {
            'p10': np.percentile(final_values, 10),
            'p25': np.percentile(final_values, 25),
            'p50': np.percentile(final_values, 50),
            'p75': np.percentile(final_values, 75),
            'p90': np.percentile(final_values, 90)
        }

        success_probability = 0
        if plan.target_amount:
            successful_sims = len([fv for fv in final_values if fv >= plan.target_amount])
            success_probability = successful_sims / num_simulations

        avg_portfolio_values = np.mean([sim['portfolio_values'] for sim in simulations], axis=0)
        avg_contributions = np.mean([sim['contributions'] for sim in simulations], axis=0)

        return {
            'scenario': 'portfolio_based',
            'timeline': timeline,
            'avg_portfolio_values': avg_portfolio_values,
            'avg_contributions': avg_contributions,
            'percentiles': percentiles,
            'success_probability': success_probability,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'simulations': simulations[:50],
            'metrics': self._calculate_investment_metrics(avg_portfolio_values, avg_contributions, plan)
        }

    def _calculate_investment_metrics(self, portfolio_values, contributions, plan: InvestmentPlan) -> Dict:
        """
        Oblicza zaawansowane metryki inwestycyjne
        """
        final_value = portfolio_values[-1]
        total_contributions = contributions[-1]
        total_gains = final_value - total_contributions

        # CAGR (Compound Annual Growth Rate)
        cagr = (final_value / plan.initial_investment) ** (1 / plan.investment_horizon_years) - 1

        # IRR calculation (simplified)
        monthly_contributions = [plan.initial_investment] + [plan.monthly_contribution] * (len(portfolio_values) - 1)
        cash_flows = [-plan.initial_investment] + [-plan.monthly_contribution] * (len(portfolio_values) - 1) + [
            final_value]

        try:
            irr = np.irr(cash_flows) * 12  # Annualized
        except:
            irr = cagr  # Fallback to CAGR

        # Return multiples
        money_multiple = final_value / total_contributions

        # Tax implications
        capital_gains = max(0, total_gains)
        estimated_taxes = capital_gains * self.tax_rate
        after_tax_value = final_value - estimated_taxes

        # Inflation-adjusted values
        inflation_factor = (1 + self.inflation_rate) ** plan.investment_horizon_years
        real_final_value = final_value / inflation_factor
        real_contributions = total_contributions / inflation_factor

        return {
            'final_value': final_value,
            'total_contributions': total_contributions,
            'total_gains': total_gains,
            'cagr': cagr,
            'irr': irr,
            'money_multiple': money_multiple,
            'estimated_taxes': estimated_taxes,
            'after_tax_value': after_tax_value,
            'real_final_value': real_final_value,
            'real_gains': real_final_value - real_contributions,
            'break_even_months': self._calculate_break_even(portfolio_values, contributions)
        }

    def _calculate_break_even(self, portfolio_values, contributions) -> int:
        """
        Oblicza w którym miesiącu portfel osiągnie break-even
        """
        for i, (portfolio_val, contrib) in enumerate(zip(portfolio_values, contributions)):
            if portfolio_val >= contrib:
                return i + 1
        return len(portfolio_values)  # Never breaks even in the period

    def optimize_allocation(self, plan: InvestmentPlan, etf_metrics: Dict, target_return: float = None) -> Dict:
        """
        Optymalizuje alokację portfela używając Modern Portfolio Theory
        """
        if not etf_metrics or len(plan.selected_etfs) < 2:
            return None

        # Extract returns and volatilities
        returns = []
        volatilities = []
        valid_etfs = []

        for etf in plan.selected_etfs:
            if etf in etf_metrics and etf_metrics[etf]:
                returns.append(etf_metrics[etf].get('annual_return', 0.08))
                volatilities.append(etf_metrics[etf].get('volatility', 0.15))
                valid_etfs.append(etf)

        if len(valid_etfs) < 2:
            return None

        returns = np.array(returns)
        volatilities = np.array(volatilities)

        # Simple correlation matrix (in real implementation, calculate from historical data)
        n_assets = len(valid_etfs)
        correlation_matrix = np.eye(n_assets)

        # Add some realistic correlations
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                # Assume moderate correlation between different assets
                correlation_matrix[i, j] = correlation_matrix[j, i] = np.random.uniform(0.3, 0.7)

        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

        # Optimize for maximum Sharpe ratio
        def portfolio_performance(weights):
            portfolio_return = np.sum(returns * weights)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return portfolio_return, portfolio_volatility, sharpe_ratio

        def negative_sharpe(weights):
            return -portfolio_performance(weights)[2]

        # Constraints and bounds
        from scipy.optimize import minimize

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Equal weight starting point
        initial_weights = np.array([1 / n_assets] * n_assets)

        # Optimize
        try:
            result = minimize(negative_sharpe, initial_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints)

            optimal_weights = result.x
            opt_return, opt_volatility, opt_sharpe = portfolio_performance(optimal_weights)

            return {
                'optimal_weights': dict(zip(valid_etfs, optimal_weights)),
                'expected_return': opt_return,
                'volatility': opt_volatility,
                'sharpe_ratio': opt_sharpe,
                'optimization_success': result.success
            }
        except Exception as e:
            print(f"Optimization failed: {e}")
            return None

    def calculate_dca_advantage(self, plan: InvestmentPlan, etf_metrics: Dict) -> Dict:
        """
        Oblicza przewagę Dollar Cost Averaging vs Lump Sum
        """
        # Simulate lump sum investment
        total_investment = plan.initial_investment + (plan.monthly_contribution * 12 * plan.investment_horizon_years)

        # Get portfolio expected return
        if etf_metrics:
            portfolio_return = np.mean(
                [metrics.get('annual_return', 0.08) for metrics in etf_metrics.values() if metrics])
        else:
            portfolio_return = 0.08

        # Lump sum final value
        lump_sum_final = total_investment * ((1 + portfolio_return) ** plan.investment_horizon_years)

        # DCA simulation (simplified)
        dca_final = plan.initial_investment
        monthly_return = portfolio_return / 12

        for month in range(plan.investment_horizon_years * 12):
            dca_final *= (1 + monthly_return)
            dca_final += plan.monthly_contribution

        return {
            'lump_sum_final': lump_sum_final,
            'dca_final': dca_final,
            'dca_advantage': dca_final - lump_sum_final,
            'dca_advantage_percent': ((dca_final / lump_sum_final) - 1) * 100
        }


# Test investment planner
if __name__ == "__main__":
    planner = AdvancedInvestmentPlanner()

    test_plan = InvestmentPlan(
        name="Test Plan",
        initial_investment=10000,
        monthly_contribution=1000,
        investment_horizon_years=10,
        selected_etfs=['VTI', 'VEA', 'BND'],
        allocation_weights={'VTI': 60, 'VEA': 30, 'BND': 10},
        risk_profile='moderate',
        target_amount=500000
    )

    print("Investment planner initialized successfully!")