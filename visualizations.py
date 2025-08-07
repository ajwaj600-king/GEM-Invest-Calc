import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
from config import BRAND_COLORS, BENCHMARK_INDICES


class ProfessionalVisualizer:
    def __init__(self):
        self.colors = BRAND_COLORS
        self.template = self._create_custom_template()

    def _create_custom_template(self):
        """Creates custom Plotly template"""
        return {
            'layout': {
                'paper_bgcolor': 'white',
                'plot_bgcolor': 'white',
                'font': {'family': 'Inter, sans-serif', 'size': 12},
                'colorway': [self.colors['primary'], self.colors['secondary'],
                             self.colors['success'], self.colors['danger'],
                             self.colors['warning'], self.colors['info']],
                'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60},
                'xaxis': {'gridcolor': '#E0E0E0', 'linecolor': '#E0E0E0'},
                'yaxis': {'gridcolor': '#E0E0E0', 'linecolor': '#E0E0E0'}
            }
        }

    def create_executive_dashboard(self, backtest_results, benchmark_data=None):
        """
        Creates comprehensive executive dashboard with proper error handling
        """
        try:
            # Check if we have valid backtest data
            if not backtest_results or 'backtest_data' not in backtest_results:
                return self._create_placeholder_dashboard()

            df = backtest_results['backtest_data']

            # If DataFrame is empty or doesn't have required columns, create placeholder
            if df.empty or 'portfolio_value' not in df.columns:
                return self._create_placeholder_dashboard()

            # Create subplot figure with multiple charts
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Portfolio Performance vs Benchmarks', 'Performance Metrics',
                                'Portfolio Allocation Timeline', 'Risk Analysis'),
                specs=[[{"colspan": 2}, None],
                       [{"type": "bar"}, {"type": "scatter"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )

            # 1. Main performance chart
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['portfolio_value'],
                    mode='lines',
                    name='GEM Strategy',
                    line=dict(color=self.colors['primary'], width=3),
                    hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )

            # Add benchmark comparison if available
            if benchmark_data is not None:
                self._add_benchmark_traces(fig, df, benchmark_data, row=1, col=1)

            # 2. Performance metrics bar chart
            if 'performance_metrics' in backtest_results:
                self._add_performance_metrics_chart(fig, backtest_results['performance_metrics'], row=2, col=1)

            # 3. Risk analysis scatter
            if 'period_returns' in df.columns:
                self._add_risk_analysis_scatter(fig, df, row=2, col=2)

            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="🚀 GEM Strategy - Executive Performance Dashboard",
                title_x=0.5,
                title_font_size=20,
                template='plotly_white'
            )

            return fig

        except Exception as e:
            print(f"Error creating executive dashboard: {e}")
            return self._create_placeholder_dashboard()

    def _create_placeholder_dashboard(self):
        """Creates a placeholder dashboard when data is not available"""
        try:
            fig = go.Figure()

            # Create sample data for demonstration
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            sample_portfolio = 100000 * (1 + np.cumsum(np.random.normal(0.0005, 0.015, len(dates))))

            fig.add_trace(go.Scatter(
                x=dates,
                y=sample_portfolio,
                mode='lines',
                name='Sample Portfolio Simulation',
                line=dict(color=self.colors['primary'], width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ))

            fig.update_layout(
                title="📊 Portfolio Performance (Demo Data)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                template='plotly_white',
                annotations=[
                    dict(
                        x=0.5, y=0.5,
                        xref="paper", yref="paper",
                        text="Demo visualization<br>Run backtest for real data",
                        showarrow=False,
                        font=dict(size=16, color="gray"),
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                ]
            )

            return fig

        except Exception as e:
            print(f"Error creating placeholder dashboard: {e}")
            return None

    def _add_benchmark_traces(self, fig, df, benchmark_data, row, col):
        """Add benchmark traces to the chart"""
        try:
            if benchmark_data is None or df.empty:
                return

            initial_value = df['portfolio_value'].iloc[0] if 'portfolio_value' in df.columns else 100000

            for benchmark, data in benchmark_data.items():
                if benchmark in ['^GSPC', 'WIG20.WA'] and data is not None:
                    try:
                        # Align data with portfolio dates
                        aligned_data = data.reindex(df.index, method='ffill').dropna()
                        if len(aligned_data) > 0:
                            normalized_benchmark = (aligned_data / aligned_data.iloc[0]) * initial_value

                            color = self.colors['danger'] if benchmark == '^GSPC' else self.colors['warning']
                            name = 'S&P 500' if benchmark == '^GSPC' else 'WIG20'

                            fig.add_trace(
                                go.Scatter(
                                    x=aligned_data.index,
                                    y=normalized_benchmark,
                                    mode='lines',
                                    name=name,
                                    line=dict(color=color, width=2, dash='dash'),
                                    hovertemplate=f'{name}: %{{y:,.0f}}<extra></extra>'
                                ),
                                row=row, col=col
                            )
                    except Exception as e:
                        print(f"Error adding benchmark {benchmark}: {e}")
                        continue
        except Exception as e:
            print(f"Error in _add_benchmark_traces: {e}")

    def _add_performance_metrics_chart(self, fig, metrics, row, col):
        """Add performance metrics bar chart"""
        try:
            if not metrics:
                return

            # Safe metric extraction with defaults
            metric_names = ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown']
            metric_values = [
                metrics.get('total_return', 0) * 100,
                metrics.get('annualized_return', 0) * 100,
                metrics.get('sharpe_ratio', 0) * 50,  # Scale for visibility
                abs(metrics.get('max_drawdown', 0)) * 100
            ]

            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name='Performance Metrics',
                    marker_color=[self.colors['success'], self.colors['primary'],
                                  self.colors['info'], self.colors['danger']],
                    hovertemplate='%{x}: %{y:.1f}<extra></extra>'
                ),
                row=row, col=col
            )

        except Exception as e:
            print(f"Error adding performance metrics chart: {e}")

    def _add_risk_analysis_scatter(self, fig, df, row, col):
        """Add risk analysis scatter plot"""
        try:
            if 'period_returns' not in df.columns or df['period_returns'].empty:
                return

            returns = df['period_returns'].dropna()
            if len(returns) < 2:
                return

            # Rolling volatility
            rolling_vol = returns.rolling(window=30, min_periods=10).std() * np.sqrt(252) * 100
            rolling_ret = returns.rolling(window=30, min_periods=10).mean() * 252 * 100

            # Remove NaN values
            valid_data = pd.DataFrame({'vol': rolling_vol, 'ret': rolling_ret}).dropna()

            if not valid_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=valid_data['vol'],
                        y=valid_data['ret'],
                        mode='markers',
                        name='Risk-Return Profile',
                        marker=dict(
                            size=8,
                            color=self.colors['secondary'],
                            opacity=0.6
                        ),
                        hovertemplate='Volatility: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
                    ),
                    row=row, col=col
                )

        except Exception as e:
            print(f"Error adding risk analysis scatter: {e}")

    def create_benchmark_comparison_advanced(self, backtest_results, benchmark_data, etf_metrics):
        """
        Advanced benchmark comparison with error handling
        """
        try:
            if not backtest_results or 'performance_metrics' not in backtest_results:
                return self._create_simple_metrics_chart(etf_metrics)

            metrics = backtest_results['performance_metrics']

            # Prepare data for comparison
            comparison_data = {
                'Strategy/Asset': ['GEM Strategy'],
                'Annual Return': [metrics.get('annualized_return', 0)],
                'Volatility': [metrics.get('volatility', 0)],
                'Sharpe Ratio': [metrics.get('sharpe_ratio', 0)],
                'Max Drawdown': [abs(metrics.get('max_drawdown', 0))],
            }

            # Add benchmark data if available
            if benchmark_data is not None:
                self._add_benchmark_metrics(comparison_data, benchmark_data)

            # Add ETF data if available
            if etf_metrics:
                self._add_etf_metrics(comparison_data, etf_metrics)

            # Create comparison chart
            df_comparison = pd.DataFrame(comparison_data)

            if len(df_comparison) == 0:
                return None

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Annual Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Risk-Return Profile'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )

            colors = [self.colors['primary']] + [self.colors['secondary']] * (len(df_comparison) - 1)

            # Annual Return
            fig.add_trace(
                go.Bar(
                    x=df_comparison['Strategy/Asset'],
                    y=[r * 100 for r in df_comparison['Annual Return']],
                    name='Annual Return %',
                    marker_color=colors,
                    showlegend=False
                ),
                row=1, col=1
            )

            # Sharpe Ratio
            fig.add_trace(
                go.Bar(
                    x=df_comparison['Strategy/Asset'],
                    y=df_comparison['Sharpe Ratio'],
                    name='Sharpe Ratio',
                    marker_color=colors,
                    showlegend=False
                ),
                row=1, col=2
            )

            # Max Drawdown
            fig.add_trace(
                go.Bar(
                    x=df_comparison['Strategy/Asset'],
                    y=[d * 100 for d in df_comparison['Max Drawdown']],
                    name='Max Drawdown %',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )

            # Risk-Return Scatter
            fig.add_trace(
                go.Scatter(
                    x=[v * 100 for v in df_comparison['Volatility']],
                    y=[r * 100 for r in df_comparison['Annual Return']],
                    mode='markers+text',
                    text=df_comparison['Strategy/Asset'],
                    textposition="top center",
                    marker=dict(size=15, color=colors),
                    name='Risk-Return Profile',
                    showlegend=False
                ),
                row=2, col=2
            )

            fig.update_layout(
                height=600,
                title_text="📊 Comprehensive Performance Comparison",
                title_x=0.5,
                template='plotly_white'
            )

            return fig

        except Exception as e:
            print(f"Error creating benchmark comparison: {e}")
            return self._create_simple_metrics_chart(etf_metrics)

    def _create_simple_metrics_chart(self, etf_metrics):
        """Create simple metrics chart when advanced comparison fails"""
        try:
            if not etf_metrics:
                return None

            # Extract basic metrics
            etf_names = []
            annual_returns = []
            sharpe_ratios = []

            for etf, metrics in etf_metrics.items():
                if metrics and isinstance(metrics, dict):
                    etf_names.append(etf)
                    annual_returns.append(metrics.get('annual_return', 0) * 100)
                    sharpe_ratios.append(metrics.get('sharpe_ratio', 0))

            if not etf_names:
                return None

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Annual Returns (%)', 'Sharpe Ratios')
            )

            # Annual returns
            fig.add_trace(
                go.Bar(
                    x=etf_names,
                    y=annual_returns,
                    name='Annual Return',
                    marker_color=self.colors['primary']
                ),
                row=1, col=1
            )

            # Sharpe ratios
            fig.add_trace(
                go.Bar(
                    x=etf_names,
                    y=sharpe_ratios,
                    name='Sharpe Ratio',
                    marker_color=self.colors['success']
                ),
                row=1, col=2
            )

            fig.update_layout(
                height=400,
                title_text="📊 ETF Performance Metrics",
                title_x=0.5,
                template='plotly_white',
                showlegend=False
            )

            return fig

        except Exception as e:
            print(f"Error creating simple metrics chart: {e}")
            return None

    def _add_benchmark_metrics(self, comparison_data, benchmark_data):
        """Add benchmark metrics to comparison data"""
        try:
            for benchmark, data in benchmark_data.items():
                if benchmark in ['^GSPC', 'WIG20.WA'] and data is not None:
                    try:
                        returns = data.pct_change().dropna()
                        if len(returns) > 0:
                            annual_ret = (1 + returns.mean()) ** 252 - 1
                            vol = returns.std() * np.sqrt(252)
                            sharpe = (annual_ret - 0.02) / vol if vol > 0 else 0

                            # Calculate max drawdown
                            rolling_max = data.expanding().max()
                            drawdowns = (data - rolling_max) / rolling_max
                            max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

                            name = 'S&P 500' if benchmark == '^GSPC' else 'WIG20'
                            comparison_data['Strategy/Asset'].append(name)
                            comparison_data['Annual Return'].append(annual_ret)
                            comparison_data['Volatility'].append(vol)
                            comparison_data['Sharpe Ratio'].append(sharpe)
                            comparison_data['Max Drawdown'].append(max_dd)
                    except Exception as e:
                        print(f"Error processing benchmark {benchmark}: {e}")
                        continue
        except Exception as e:
            print(f"Error in _add_benchmark_metrics: {e}")

    def _add_etf_metrics(self, comparison_data, etf_metrics):
        """Add ETF metrics to comparison data"""
        try:
            for etf, metric_data in etf_metrics.items():
                if metric_data and isinstance(metric_data, dict):
                    try:
                        comparison_data['Strategy/Asset'].append(f"{etf} (B&H)")
                        comparison_data['Annual Return'].append(metric_data.get('annual_return', 0))
                        comparison_data['Volatility'].append(metric_data.get('volatility', 0))
                        comparison_data['Sharpe Ratio'].append(metric_data.get('sharpe_ratio', 0))
                        comparison_data['Max Drawdown'].append(abs(metric_data.get('max_drawdown', 0)))
                    except Exception as e:
                        print(f"Error processing ETF {etf}: {e}")
                        continue
        except Exception as e:
            print(f"Error in _add_etf_metrics: {e}")

    def create_correlation_heatmap(self, correlation_matrix):
        """
        Creates professional correlation heatmap with error handling
        """
        try:
            if correlation_matrix is None or correlation_matrix.empty:
                return None

            # Create annotation text
            annotations = []
            for i, row in enumerate(correlation_matrix.index):
                for j, col in enumerate(correlation_matrix.columns):
                    try:
                        corr_val = correlation_matrix.iloc[i, j]
                        if pd.notna(corr_val):
                            annotations.append(
                                dict(
                                    x=col, y=row,
                                    text=f"{corr_val:.2f}",
                                    showarrow=False,
                                    font=dict(color="white" if abs(corr_val) > 0.5 else "black")
                                )
                            )
                    except Exception as e:
                        continue

            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))

            # Add annotations if we have them
            if annotations:
                fig.update_layout(annotations=annotations)

            fig.update_layout(
                title="🔗 Asset Correlation Matrix",
                title_x=0.5,
                width=600,
                height=500,
                template='plotly_white'
            )

            return fig

        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            return None

    def create_monthly_returns_heatmap(self, backtest_results):
        """
        Creates monthly returns heatmap with error handling
        """
        try:
            if not backtest_results or 'backtest_data' not in backtest_results:
                return None

            df = backtest_results['backtest_data']

            if df.empty or 'portfolio_value' not in df.columns:
                return None

            # Calculate monthly returns
            monthly_returns = df['portfolio_value'].resample('M').last().pct_change().dropna()

            if monthly_returns.empty:
                return None

            # Create matrix for heatmap
            monthly_data = monthly_returns.to_frame('returns')
            monthly_data['year'] = monthly_data.index.year
            monthly_data['month'] = monthly_data.index.month

            # Pivot to create heatmap structure
            heatmap_data = monthly_data.pivot(index='year', columns='month', values='returns')

            if heatmap_data.empty:
                return None

            # Month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # Ensure we have the right number of month columns
            available_months = sorted(heatmap_data.columns)
            month_labels = [month_names[i - 1] for i in available_months if 1 <= i <= 12]

            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values * 100,  # Convert to percentage
                x=month_labels,
                y=heatmap_data.index,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(heatmap_data.values * 100, 1),
                texttemplate="%{text}%",
                textfont={"size": 10},
                hovertemplate='%{y} %{x}<br>Return: %{z:.1f}%<extra></extra>'
            ))

            fig.update_layout(
                title="📅 Monthly Returns Heatmap",
                title_x=0.5,
                xaxis_title="Month",
                yaxis_title="Year",
                height=400,
                template='plotly_white'
            )

            return fig

        except Exception as e:
            print(f"Error creating monthly returns heatmap: {e}")
            return None

    def create_rolling_metrics_dashboard(self, backtest_results):
        """
        Creates rolling metrics dashboard with error handling
        """
        try:
            if not backtest_results or 'backtest_data' not in backtest_results:
                return None

            df = backtest_results['backtest_data']

            if df.empty or 'portfolio_value' not in df.columns:
                return None

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Portfolio Value Over Time', 'Rolling Volatility',
                                'Rolling Sharpe Ratio', 'Drawdown Analysis'),
                vertical_spacing=0.1
            )

            # Portfolio value over time
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['portfolio_value'],
                    mode='lines', name='Portfolio Value',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=1, col=1
            )

            # Rolling metrics if we have period returns
            if 'period_returns' in df.columns and not df['period_returns'].empty:
                returns = df['period_returns'].dropna()

                if len(returns) > 12:  # Need minimum data for rolling calculations
                    # Rolling volatility
                    rolling_vol = returns.rolling(window=12).std() * np.sqrt(12)

                    fig.add_trace(
                        go.Scatter(
                            x=rolling_vol.index, y=rolling_vol * 100,
                            mode='lines', name='Volatility %',
                            line=dict(color=self.colors['warning'], width=2)
                        ),
                        row=1, col=2
                    )

                    # Rolling Sharpe
                    rolling_sharpe = returns.rolling(window=12).apply(
                        lambda x: (x.mean() / x.std()) * np.sqrt(12) if x.std() > 0 else 0
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=rolling_sharpe.index, y=rolling_sharpe,
                            mode='lines', name='Sharpe Ratio',
                            line=dict(color=self.colors['success'], width=2)
                        ),
                        row=2, col=1
                    )

            # Drawdown analysis
            rolling_max = df['portfolio_value'].expanding().max()
            drawdowns = (df['portfolio_value'] - rolling_max) / rolling_max * 100

            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index, y=drawdowns,
                    mode='lines', name='Drawdown %',
                    line=dict(color=self.colors['danger'], width=2),
                    fill='tozeroy',
                    fillcolor=f"rgba(244, 67, 54, 0.3)"
                ),
                row=2, col=2
            )

            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="📈 Rolling Performance Metrics",
                title_x=0.5,
                template='plotly_white'
            )

            return fig

        except Exception as e:
            print(f"Error creating rolling metrics dashboard: {e}")
            return None


# Test advanced visualizations
if __name__ == "__main__":
    print("Professional visualizer with error handling ready! 🚀")


    def create_investment_projection_chart(self, projection_results: Dict, plan) -> go.Figure:
        """
        Tworzy zaawansowany wykres projekcji inwestycyjnej
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Investment Growth Projection', 'Contributions vs Returns',
                                'Scenario Comparison', 'Probability Distribution'),
                specs=[[{"colspan": 2}, None],
                       [{"type": "bar"}, {"type": "histogram"}]],
                vertical_spacing=0.12
            )

            colors = [self.colors['primary'], self.colors['success'], self.colors['warning'], self.colors['danger']]

            # Main projection chart
            for i, (scenario, data) in enumerate(projection_results.items()):
                if 'timeline' in data and 'avg_portfolio_values' in data:
                    fig.add_trace(
                        go.Scatter(
                            x=data['timeline'],
                            y=data['avg_portfolio_values'],
                            mode='lines',
                            name=f'{scenario.title()} Scenario',
                            line=dict(color=colors[i % len(colors)], width=3),
                            hovertemplate=f'{scenario}: %{{y:$,.0f}}<extra></extra>'
                        ),
                        row=1, col=1
                    )

                    # Add contributions line
                    fig.add_trace(
                        go.Scatter(
                            x=data['timeline'],
                            y=data['avg_contributions'],
                            mode='lines',
                            name=f'{scenario.title()} Contributions',
                            line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                            opacity=0.7,
                            hovertemplate=f'Contributions: %{{y:$,.0f}}<extra></extra>'
                        ),
                        row=1, col=1
                    )

            # Contributions vs Returns breakdown
            if 'moderate' in projection_results:
                moderate_data = projection_results['moderate']
                final_value = moderate_data['avg_portfolio_values'][-1]
                total_contributions = moderate_data['avg_contributions'][-1]
                gains = final_value - total_contributions

                fig.add_trace(
                    go.Bar(
                        x=['Total Contributions', 'Investment Gains'],
                        y=[total_contributions, gains],
                        marker_color=[self.colors['info'], self.colors['success']],
                        name='Breakdown',
                        text=[f'${total_contributions:,.0f}', f'${gains:,.0f}'],
                        textposition='auto'
                    ),
                    row=2, col=1
                )

            # Scenario comparison
            if len(projection_results) > 1:
                scenario_names = []
                final_values = []

                for scenario, data in projection_results.items():
                    if 'avg_portfolio_values' in data:
                        scenario_names.append(scenario.title())
                        final_values.append(data['avg_portfolio_values'][-1])

                if final_values:
                    fig.add_trace(
                        go.Histogram(
                            x=final_values,
                            name='Final Values Distribution',
                            marker_color=self.colors['secondary'],
                            opacity=0.7
                        ),
                        row=2, col=2
                    )

            fig.update_layout(
                height=800,
                title_text=f"📈 Investment Plan: {plan.name}",
                title_x=0.5,
                template='plotly_white',
                showlegend=True
            )

            return fig

        except Exception as e:
            print(f"Error creating investment projection chart: {e}")
            return None


    def create_risk_return_optimization_chart(self, optimization_result: Dict, etf_metrics: Dict) -> go.Figure:
        """
        Tworzy wykres optymalizacji risk-return
        """
        try:
            if not optimization_result:
                return None

            fig = go.Figure()

            # Individual ETFs
            etf_returns = []
            etf_volatilities = []
            etf_names = []

            for etf, metrics in etf_metrics.items():
                if metrics:
                    etf_returns.append(metrics.get('annual_return', 0) * 100)
                    etf_volatilities.append(metrics.get('volatility', 0) * 100)
                    etf_names.append(etf)

            # Plot individual ETFs
            fig.add_trace(
                go.Scatter(
                    x=etf_volatilities,
                    y=etf_returns,
                    mode='markers+text',
                    text=etf_names,
                    textposition="top center",
                    marker=dict(size=12, color=self.colors['secondary'], opacity=0.7),
                    name='Individual ETFs',
                    hovertemplate='%{text}<br>Return: %{y:.1f}%<br>Risk: %{x:.1f}%<extra></extra>'
                )
            )

            # Plot optimized portfolio
            opt_return = optimization_result['expected_return'] * 100
            opt_volatility = optimization_result['volatility'] * 100

            fig.add_trace(
                go.Scatter(
                    x=[opt_volatility],
                    y=[opt_return],
                    mode='markers+text',
                    text=['Optimized Portfolio'],
                    textposition="top center",
                    marker=dict(size=20, color=self.colors['primary'], symbol='star'),
                    name='Optimized Portfolio',
                    hovertemplate='Optimized Portfolio<br>Return: %{y:.1f}%<br>Risk: %{x:.1f}%<br>Sharpe: ' + f"{optimization_result['sharpe_ratio']:.2f}<extra></extra>"
                )
            )

            fig.update_layout(
                title='🎯 Portfolio Optimization: Risk vs Return',
                xaxis_title='Risk (Volatility %)',
                yaxis_title='Expected Return (%)',
                template='plotly_white',
                height=500
            )

            return fig

        except Exception as e:
            print(f"Error creating optimization chart: {e}")
            return None


    def create_monte_carlo_simulation_chart(self, simulation_data: Dict) -> go.Figure:
        """
        Tworzy wykres symulacji Monte Carlo
        """
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Monte Carlo Simulations', 'Final Value Distribution')
            )

            # Plot multiple simulation paths
            simulations = simulation_data.get('simulations', [])
            timeline = simulation_data.get('timeline', [])

            # Show only first 20 simulations for clarity
            for i, sim in enumerate(simulations[:20]):
                fig.add_trace(
                    go.Scatter(
                        x=timeline,
                        y=sim['portfolio_values'],
                        mode='lines',
                        line=dict(color='rgba(128,128,128,0.3)', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )

            # Add average path
            avg_values = simulation_data.get('avg_portfolio_values', [])
            if avg_values:
                fig.add_trace(
                    go.Scatter(
                        x=timeline,
                        y=avg_values,
                        mode='lines',
                        name='Average Path',
                        line=dict(color=self.colors['primary'], width=4)
                    ),
                    row=1, col=1
                )

            # Final value distribution
            final_values = [sim['final_value'] for sim in simulations]
            if final_values:
                fig.add_trace(
                    go.Histogram(
                        x=final_values,
                        name='Final Values',
                        marker_color=self.colors['success'],
                        opacity=0.7
                    ),
                    row=1, col=2
                )

            fig.update_layout(
                height=500,
                title_text='🎲 Monte Carlo Investment Simulation',
                template='plotly_white'
            )

            return fig

        except Exception as e:
            print(f"Error creating Monte Carlo chart: {e}")
            return None


    def create_dca_vs_lumpsum_chart(self, dca_analysis: Dict, plan) -> go.Figure:
        """
        Tworzy wykres porównania DCA vs Lump Sum
        """
        try:
            fig = go.Figure()

            # Create timeline for DCA
            months = plan.investment_horizon_years * 12
            timeline = pd.date_range(start=datetime.now(), periods=months, freq='M')

            # DCA accumulation
            dca_values = []
            cumulative_investment = plan.initial_investment

            monthly_return = 0.08 / 12  # Assumed 8% annual return
            current_value = plan.initial_investment

            for month in range(months):
                current_value *= (1 + monthly_return)
                current_value += plan.monthly_contribution
                cumulative_investment += plan.monthly_contribution
                dca_values.append(current_value)

            # Lump sum growth
            total_investment = plan.initial_investment + (plan.monthly_contribution * months)
            lump_sum_values = [total_investment * ((1 + 0.08) ** (month / 12)) for month in range(months)]

            # Plot both strategies
            fig.add_trace(
                go.Scatter(
                    x=timeline,
                    y=dca_values,
                    mode='lines',
                    name='Dollar Cost Averaging',
                    line=dict(color=self.colors['primary'], width=3),
                    hovertemplate='DCA: $%{y:,.0f}<extra></extra>'
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=timeline,
                    y=lump_sum_values,
                    mode='lines',
                    name='Lump Sum',
                    line=dict(color=self.colors['danger'], width=3, dash='dash'),
                    hovertemplate='Lump Sum: $%{y:,.0f}<extra></extra>'
                )
            )

            # Add difference area
            differences = [dca - lump for dca, lump in zip(dca_values, lump_sum_values)]

            fig.add_trace(
                go.Scatter(
                    x=timeline,
                    y=differences,
                    mode='lines',
                    name='DCA Advantage',
                    line=dict(color=self.colors['success'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(76, 175, 80, 0.3)',
                    hovertemplate='Difference: $%{y:,.0f}<extra></extra>'
                )
            )

            fig.update_layout(
                title='💰 Dollar Cost Averaging vs Lump Sum Investment',
                xaxis_title='Time',
                yaxis_title='Portfolio Value ($)',
                template='plotly_white',
                height=500
            )

            return fig

        except Exception as e:
            print(f"Error creating DCA comparison chart: {e}")
            return None