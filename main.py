import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Import naszych zaawansowanych modułów
from data_fetcher import AdvancedDataFetcher
from gem_strategy import GEMStrategy
from backtester import GEMBacktester
from visualizations import ProfessionalVisualizer
from investment_planner import AdvancedInvestmentPlanner, InvestmentPlan
from config import UK_ETFS, US_REFERENCE_ETFS, BENCHMARK_INDICES, BRAND_COLORS, CUSTOM_CSS, POPULAR_ETFS, \
    INVESTMENT_SCENARIOS
from translations import get_text, get_available_languages

# ================================
# KONFIGURACJA ENTERPRISE APP
# ================================

st.set_page_config(
    page_title="GEM Strategy - Professional Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# GEM Strategy Professional Dashboard v3.0\nProfessional Global Equities Momentum Strategy Calculator with Investment Planning Suite"
    }
)

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'custom_etfs' not in st.session_state:
    st.session_state.custom_etfs = {}
if 'investment_plans' not in st.session_state:
    st.session_state.investment_plans = {}
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = None

# Custom CSS styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ================================
# LANGUAGE HELPER FUNCTION
# ================================

def t(key):
    """Shorthand function for getting translated text"""
    return get_text(key, st.session_state.language)


# ================================
# CACHE & INITIALIZATION
# ================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def initialize_advanced_components():
    """Initialize all advanced components with caching"""
    return AdvancedDataFetcher(), GEMStrategy(), ProfessionalVisualizer(), GEMBacktester(), AdvancedInvestmentPlanner()


@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_market_data(symbols, benchmarks, period='2y'):
    """Fetch and cache market data"""
    fetcher = AdvancedDataFetcher()

    try:
        # Fetch ETF data
        etf_data = fetcher.fetch_comprehensive_data(symbols, period)
        etf_metrics = fetcher.calculate_advanced_metrics(etf_data) if etf_data is not None else None

        # Fetch benchmark data
        benchmark_data = fetcher.fetch_benchmark_data(period)

        # Fetch correlation matrix
        correlation_matrix = fetcher.get_correlation_matrix(symbols, period)

        # Fetch sector information
        sector_info = fetcher.get_sector_allocation(symbols)

        return etf_data, etf_metrics, benchmark_data, correlation_matrix, sector_info
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return None, None, None, None, None


# ================================
# UTILITY FUNCTIONS
# ================================

def safe_float_conversion(value, default=0.0):
    """Safely convert value to float"""
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace('%', '').replace('$', '').replace(',', '')
        return float(value)
    except (ValueError, TypeError):
        return default


def create_kpi_card(title, value, delta=None, delta_color="normal"):
    """Create professional KPI card"""
    delta_html = ""
    if delta is not None:
        delta_val = safe_float_conversion(delta)
        color = "#4CAF50" if delta_color == "normal" and delta_val > 0 else "#F44336" if delta_color == "normal" and delta_val < 0 else "#FFC107"
        arrow = "↗" if delta_val > 0 else "↘" if delta_val < 0 else "→"
        delta_html = f'<div style="color: {color}; font-size: 0.9rem;">{arrow} {delta}</div>'

    return f"""
    <div class="kpi-box">
        <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #1E88E5;">{value}</div>
        {delta_html}
    </div>
    """


def create_alert_box(message, alert_type="info"):
    """Create professional alert box"""
    colors = {"success": "#4CAF50", "warning": "#FF9800", "danger": "#F44336", "info": "#2196F3"}
    icons = {"success": "✅", "warning": "⚠️", "danger": "🚨", "info": "ℹ️"}

    return f"""
    <div style="background: {colors[alert_type]}15; border-left: 4px solid {colors[alert_type]}; 
                padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <span style="font-size: 1.2rem;">{icons[alert_type]}</span> 
        <strong>{message}</strong>
    </div>
    """


def export_to_json(data, filename):
    """Export data to JSON with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp}.json"

    try:
        with open(full_filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return full_filename
    except Exception as e:
        st.error(f"Export error: {e}")
        return None


def calculate_portfolio_health_score(metrics):
    """Calculate overall portfolio health score (0-100) with safe conversions"""
    if not metrics:
        return 0

    try:
        weights = {'sharpe_ratio': 0.3, 'annual_return': 0.25, 'max_drawdown': 0.2, 'volatility': 0.15, 'win_rate': 0.1}
        scores = {}

        sharpe = safe_float_conversion(metrics.get('sharpe_ratio', 0))
        annual_ret = safe_float_conversion(metrics.get('annual_return', 0))
        max_dd = abs(safe_float_conversion(metrics.get('max_drawdown', 0)))
        vol = safe_float_conversion(metrics.get('volatility', 0))
        win_rate = safe_float_conversion(metrics.get('win_rate', 0))

        scores['sharpe_ratio'] = min(100, max(0, (sharpe + 2) * 25))
        scores['annual_return'] = min(100, max(0, annual_ret * 500))
        scores['max_drawdown'] = min(100, max(0, (1 - max_dd) * 100))
        scores['volatility'] = min(100, max(0, (1 - min(vol, 1)) * 100))
        scores['win_rate'] = win_rate * 100

        health_score = sum(scores[metric] * weights[metric] for metric in weights.keys())
        return round(health_score, 1)

    except Exception as e:
        st.error(f"Error calculating health score: {e}")
        return 50.0


def safe_calculate_avg_metrics(etf_metrics):
    """Safely calculate average metrics with proper type conversion"""
    if not etf_metrics:
        return {}

    try:
        valid_metrics = []
        for symbol, metrics in etf_metrics.items():
            if metrics and isinstance(metrics, dict):
                converted_metrics = {}
                for key, value in metrics.items():
                    converted_metrics[key] = safe_float_conversion(value)
                valid_metrics.append(converted_metrics)

        if not valid_metrics:
            return {}

        avg_metrics = {}
        metric_keys = ['sharpe_ratio', 'annual_return', 'max_drawdown', 'volatility', 'win_rate']

        for key in metric_keys:
            values = [m.get(key, 0) for m in valid_metrics if key in m]
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0

        return avg_metrics

    except Exception as e:
        st.error(f"Error calculating average metrics: {e}")
        return {'sharpe_ratio': 0.0, 'annual_return': 0.0, 'max_drawdown': 0.0, 'volatility': 0.0, 'win_rate': 0.0}


def create_simple_performance_overview(etf_metrics, benchmark_data):
    """Create simple performance overview when full dashboard is not available"""
    try:
        if not etf_metrics:
            return None

        fig = make_subplots(rows=1, cols=2, subplot_titles=('ETF Annual Returns (%)', 'ETF Sharpe Ratios'))

        etf_names, annual_returns, sharpe_ratios = [], [], []

        for etf, metrics in etf_metrics.items():
            if metrics and isinstance(metrics, dict):
                etf_names.append(etf)
                annual_returns.append(safe_float_conversion(metrics.get('annual_return', 0)) * 100)
                sharpe_ratios.append(safe_float_conversion(metrics.get('sharpe_ratio', 0)))

        if etf_names:
            fig.add_trace(
                go.Bar(x=etf_names, y=annual_returns, name='Annual Return (%)', marker_color=BRAND_COLORS['primary']),
                row=1, col=1)
            fig.add_trace(
                go.Bar(x=etf_names, y=sharpe_ratios, name='Sharpe Ratio', marker_color=BRAND_COLORS['success']), row=1,
                col=2)

            fig.update_layout(height=400, title_text="📊 Current ETF Performance Overview", title_x=0.5,
                              template='plotly_white', showlegend=False)
            return fig

        return None

    except Exception as e:
        st.error(f"Error creating performance overview: {e}")
        return None


def save_decision_history(decision_data, filename="gem_decisions.json"):
    """Zapisuje historię decyzji do pliku JSON"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                history = json.load(f)
        else:
            history = []

        decision_data['timestamp'] = datetime.now().isoformat()
        history.append(decision_data)

        with open(filename, 'w') as f:
            json.dump(history, f, indent=2, default=str)

        return True
    except Exception as e:
        st.error(f"Błąd zapisywania historii: {e}")
        return False


def load_decision_history(filename="gem_decisions.json"):
    """Wczytuje historię decyzji z pliku JSON"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Błąd wczytywania historii: {e}")
        return []


def save_investment_plan(plan_data, filename="investment_plans.json"):
    """Save investment plan to JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                plans = json.load(f)
        else:
            plans = {}

        plans[plan_data['name']] = plan_data
        plans[plan_data['name']]['timestamp'] = datetime.now().isoformat()

        with open(filename, 'w') as f:
            json.dump(plans, f, indent=2, default=str)

        return True
    except Exception as e:
        st.error(f"Error saving investment plan: {e}")
        return False


def load_investment_plans(filename="investment_plans.json"):
    """Load investment plans from JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading investment plans: {e}")
        return {}


# ================================
# ETF MANAGEMENT FUNCTIONS
# ================================

def add_custom_etf_interface():
    """Interface for adding custom ETFs"""
    with st.expander(t('add_custom_etf')):
        col1, col2, col3 = st.columns(3)

        with col1:
            custom_symbol = st.text_input(t('etf_symbol'), placeholder="AAPL, TSLA, etc.")
        with col2:
            custom_name = st.text_input(t('name'), placeholder="Apple Inc., Tesla Inc., etc.")
        with col3:
            if st.button("➕ Add ETF"):
                if custom_symbol and custom_name:
                    st.session_state.custom_etfs[custom_symbol.upper()] = custom_name
                    st.success(f"✅ Added {custom_symbol.upper()}")
                    st.rerun()
                else:
                    st.warning("⚠️ Please fill both symbol and name")


def display_etf_selector(title, etf_dict, key_prefix):
    """Display ETF selector with checkboxes"""
    selected_etfs = []

    st.markdown(f"**{title}**")

    # Create columns for better layout
    cols = st.columns(3)
    for i, (symbol, name) in enumerate(etf_dict.items()):
        with cols[i % 3]:
            if st.checkbox(f"{symbol}", key=f"{key_prefix}_{symbol}"):
                selected_etfs.append(symbol)
            st.caption(name[:30] + "..." if len(name) > 30 else name)

    return selected_etfs


# ================================
# MAIN APPLICATION
# ================================

def main():
    # Initialize advanced components
    try:
        fetcher, strategy, visualizer, backtester, investment_planner = initialize_advanced_components()
    except Exception as e:
        st.error(f"{t('error_initializing')} {e}")
        return

    # ===============================
    # HEADER SECTION
    # ===============================

    st.markdown(f"""
    <div class="main-header">
        <h1>{t('main_title')}</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            {t('main_subtitle')}
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                {t('feature_realtime')}
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                {t('feature_backtesting')}
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                {t('feature_benchmark')}
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                💰 Investment Planning
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===============================
    # ADVANCED SIDEBAR
    # ===============================

    with st.sidebar:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h2>{t('control_center')}</h2>
            <p>{t('control_subtitle')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Language Selection
        st.subheader(t('language_selection'))
        available_languages = get_available_languages()

        selected_language = st.selectbox(
            "Choose Language / Wybierz Język",
            options=list(available_languages.keys()),
            format_func=lambda x: available_languages[x],
            index=list(available_languages.keys()).index(st.session_state.language),
            key="language_selector"
        )

        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.rerun()

        st.divider()

        # ETF Selection with advanced options
        st.subheader(t('asset_selection'))

        with st.expander(t('uk_etfs_primary'), expanded=True):
            selected_uk_etfs = []
            for symbol, name in UK_ETFS.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected = st.checkbox(f"{symbol}", value=True, key=f"uk_{symbol}")
                    if selected:
                        selected_uk_etfs.append(symbol)
                with col2:
                    st.info("🇬🇧")

        with st.expander(t('us_etfs_benchmark'), expanded=False):
            selected_us_etfs = []
            for symbol, name in US_REFERENCE_ETFS.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected = st.checkbox(f"{symbol}", value=True, key=f"us_{symbol}")
                    if selected:
                        selected_us_etfs.append(symbol)
                with col2:
                    st.info("🇺🇸")

        # Custom ETFs
        if st.session_state.custom_etfs:
            with st.expander(t('custom_etfs'), expanded=False):
                selected_custom_etfs = []
                for symbol, name in st.session_state.custom_etfs.items():
                    if st.checkbox(f"{symbol}", key=f"custom_{symbol}"):
                        selected_custom_etfs.append(symbol)
                    st.caption(name)
        else:
            selected_custom_etfs = []

        st.divider()

        # Advanced Strategy Parameters
        st.subheader(t('strategy_parameters'))

        col1, col2 = st.columns(2)
        with col1:
            lookback_months = st.slider(t('lookback_period'), 6, 24, 12, help="Momentum calculation period")
        with col2:
            confidence_threshold = st.slider(t('confidence_threshold'), 60, 95, 75, help="Signal confidence threshold")

        rebalance_freq = st.selectbox(
            t('rebalancing'),
            ["M", "Q", "2M"],
            format_func=lambda x: t('monthly') if x == "M" else t('quarterly') if x == "Q" else t('bimonthly')
        )

        risk_management = st.checkbox(t('risk_management'), value=True, help=t('risk_management_help'))

        st.divider()

        # Real-time settings
        st.subheader(t('realtime_settings'))
        auto_refresh = st.checkbox(t('auto_refresh'), help=t('auto_refresh_help'))
        if auto_refresh:
            refresh_interval = st.slider(t('refresh_interval'), 1, 15, 5)

        # Export options
        st.subheader(t('export_options'))
        export_format = st.selectbox(t('format'), ["JSON", "CSV", "Excel"])

        if st.button(t('export_analysis'), type="primary"):
            st.success(t('export_initiated'))

    # ===============================
    # MAIN DASHBOARD TABS
    # ===============================

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        t('tab_executive'),
        t('tab_live_signal'),
        t('tab_backtesting'),
        t('tab_investment_planning'),
        t('tab_market_analysis'),
        t('tab_portfolio'),
        t('tab_ai_insights')
    ])

    # ===============================
    # TAB 1: EXECUTIVE DASHBOARD
    # ===============================

    with tab1:
        st.markdown(f"### {t('executive_summary')}")

        if selected_uk_etfs:
            with st.spinner(t('loading_dashboard')):
                try:
                    etf_data, etf_metrics, benchmark_data, correlation_matrix, sector_info = fetch_market_data(
                        selected_uk_etfs, list(BENCHMARK_INDICES.keys())
                    )

                    if etf_data is not None and etf_metrics:
                        avg_metrics = safe_calculate_avg_metrics(etf_metrics)
                        health_score = calculate_portfolio_health_score(avg_metrics)

                        # KPI Row
                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            st.markdown(create_kpi_card(
                                t('portfolio_health'),
                                f"{health_score}/100",
                                f"+{health_score - 70:.1f}" if health_score > 70 else f"{health_score - 70:.1f}"
                            ), unsafe_allow_html=True)

                        with col2:
                            avg_return = safe_float_conversion(avg_metrics.get('annual_return', 0))
                            st.markdown(create_kpi_card(
                                t('avg_annual_return'),
                                f"{avg_return:.1%}",
                                f"+{(avg_return - 0.08) * 100:.1f}%" if avg_return > 0.08 else f"{(avg_return - 0.08) * 100:.1f}%"
                            ), unsafe_allow_html=True)

                        with col3:
                            avg_sharpe = safe_float_conversion(avg_metrics.get('sharpe_ratio', 0))
                            st.markdown(create_kpi_card(
                                t('avg_sharpe_ratio'),
                                f"{avg_sharpe:.2f}",
                                f"+{avg_sharpe - 1:.2f}" if avg_sharpe > 1 else f"{avg_sharpe - 1:.2f}"
                            ), unsafe_allow_html=True)

                        with col4:
                            max_dd = abs(safe_float_conversion(avg_metrics.get('max_drawdown', 0)))
                            st.markdown(create_kpi_card(
                                t('avg_max_drawdown'),
                                f"{max_dd:.1%}",
                                f"+{(max_dd - 0.15) * 100:.1f}%" if max_dd > 0.15 else f"{(max_dd - 0.15) * 100:.1f}%"
                            ), unsafe_allow_html=True)

                        with col5:
                            if avg_return > 0.1:
                                market_regime = t('regime_bullish')
                            elif avg_return > 0.05:
                                market_regime = t('regime_neutral')
                            else:
                                market_regime = t('regime_bearish')

                            st.markdown(create_kpi_card(
                                t('market_regime'),
                                market_regime,
                                None
                            ), unsafe_allow_html=True)

                        # Alert system
                        if health_score > 80:
                            st.markdown(create_alert_box(t('alert_excellent'), "success"), unsafe_allow_html=True)
                        elif health_score > 60:
                            st.markdown(create_alert_box(t('alert_moderate'), "warning"), unsafe_allow_html=True)
                        else:
                            st.markdown(create_alert_box(t('alert_attention'), "danger"), unsafe_allow_html=True)

                        # Advanced Charts Section
                        st.markdown(f"### {t('advanced_analytics')}")

                        if etf_metrics:
                            simple_chart = create_simple_performance_overview(etf_metrics, benchmark_data)
                            if simple_chart:
                                st.plotly_chart(simple_chart, use_container_width=True)
                            else:
                                st.info("📊 Run backtest in tab 3 for advanced performance charts")
                        else:
                            st.info("📊 Performance charts will appear after data loading")

                        # Benchmark comparison
                        col1, col2 = st.columns(2)

                        with col1:
                            if benchmark_data is not None:
                                try:
                                    comparison_chart = visualizer.create_benchmark_comparison_advanced(
                                        None, benchmark_data, etf_metrics
                                    )
                                    if comparison_chart:
                                        st.plotly_chart(comparison_chart, use_container_width=True)
                                    else:
                                        st.info("📊 Benchmark comparison charts coming soon")
                                except Exception as e:
                                    st.warning(f"{t('chart_unavailable')} {e}")

                        with col2:
                            if correlation_matrix is not None:
                                try:
                                    corr_chart = visualizer.create_correlation_heatmap(correlation_matrix)
                                    if corr_chart:
                                        st.plotly_chart(corr_chart, use_container_width=True)
                                    else:
                                        st.info("📊 Correlation matrix charts coming soon")
                                except Exception as e:
                                    st.warning(f"{t('chart_unavailable')} {e}")

                    else:
                        st.error(t('error_market_data'))

                        if etf_metrics:
                            st.markdown(f"#### {t('basic_etf_metrics')}")
                            metrics_data = []
                            for symbol, metrics in etf_metrics.items():
                                metrics_data.append({
                                    t('etf'): symbol,
                                    t('name'): UK_ETFS.get(symbol, symbol),
                                    t('current_price'): f"${safe_float_conversion(metrics.get('current_price', 0)):.2f}",
                                    t('annual_return'): f"{safe_float_conversion(metrics.get('annual_return', 0)):.1%}",
                                    t('sharpe_ratio'): f"{safe_float_conversion(metrics.get('sharpe_ratio', 0)):.2f}"
                                })

                            df_metrics = pd.DataFrame(metrics_data)
                            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"{t('error_loading')} {e}")

                    st.markdown(f"#### {t('fallback_analysis')}")
                    st.info(t('attempting_simple'))

                    try:
                        signal = strategy.get_gem_signal(selected_uk_etfs)
                        if signal:
                            action_text = t('action_buy_equity') if signal['action'] == 'BUY_EQUITY' else t(
                                'action_buy_bonds') if signal['action'] == 'BUY_BONDS' else t('action_cash')
                            st.success(f"✅ {t('current_recommendation')}: **{action_text}** for **{signal['etf']}**")
                        else:
                            st.warning(t('unable_generate_signal'))
                    except Exception as signal_error:
                        st.error(f"{t('signal_failed')} {signal_error}")
        else:
            st.warning(t('warning_select_etf'))

    # ===============================
    # TAB 2: LIVE SIGNAL (Skrócona wersja)
    # ===============================

    with tab2:
        st.markdown(f"### {t('live_signal_analysis')}")

        if selected_uk_etfs:
            col1, col2 = st.columns([2, 1])

            with col1:
                with st.spinner(t('analyzing_data')):
                    try:
                        signal = strategy.get_gem_signal(selected_uk_etfs)

                        if signal:
                            if signal['action'] == 'BUY_EQUITY':
                                action_text = t('action_buy_equity')
                            elif signal['action'] == 'BUY_BONDS':
                                action_text = t('action_buy_bonds')
                            else:
                                action_text = t('action_cash')

                            etf_text = signal['etf'] or t('cash_bonds')

                            st.markdown(f"""
                            <div class="strategy-card">
                                <h2>{t('current_recommendation')}</h2>
                                <h1>{action_text}</h1>
                                <p style="font-size: 1.2rem;">
                                    <strong>{t('target_etf')}:</strong> {etf_text}<br>
                                    <strong>{t('momentum_score')}:</strong> {signal['momentum_score']:.2%}<br>
                                    <strong>{t('allocation')}:</strong> {signal['allocation']:.0%}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                            signal_strength = abs(safe_float_conversion(signal['momentum_score'])) * 100
                            if signal_strength > 15:
                                strength_color = "#4CAF50"
                                strength_text = t('strength_strong')
                            elif signal_strength > 5:
                                strength_color = "#FF9800"
                                strength_text = t('strength_moderate')
                            else:
                                strength_color = "#F44336"
                                strength_text = t('strength_weak')

                            st.markdown(f"""
                            <div style="text-align: center; margin: 2rem 0;">
                                <div style="background: {strength_color}; color: white; padding: 1rem; 
                                           border-radius: 10px; font-size: 1.2rem; font-weight: bold;">
                                    📊 {t('signal_strength')}: {strength_text} ({signal_strength:.1f}%)
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Action buttons
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                if st.button(t('save_signal'), type="primary"):
                                    decision_data = {
                                        'signal': signal,
                                        'timestamp': datetime.now().isoformat(),
                                        'etfs_analyzed': selected_uk_etfs,
                                        'confidence': confidence_threshold
                                    }
                                    if save_decision_history(decision_data):
                                        st.success(t('signal_saved'))

                            with col2:
                                if st.button(t('send_alert')):
                                    st.info(t('alert_coming_soon'))

                            with col3:
                                if st.button(t('detailed_analysis')):
                                    st.info(t('redirecting'))

                            with col4:
                                if st.button(t('refresh_data')):
                                    st.cache_data.clear()
                                    st.rerun()

                        else:
                            st.error(t('unable_generate_signal'))

                    except Exception as e:
                        st.error(f"{t('signal_generation_error')} {e}")
                        st.info(t('try_different_etfs'))

            with col2:
                st.markdown(f"#### {t('market_indicators')}")

                try:
                    sentiment_score = np.random.randint(40, 90)
                    sentiment_color = "#4CAF50" if sentiment_score > 70 else "#FF9800" if sentiment_score > 50 else "#F44336"

                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <div style="background: {sentiment_color}15; border: 2px solid {sentiment_color}; 
                                   padding: 1rem; border-radius: 10px;">
                            <strong>{t('market_sentiment')}</strong><br>
                            <span style="font-size: 1.5rem; color: {sentiment_color};">{sentiment_score}/100</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric(t('vix_fear_index'), "22.5", delta="-1.2", help=t('market_volatility'))
                    st.metric(t('sp500_change'), "1.2%", delta="1.2%")

                except Exception as e:
                    st.warning(f"{t('market_indicators_unavailable')} {e}")

        else:
            st.warning(t('warning_select_etf'))

    # ===============================
    # TAB 3: ADVANCED BACKTESTING (Skrócona wersja)
    # ===============================

    with tab3:
        st.markdown(f"### {t('backtesting_suite')}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            start_date = st.date_input(t('start_date'), value=datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input(t('end_date'), value=datetime.now())
        with col3:
            initial_capital = st.number_input(t('initial_capital'), value=100000, min_value=1000, step=1000)
        with col4:
            transaction_cost = st.number_input(t('transaction_cost'), value=0.1, min_value=0.0, max_value=1.0,
                                               step=0.05)

        if st.button(t('run_backtest'), type="primary"):
            if selected_uk_etfs:
                progress_bar = st.progress(0)
                status_text = st.empty()

                with st.spinner(t('running_analysis')):
                    try:
                        backtester_advanced = GEMBacktester(initial_capital=initial_capital)

                        progress_bar.progress(20)
                        status_text.text(t('fetching_data'))

                        results = backtester_advanced.run_backtest(
                            symbols=selected_uk_etfs,
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d'),
                            rebalance_freq=rebalance_freq
                        )

                        progress_bar.progress(80)
                        status_text.text(t('calculating_metrics'))

                        if results:
                            progress_bar.progress(100)
                            status_text.text(t('analysis_complete'))

                            metrics = results['performance_metrics']
                            st.success(t('backtest_completed'))

                            st.markdown(f"#### {t('performance_summary')}")

                            col1, col2, col3, col4, col5 = st.columns(5)

                            with col1:
                                st.metric(t('total_return'),
                                          f"{safe_float_conversion(metrics.get('total_return', 0)):.1%}")
                            with col2:
                                st.metric(t('annual_return'),
                                          f"{safe_float_conversion(metrics.get('annualized_return', 0)):.1%}")
                            with col3:
                                st.metric(t('sharpe_ratio'),
                                          f"{safe_float_conversion(metrics.get('sharpe_ratio', 0)):.2f}")
                            with col4:
                                st.metric(t('max_drawdown'),
                                          f"{safe_float_conversion(metrics.get('max_drawdown', 0)):.1%}")
                            with col5:
                                volatility = safe_float_conversion(metrics.get('volatility', 0))
                                st.metric(t('volatility'), f"{volatility:.1%}")

                            try:
                                executive_dashboard = visualizer.create_executive_dashboard(results, None)
                                if executive_dashboard:
                                    st.plotly_chart(executive_dashboard, use_container_width=True)
                                else:
                                    st.info("📊 Advanced charts temporarily unavailable")
                            except Exception as chart_error:
                                st.warning(f"{t('chart_unavailable')} {chart_error}")

                        else:
                            st.error(t('backtest_failed'))

                        progress_bar.empty()
                        status_text.empty()

                    except Exception as e:
                        st.error(f"{t('backtest_error')} {e}")
                        progress_bar.empty()
                        status_text.empty()
            else:
                st.warning(t('warning_select_etf'))

    # ===============================
    # TAB 4: INVESTMENT PLANNING SUITE 💎🚀
    # ===============================

    with tab4:
        st.markdown(f"### {t('investment_planning_suite')}")

        # Create two main columns
        config_col, results_col = st.columns([1, 2])

        with config_col:
            st.markdown(f"#### {t('create_investment_plan')}")

            # Plan Name
            plan_name = st.text_input(t('plan_name'), value="My Investment Plan")

            # Basic Parameters
            st.markdown("**Basic Parameters**")
            initial_investment = st.number_input(t('initial_investment'), value=10000, min_value=100, step=100)
            monthly_contribution = st.number_input(t('monthly_contribution'), value=1000, min_value=0, step=50)
            investment_horizon = st.slider(t('investment_horizon'), 1, 50, 10)
            target_amount = st.number_input(t('target_amount'), value=500000, min_value=0, step=1000,
                                            help="Optional target amount")

            # Risk Profile
            risk_profile = st.selectbox(t('risk_profile'),
                                        options=['conservative', 'moderate', 'aggressive'],
                                        format_func=lambda x: t(x))

            st.divider()

            # ETF Selection for Investment Plan
            st.markdown(f"#### {t('portfolio_allocation')}")

            # Add custom ETF interface
            add_custom_etf_interface()

            # Combine all available ETFs
            all_available_etfs = {**UK_ETFS, **US_REFERENCE_ETFS, **POPULAR_ETFS, **st.session_state.custom_etfs}

            # ETF Selection with Popular ETFs
            with st.expander(t('popular_etfs'), expanded=True):
                investment_etfs = display_etf_selector("", POPULAR_ETFS, "invest_popular")

            # Custom ETFs
            if st.session_state.custom_etfs:
                with st.expander(t('custom_etfs')):
                    custom_investment_etfs = display_etf_selector("", st.session_state.custom_etfs, "invest_custom")
                    investment_etfs.extend(custom_investment_etfs)

            # Allocation Weights
            if investment_etfs:
                st.markdown(f"#### {t('allocation_weight')}")
                allocation_weights = {}
                total_allocation = 0

                for etf in investment_etfs:
                    weight = st.slider(f"{etf}", 0, 100, 100 // len(investment_etfs), key=f"weight_{etf}")
                    allocation_weights[etf] = weight
                    total_allocation += weight

                # Allocation validation
                if total_allocation != 100:
                    st.warning(f"{t('invalid_allocation')} Current: {total_allocation}%")
                else:
                    st.success(f"✅ Total allocation: {total_allocation}%")

            st.divider()

            # Advanced Settings
            with st.expander(t('advanced_settings')):
                st.markdown(f"**{t('contribution_increases')}**")

                # Contribution increases over time
                if 'contribution_increases' not in st.session_state:
                    st.session_state.contribution_increases = {}

                col_year, col_increase = st.columns(2)
                with col_year:
                    increase_year = st.number_input(t('year'), 1, investment_horizon, 5)
                with col_increase:
                    increase_percent = st.number_input(t('increase_percent'), 0.0, 50.0, 5.0)

                if st.button(t('add_increase')):
                    st.session_state.contribution_increases[increase_year] = increase_percent
                    st.rerun()

                # Display current increases
                if st.session_state.contribution_increases:
                    for year, increase in st.session_state.contribution_increases.items():
                        st.text(f"Year {year}: +{increase}%")

            # Action Buttons
            col_run, col_opt = st.columns(2)

            with col_run:
                run_projection = st.button(t('run_projection'), type="primary")

            with col_opt:
                if len(investment_etfs) >= 2:
                    optimize_portfolio = st.button(t('optimize_allocation'))
                else:
                    st.info(t('insufficient_etfs'))
                    optimize_portfolio = False

        with results_col:
            if 'run_projection' in locals() and run_projection and investment_etfs and total_allocation == 100:
                st.markdown(f"#### {t('investment_summary')}")

                # Create Investment Plan
                plan = InvestmentPlan(
                    name=plan_name,
                    initial_investment=initial_investment,
                    monthly_contribution=monthly_contribution,
                    investment_horizon_years=investment_horizon,
                    selected_etfs=investment_etfs,
                    allocation_weights=allocation_weights,
                    risk_profile=risk_profile,
                    target_amount=target_amount if target_amount > 0 else None,
                    contribution_increases=st.session_state.contribution_increases if st.session_state.contribution_increases else None
                )

                # Store current plan
                st.session_state.current_plan = plan

                with st.spinner("🚀 Running investment projection..."):
                    try:
                        # Get ETF metrics for selected ETFs
                        etf_data, etf_metrics, benchmark_data, _, _ = fetch_market_data(investment_etfs, [])

                        # Run projection
                        projection_results = investment_planner.create_investment_projection(plan, etf_metrics)

                        if projection_results:
                            st.success(t('projection_completed'))

                            # Display key metrics
                            if 'moderate' in projection_results:
                                moderate_results = projection_results['moderate']
                                metrics = moderate_results['metrics']

                                # KPI Cards
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.markdown(create_kpi_card(
                                        t('projected_final_value'),
                                        f"${metrics['final_value']:,.0f}",
                                        f"+{((metrics['final_value'] / initial_investment) - 1) * 100:.0f}%"
                                    ), unsafe_allow_html=True)

                                with col2:
                                    st.markdown(create_kpi_card(
                                        t('total_contributions'),
                                        f"${metrics['total_contributions']:,.0f}",
                                        None
                                    ), unsafe_allow_html=True)

                                with col3:
                                    st.markdown(create_kpi_card(
                                        t('expected_gains'),
                                        f"${metrics['total_gains']:,.0f}",
                                        f"{((metrics['total_gains'] / metrics['total_contributions']) * 100):.0f}%"
                                    ), unsafe_allow_html=True)

                                with col4:
                                    success_prob = moderate_results.get('success_probability', 0)
                                    st.markdown(create_kpi_card(
                                        t('success_probability'),
                                        f"{success_prob:.0%}",
                                        None
                                    ), unsafe_allow_html=True)

                                # Main projection chart
                                projection_chart = visualizer.create_investment_projection_chart(projection_results,
                                                                                                 plan)
                                if projection_chart:
                                    st.plotly_chart(projection_chart, use_container_width=True)

                                # Scenario Analysis Table
                                st.markdown(f"#### {t('scenario_analysis')}")

                                scenario_data = []
                                for scenario_name, scenario_results in projection_results.items():
                                    if 'metrics' in scenario_results:
                                        sm = scenario_results['metrics']
                                        scenario_data.append({
                                            'Scenario': scenario_name.title(),
                                            'Final Value': f"${sm['final_value']:,.0f}",
                                            'CAGR': f"{sm['cagr']:.1%}",
                                            'Money Multiple': f"{sm['money_multiple']:.1f}x",
                                            'Break-even': f"{sm['break_even_months']} {t('months_short')}"
                                        })

                                if scenario_data:
                                    df_scenarios = pd.DataFrame(scenario_data)
                                    st.dataframe(df_scenarios, use_container_width=True, hide_index=True)

                                # Percentile Analysis
                                st.markdown(f"#### {t('percentile_analysis')}")

                                percentiles = moderate_results.get('percentiles', {})
                                if percentiles:
                                    col1, col2, col3, col4, col5 = st.columns(5)

                                    with col1:
                                        st.metric(t('worst_case'), f"${percentiles.get('p10', 0):,.0f}")
                                    with col2:
                                        st.metric(t('bad_case'), f"${percentiles.get('p25', 0):,.0f}")
                                    with col3:
                                        st.metric(t('expected_case'), f"${percentiles.get('p50', 0):,.0f}")
                                    with col4:
                                        st.metric(t('good_case'), f"${percentiles.get('p75', 0):,.0f}")
                                    with col5:
                                        st.metric(t('best_case'), f"${percentiles.get('p90', 0):,.0f}")

                                # Monte Carlo Simulation Chart
                                if 'simulations' in moderate_results:
                                    monte_carlo_chart = visualizer.create_monte_carlo_simulation_chart(moderate_results)
                                    if monte_carlo_chart:
                                        st.plotly_chart(monte_carlo_chart, use_container_width=True)

                                # DCA vs Lump Sum Analysis
                                st.markdown(f"#### {t('dca_analysis')}")

                                dca_analysis = investment_planner.calculate_dca_advantage(plan, etf_metrics)
                                if dca_analysis:
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.metric(t('lump_sum_final'), f"${dca_analysis['lump_sum_final']:,.0f}")
                                        st.metric(t('dca_final'), f"${dca_analysis['dca_final']:,.0f}")

                                    with col2:
                                        st.metric(t('advantage_amount'), f"${dca_analysis['dca_advantage']:,.0f}")
                                        st.metric(t('advantage_percent'),
                                                  f"{dca_analysis['dca_advantage_percent']:+.1f}%")

                                    # DCA vs Lump Sum Chart
                                    dca_chart = visualizer.create_dca_vs_lumpsum_chart(dca_analysis, plan)
                                    if dca_chart:
                                        st.plotly_chart(dca_chart, use_container_width=True)

                        else:
                            st.error(t('projection_error'))

                    except Exception as e:
                        st.error(f"{t('projection_error')} {e}")

            elif 'optimize_portfolio' in locals() and optimize_portfolio and len(investment_etfs) >= 2:
                st.markdown(f"#### {t('risk_return_optimization')}")

                with st.spinner("🎯 Optimizing portfolio allocation..."):
                    try:
                        # Get ETF metrics
                        etf_data, etf_metrics, _, _, _ = fetch_market_data(investment_etfs, [])

                        if etf_metrics:
                            # Create temporary plan for optimization
                            temp_plan = InvestmentPlan(
                                name="Optimization",
                                initial_investment=initial_investment,
                                monthly_contribution=monthly_contribution,
                                investment_horizon_years=investment_horizon,
                                selected_etfs=investment_etfs,
                                allocation_weights=allocation_weights,
                                risk_profile=risk_profile
                            )

                            # Run optimization
                            optimization_result = investment_planner.optimize_allocation(temp_plan, etf_metrics)

                            if optimization_result and optimization_result['optimization_success']:
                                st.success(t('optimization_completed'))

                                # Display optimal allocation
                                st.markdown(f"**{t('optimal_allocation')}:**")

                                optimal_weights = optimization_result['optimal_weights']
                                for etf, weight in optimal_weights.items():
                                    st.write(f"• {etf}: {weight:.1%}")

                                # Display optimization metrics
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric(t('expected_return_annual'),
                                              f"{optimization_result['expected_return']:.1%}")
                                with col2:
                                    st.metric(t('portfolio_volatility'), f"{optimization_result['volatility']:.1%}")
                                with col3:
                                    st.metric(t('sharpe_ratio_portfolio'), f"{optimization_result['sharpe_ratio']:.2f}")

                                # Risk-Return Optimization Chart
                                optimization_chart = visualizer.create_risk_return_optimization_chart(
                                    optimization_result, etf_metrics)
                                if optimization_chart:
                                    st.plotly_chart(optimization_chart, use_container_width=True)

                            else:
                                st.error("❌ Portfolio optimization failed")

                        else:
                            st.error("❌ Could not fetch ETF metrics for optimization")

                    except Exception as e:
                        st.error(f"❌ Optimization error: {e}")

            else:
                # Show placeholder information
                st.markdown(f"""
                <div class="investment-summary">
                    <h3>💰 Professional Investment Planning</h3>
                    <p>Configure your investment plan in the left panel and click "Run Projection" to see:</p>
                    <ul>
                        <li>📈 Monte Carlo simulation with multiple scenarios</li>
                        <li>🎯 Portfolio optimization using Modern Portfolio Theory</li>
                        <li>📊 Risk-return analysis and percentile projections</li>
                        <li>💰 DCA vs Lump Sum comparison</li>
                        <li>📅 Timeline visualization with contributions vs returns</li>
                        <li>🎲 Success probability analysis</li>
                    </ul>
                    <p><strong>Select at least 2 ETFs and ensure allocation totals 100%</strong></p>
                </div>
                """, unsafe_allow_html=True)

            # Save/Load Plans
            st.divider()
            col_save, col_load = st.columns(2)

            with col_save:
                if st.button(t('save_plan')) and st.session_state.current_plan:
                    plan_data = {
                        'name': st.session_state.current_plan.name,
                        'initial_investment': st.session_state.current_plan.initial_investment,
                        'monthly_contribution': st.session_state.current_plan.monthly_contribution,
                        'investment_horizon_years': st.session_state.current_plan.investment_horizon_years,
                        'selected_etfs': st.session_state.current_plan.selected_etfs,
                        'allocation_weights': st.session_state.current_plan.allocation_weights,
                        'risk_profile': st.session_state.current_plan.risk_profile,
                        'target_amount': st.session_state.current_plan.target_amount,
                        'contribution_increases': st.session_state.current_plan.contribution_increases
                    }

                    if save_investment_plan(plan_data):
                        st.success(t('plan_saved'))

            with col_load:
                saved_plans = load_investment_plans()
                if saved_plans:
                    selected_plan = st.selectbox("Select Plan", options=list(saved_plans.keys()))
                    if st.button(t('load_plan')) and selected_plan:
                        # Load plan logic would go here
                        st.success(t('plan_loaded'))

    # ===============================
    # TAB 5-7: PLACEHOLDER TABS
    # ===============================

    with tab5:
        st.markdown(f"### {t('tab_market_analysis')}")
        st.info(t('market_analysis_soon'))

    with tab6:
        st.markdown(f"### {t('tab_portfolio')}")
        st.info(t('portfolio_manager_soon'))

    with tab7:
        st.markdown(f"### {t('tab_ai_insights')}")
        st.info(t('ai_insights_soon'))

    # ===============================
    # FOOTER
    # ===============================

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚀 <strong>{t('footer_title')}</strong></p>
        <p>{t('footer_subtitle')}</p>
        <p><em>{t('footer_disclaimer')}</em></p>
    </div>
    """, unsafe_allow_html=True)


# ===============================
# RUN APPLICATION
# ===============================

if __name__ == "__main__":
    main()