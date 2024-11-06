import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize

# Set page config
st.set_page_config(layout="wide")

# Utility Functions
def get_stock_data(symbols, start_date, end_date):
    """Fetch stock data using yfinance"""
    df_list = []
    for symbol in symbols:
        try:
            stock = yf.download(symbol, start=start_date, end=end_date)
            if not stock.empty:
                df_list.append(stock['Adj Close'])
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
    if df_list:
        df = pd.concat(df_list, axis=1)
        df.columns = symbols
        return df
    return None

def calculate_returns_volatility(weights, returns):
    """Calculate portfolio return and volatility"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_vol

def calculate_sharpe_ratio(weights, returns, risk_free_rate):
    """Calculate Sharpe ratio"""
    portfolio_return, portfolio_vol = calculate_returns_volatility(weights, returns)
    return (portfolio_return - risk_free_rate) / portfolio_vol

def simulate_portfolio(returns, weights, initial_investment, num_simulations, num_days):
    """Run Monte Carlo simulation"""
    portfolio_simulations = np.zeros((num_simulations, num_days))
    
    for sim in range(num_simulations):
        portfolio_simulations[sim, 0] = initial_investment
        
        for day in range(1, num_days):
            daily_returns = np.random.multivariate_normal(
                returns.mean(), 
                returns.cov(), 
                1
            )
            portfolio_return = np.sum(daily_returns * weights)
            portfolio_simulations[sim, day] = portfolio_simulations[sim, day-1] * (1 + portfolio_return)
    
    return portfolio_simulations

def portfolio_creation_tab():
    st.header("Portfolio Creation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stocks_input = st.text_area(
            'Enter stock symbols (comma-separated)',
            'AAPL,GOOGL,MSFT,AMZN',
            key='stock_input'
        )
        symbols = [s.strip() for s in stocks_input.split(',')]
        
        start_date = st.date_input(
            'Start Date',
            datetime.now() - timedelta(days=365),
            key='start_date'
        )
        end_date = st.date_input('End Date', datetime.now(), key='end_date')
    
    if st.button('Create Portfolio', key='create_portfolio'):
        with st.spinner('Fetching stock data...'):
            prices_df = get_stock_data(symbols, start_date, end_date)
            
            if prices_df is not None:
                st.session_state['prices_df'] = prices_df
                st.session_state['symbols'] = symbols
                
                fig = px.line(prices_df, title='Historical Prices')
                st.plotly_chart(fig, use_container_width=True)
                
                returns = prices_df.pct_change().dropna()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader('Annual Returns')
                    annual_returns = (returns.mean() * 252 * 100).round(2)
                    st.dataframe(pd.DataFrame({'Annual Return (%)': annual_returns}))
                
                with col2:
                    st.subheader('Annual Volatility')
                    annual_vol = (returns.std() * np.sqrt(252) * 100).round(2)
                    st.dataframe(pd.DataFrame({'Annual Volatility (%)': annual_vol}))
                
                st.subheader('Correlation Matrix')
                corr = returns.corr()
                fig = px.imshow(corr, 
                              labels=dict(color="Correlation"),
                              color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)

def sharpe_ratio_calculator_tab():
    st.header("Sharpe Ratio Calculator")
    
    if 'prices_df' not in st.session_state:
        st.warning('Please create a portfolio first in the Portfolio Creation tab.')
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_free_rate = st.number_input(
            'Risk-free rate (%)', 
            value=2.0,
            key='sharpe_rfr'
        ) / 100
        
        weights = []
        total_weight = 0
        
        for symbol in st.session_state['symbols']:
            remaining_weight = 100 - total_weight
            max_weight = min(100, remaining_weight + 1)
            
            weight = st.slider(
                f'{symbol} weight (%)', 
                0, 
                max_weight,
                value=min(max_weight, 100 // len(st.session_state['symbols'])),
                key=f'weight_{symbol}'
            )
            weights.append(weight / 100)
            total_weight += weight
        
        weights = np.array(weights) / np.sum(weights)
    
    returns = st.session_state['prices_df'].pct_change().dropna()
    portfolio_return, portfolio_vol = calculate_returns_volatility(weights, returns)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
    
    with col2:
        st.subheader('Portfolio Metrics')
        metrics_df = pd.DataFrame({
            'Metric': ['Expected Annual Return', 'Annual Volatility', 'Sharpe Ratio'],
            'Value': [
                f'{portfolio_return:.2%}',
                f'{portfolio_vol:.2%}',
                f'{sharpe_ratio:.2f}'
            ]
        })
        st.dataframe(metrics_df, hide_index=True)
        
        fig = go.Figure(data=[go.Pie(
            labels=st.session_state['symbols'],
            values=weights,
            title='Portfolio Allocation'
        )])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader('Risk-Return Analysis')
        risk_return_df = pd.DataFrame({
            'Volatility': [],
            'Return': [],
            'Type': []
        })
        
        for _ in range(1000):
            w = np.random.random(len(weights))
            w = w / np.sum(w)
            r, v = calculate_returns_volatility(w, returns)
            risk_return_df = pd.concat([
                risk_return_df,
                pd.DataFrame({
                    'Volatility': [v],
                    'Return': [r],
                    'Type': ['Random']
                })
            ])
        
        risk_return_df = pd.concat([
            risk_return_df,
            pd.DataFrame({
                'Volatility': [portfolio_vol],
                'Return': [portfolio_return],
                'Type': ['Current']
            })
        ])
        
        fig = px.scatter(
            risk_return_df,
            x='Volatility',
            y='Return',
            color='Type',
            title='Risk-Return Profile'
        )
        st.plotly_chart(fig, use_container_width=True)

def monte_carlo_tab():
    st.header("Monte Carlo Simulation")
    
    if 'prices_df' not in st.session_state:
        st.warning('Please create a portfolio first in the Portfolio Creation tab.')
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        initial_investment = st.number_input(
            'Initial Investment ($)', 
            value=10000,
            key='mc_initial_investment'
        )
        num_simulations = st.slider(
            'Number of Simulations', 
            100, 1000, 500,
            key='mc_num_simulations'
        )
        num_days = st.slider(
            'Number of Trading Days', 
            30, 252, 252,
            key='mc_num_days'
        )
        
        weights = np.array([1/len(st.session_state['symbols'])] * len(st.session_state['symbols']))
    
    returns = st.session_state['prices_df'].pct_change().dropna()
    
    if st.button('Run Simulation', key='run_simulation'):
        with st.spinner('Running simulations...'):
            simulations = simulate_portfolio(
                returns, weights, initial_investment, num_simulations, num_days
            )
        
            fig = go.Figure()
            
            for i in range(num_simulations):
                fig.add_trace(go.Scatter(
                    y=simulations[i],
                    mode='lines',
                    opacity=0.1,
                    line=dict(color='blue'),
                    showlegend=False
                ))
            
            percentiles = [5, 50, 95]
            colors = ['red', 'green', 'red']
            
            for p, color in zip(percentiles, colors):
                fig.add_trace(go.Scatter(
                    y=np.percentile(simulations, p, axis=0),
                    mode='lines',
                    name=f'{p}th Percentile',
                    line=dict(color=color, width=2)
                ))
            
            fig.update_layout(
                title='Monte Carlo Simulation Results',
                xaxis_title='Trading Days',
                yaxis_title='Portfolio Value ($)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            final_values = simulations[:, -1]
            
            stats_df = pd.DataFrame({
                'Metric': [
                    'Expected Portfolio Value',
                    '5th Percentile (95% VaR)',
                    '95th Percentile (Best Case)'
                ],
                'Value': [
                    f'${np.mean(final_values):,.2f}',
                    f'${np.percentile(final_values, 5):,.2f}',
                    f'${np.percentile(final_values, 95):,.2f}'
                ]
            })
            
            st.dataframe(stats_df, hide_index=True)

def mathematical_optimization_tab():
    st.header("Mathematical Optimization")
    
    if 'prices_df' not in st.session_state:
        st.warning('Please create a portfolio first in the Portfolio Creation tab.')
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimization_objective = st.selectbox(
            'Optimization Objective',
            ['Maximum Sharpe Ratio', 'Minimum Volatility', 'Maximum Return'],
            key='opt_objective'
        )
        
        risk_free_rate = st.number_input(
            'Risk-free rate (%)', 
            value=2.0,
            key='opt_rfr'
        ) / 100
        
        min_weight = st.number_input(
            'Minimum Weight (%)', 
            value=5,
            key='opt_min_weight'
        ) / 100
        
        max_weight = st.number_input(
            'Maximum Weight (%)', 
            value=40,
            key='opt_max_weight'
        ) / 100
    
    returns = st.session_state['prices_df'].pct_change().dropna()
    
    def objective(weights):
        portfolio_return, portfolio_vol = calculate_returns_volatility(weights, returns)
        
        if optimization_objective == 'Maximum Sharpe Ratio':
            return -(portfolio_return - risk_free_rate) / portfolio_vol
        elif optimization_objective == 'Minimum Volatility':
            return portfolio_vol
        else:  # Maximum Return
            return -portfolio_return
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    ]
    
    bounds = [(min_weight, max_weight) for _ in range(len(st.session_state['symbols']))]
    
    if st.button('Optimize Portfolio', key='optimize_portfolio'):
        with st.spinner('Optimizing portfolio...'):
            n_assets = len(st.session_state['symbols'])
            initial_weights = np.array([1/n_assets] * n_assets)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if not result.success:
                st.error('Optimization failed. Try adjusting the constraints.')
                return
            
            optimal_weights = result.x
            portfolio_return, portfolio_vol = calculate_returns_volatility(optimal_weights, returns)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
            
            st.subheader('Optimization Results')
            
            weights_df = pd.DataFrame({
                'Stock': st.session_state['symbols'],
                'Weight (%)': (optimal_weights * 100).round(2)
            })
            st.write('Optimal Weights:')
            st.dataframe(weights_df, hide_index=True)
            
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Expected Annual Return',
                    'Annual Volatility',
                    'Sharpe Ratio'
                ],
                'Value': [
                    f'{portfolio_return:.2%}',
                    f'{portfolio_vol:.2%}',
                    f'{sharpe_ratio:.2f}'
                ]
            })
            st.dataframe(metrics_df, hide_index=True)
            
            fig = go.Figure(data=[go.Pie(
                labels=st.session_state['symbols'],
                values=optimal_weights,
                title='Optimal Portfolio Allocation'
            )])
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.title('Portfolio Analysis Dashboard')
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Portfolio Creation",
        "Sharpe Ratio Calculator",
        "Monte Carlo Simulation",
        "Mathematical Optimization"
    ])
    
    with tab1:
        portfolio_creation_tab()
    
    with tab2:
        sharpe_ratio_calculator_tab()
    
    with tab3:
        monte_carlo_tab()
    
    with tab4:
        mathematical_optimization_tab()

if __name__ == '__main__':
    main()