import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


st.success(
    """
    ### Steps to Create Portfolio
    1. Click on the **Show sectors** tab.
    2. Select an industry, check the stocks by clicking on checkboxes, and click **Submit** for that industry to add them.
    3. You will see all the selected stocks listed below.
    4. Select dates to consider returns of stocks.
    5. Click **Submit** to build the portfolio.
    """
)

# Load the CSV file
try:
    Read_data = pd.read_csv('ind_nifty200list.csv')
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
    st.stop()

# Initialize session state for storing selected stocks
if "final_selected_data" not in st.session_state:
    st.session_state.final_selected_data = []

# Create a dictionary for each industry
industry_dfs = {}
for industry in Read_data["Industry"].unique():
    industry_dfs[industry] = Read_data[Read_data["Industry"] == industry]

# Top-level dropdown for Nifty 50
if st.checkbox("Show Sectors"):
    # Dropdown to select an industry
    industry_selection = st.selectbox(
        "Select an Industry",
        options=["Select Industry"] + list(Read_data["Industry"].unique())
    )
    
    if industry_selection != "Select Industry":
        # Display the selected industry's stocks in tabular form with checkboxes
        st.subheader(f"Select Stocks from the {industry_selection} Industry")
        industry_df = industry_dfs[industry_selection].reset_index(drop=True)  # Reset index for display
        
        # Track selected stocks
        selected_data = []

        # Display table with checkboxes
        for index, row in industry_df.iterrows():
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(row['Company Name'])  # Company Name
            with col2:
                st.write(row['Symbol'])  # Symbol
            with col3:
                # Checkbox for selection
                checkbox_key = f"checkbox_{industry_selection}_{index}"  # Unique key
                if st.checkbox("Select", key=checkbox_key):
                    selected_data.append(row['Symbol'])
        
        # Submit button for this industry
        if st.button(f"Submit for {industry_selection}"):
            # Add unique stocks to the session state list
            st.session_state.final_selected_data = list(
                set(st.session_state.final_selected_data + selected_data)
            )
            st.success(f"Added selected stocks from {industry_selection} to the final list!")

# Display the final list of selected stocks
if st.session_state.final_selected_data:
    st.subheader("Final Selected Stocks")
    st.dataframe(pd.DataFrame({"Selected Stocks": st.session_state.final_selected_data}))


## date selection 
# Date selection for the 'from' and 'to' dates
start = st.date_input(
    "Select From Date",
    min_value=datetime(2015, 1, 1),  # Min date for the 'from' date
    max_value=datetime(2020, 12, 31),  # Max date for the 'from' date
    value=datetime(2015, 1, 1)  # Default value
)

end = st.date_input(
    "Select To Date",
    min_value=start,  # 'To' date cannot be earlier than 'From' date
    max_value=datetime(2020, 12, 31),  # Max date for the 'to' date
    value=datetime(2020, 12, 31)  # Default value
)

st.write('***************************************************************************** ')


# Display the selected dates
st.write(f"Selected From Date: {start}")
st.write(f"Selected To Date: {end}")




# Add a submit button to trigger the downloading of data
if st.button('Submit to Build Portfolio'):
    # Fetch data only if final_selected_data is not empty
    if st.session_state.final_selected_data:
        st.subheader("Fetching Data ....")

        # Initialize an empty DataFrame to store data
        selected_data_df = pd.DataFrame()

        # Loop through each stock in the final_selected_data list
        for stock in st.session_state.final_selected_data:
            try:
                # Fetch data for each stock
                #st.info(f"Fetching data for {stock}...")
                stock_data = yf.download(stock + '.NS', start=start, end=end)  # Adjust date range as needed

                # Add adjusted close prices to the selected_data DataFrame
                selected_data_df[stock] = stock_data['Adj Close']
                selected_data_df.dropna(inplace=True)


                # Calculate daily price change
                selected_data_df[f"{stock}_pct_change"] = selected_data_df[stock].pct_change()  # calculates percentage change
                
            except Exception as e:
                st.warning(f"Could not fetch data for {stock}: {e}")

        # Display the fetched data or a warning if empty
        if not selected_data_df.empty:
            st.write("Data fetched for selected stocks!")
            #st.dataframe(selected_data_df)  # Display the data in a table
        else:
            st.write("No data available!")

else:
    st.warning("No stocks selected! Please select stocks to fetch data.")




# Check if data is available for correlation matrix calculation
if not selected_data_df.empty:
    # Filter columns with percentage change data
    pct_change_columns = [col for col in selected_data_df.columns if col.endswith("_pct_change")]
    daily_returns = selected_data_df[pct_change_columns]

    # Rename columns to remove "_pct_change" suffix for cleaner display
    daily_returns.columns = [col.replace("_pct_change", "") for col in daily_returns.columns]

    # Calculate the correlation matrix
    correlation_matrix = daily_returns.corr()

    # Display the correlation matrix as a heatmap
    st.title("Stocks Correlation Heatmap")
    fig = px.imshow(
        correlation_matrix,
        width=850,
        height=850,
        color_continuous_scale='RdBu_r',
        
    )
    st.plotly_chart(fig, use_container_width=True)
st.success('Heatmap shows tht correlation between the stocks selected. Low corellation shows portfolio is well diversified')


# Check if stocks have been selected and data has been fetched
if st.session_state.final_selected_data:
    # Fetch the portfolio returns dataframe (already fetched in the earlier part of the code)
    portfolio_stocks_returns = selected_data_df.pct_change().dropna()  # Assuming `selected_data_df` contains adjusted close prices

    # Ensure that the portfolio returns only include the selected stocks
    selected_columns = [stock for stock in st.session_state.final_selected_data if stock in portfolio_stocks_returns.columns]
    portfolio_stocks_returns = portfolio_stocks_returns[selected_columns]

    # Calculate annual mean returns (assuming 252 trading days)
    annual_mean_returns = portfolio_stocks_returns.mean() * 252

    # Calculate the annual variance-covariance matrix
    annual_cov_matrix = portfolio_stocks_returns.cov() * 252

    np.random.seed(11)

    # Generate random portfolio weights (10000 portfolios)
    weights = np.random.dirichlet(np.ones(len(st.session_state.final_selected_data)), size=10000)

    # Create a DataFrame to store portfolio data
    port_data = pd.DataFrame(columns=st.session_state.final_selected_data + ['expected_returns', 'expected_volatility'])

    # Compute the (annual) expected returns for all portfolios
    all_portfolio_expected_returns = np.dot(weights, annual_mean_returns)

    # Compute the (annual) expected risk for all portfolios
    all_portfolio_expected_risk = [np.sqrt(x.T.dot(annual_cov_matrix).dot(x)) for x in weights]

    # Store each weight vector in the dataframe
    for i in range(0, len(st.session_state.final_selected_data)):
        port_data[st.session_state.final_selected_data[i]] = weights[:, i] * 100

    # Store risk and return profile for each portfolio
    port_data['expected_returns'] = all_portfolio_expected_returns
    port_data['expected_volatility'] = np.array(all_portfolio_expected_risk)

    # Display the portfolio data (portfolio weights, returns, and volatility)
    st.subheader("10000 sets of portfolio")
    st.success("The table shows stocks with diffrent weights with the portfolio's expected returns and expected volatility" )
    st.dataframe(port_data)

    
import plotly.express as px

# Create a new column with maximum returns per unit of risk (Sharpe ratio)
port_data['Sharpe ratio'] = port_data['expected_returns'] / port_data['expected_volatility']
st.success("Maximum sharpe ratio portfolio is considered to be the best")

# Find the portfolio with maximum returns per unit of risk
new_sharpe_portfolios = port_data.loc[port_data['Sharpe ratio'] == port_data['Sharpe ratio'].max()]

# Display the portfolio with the maximum Sharpe ratio
st.subheader("Portfolio with Maximum Sharpe Ratio")
st.write(new_sharpe_portfolios)  # Display the portfolios in a readable format

# Find the portfolio with minimum risk (lowest volatility)
min_risk_portfolios = port_data.loc[port_data['expected_volatility'] == port_data['expected_volatility'].min()]

# Extract X and Y coordinates for plotting - Minimum Risk Portfolio
min_x = min_risk_portfolios.expected_volatility.iloc[0]
min_y = min_risk_portfolios.expected_returns.iloc[0]

# Extract X and Y coordinates for plotting - Maximum Sharpe Portfolio
sharpe_x = new_sharpe_portfolios.expected_volatility.iloc[0]
sharpe_y = new_sharpe_portfolios.expected_returns.iloc[0]

# Display the extracted coordinates for the Minimum Risk Portfolio
st.subheader("Minimum Risk Portfolio Coordinates")
st.write(f"Minimum Expected Volatility : {min_x}")
st.write(f"Minimum Expected Returns : {min_y}")

# Display the extracted coordinates for the Maximum Sharpe Portfolio
st.subheader("Maximum Sharpe Portfolio Coordinates")
st.write(f"Maximum Expected Volatility : {sharpe_x}")
st.write(f"Maximum Expected Returns : {sharpe_y}")


import plotly.graph_objects as go

# Create a figure
fig = go.Figure()

# Add trace for portfolios
fig.add_trace(go.Scatter(
    x=port_data['expected_volatility'],
    y=port_data['expected_returns'],
    mode="markers+text",
    marker=dict(color=port_data['expected_returns']),
    name='Portfolio',
    showlegend=False
))

# Add trace for Minimum Volatility Portfolio
fig.add_trace(go.Scatter(
    x=[min_x],
    y=[min_y],
    mode="markers",
    marker=dict(color='SkyBlue', size=10),
    marker_symbol='star',
    name='Minimum Volatility Portfolio',
    showlegend=False
))

# Add trace for Maximum Sharpe Portfolio
fig.add_trace(go.Scatter(
    x=[sharpe_x],
    y=[sharpe_y],
    mode="markers",
    marker=dict(color='Green', size=10),
    marker_symbol='cross',
    name='Maximum Sharpe Portfolio',
    showlegend=False
))

# Update layout
fig.update_layout(
    title='Various Portfolios',
    xaxis_title='Expected Volatility',
    yaxis_title='Expected Returns'
)

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)


# Sort portfolios by max returns per unit of risk (Sharpe ratio) and get top 10
top_sharpe_portfolios = port_data.sort_values(by='Sharpe ratio', ascending=False).head(10)

# Display the top 10 Sharpe ratio portfolios in Streamlit
st.subheader("Top 10 Portfolios by Sharpe Ratio")
st.dataframe(top_sharpe_portfolios)

# Calculate maximum and minimum expected returns
max_expected_returns = port_data['expected_returns'].max()
min_expected_returns = port_data['expected_returns'].min()

# Calculate maximum and minimum expected volatility
max_expected_volatility = port_data['expected_volatility'].max()
min_expected_volatility = port_data['expected_volatility'].min()

# Display the results in Streamlit
st.subheader("Portfolio Metrics:")

st.write(f"**Max Expected Returns:** {max_expected_returns:.4f}")
st.write(f"**Min Expected Returns:** {min_expected_returns:.4f}")
st.write(f"**Max Expected Volatility:** {max_expected_volatility:.4f}")
st.write(f"**Min Expected Volatility:** {min_expected_volatility:.4f}")



##                   Backtesting              ##
st.subheader("Backtesting Portfolio")
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.success("Let's compare the portfolio with Nifty as benchmark and same stocks folio with equal weights with out of sample dates")

# Function to download stock data and calculate returns
def download_stock_data(stock_list, start_date, end_date):
    backtest_df = pd.DataFrame()
    for stock in stock_list:
        data = yf.download(stock + '.NS', start=start_date, end=end_date)
        data['returns'] = data['Adj Close'].pct_change()
        data.dropna(inplace=True)
        backtest_df[stock] = data['returns']
    return backtest_df

# Check if stocks are selected
if "final_selected_data" not in st.session_state or not st.session_state.final_selected_data:
    st.warning("No stocks selected. Please select stocks first.")
    st.stop()

# Date range for backtest
start_date = '2021-01-01'
end_date = datetime.today() 
st.success( f"We are backtesting portfolio from {start_date} to {end_date}")
# Download stock data for the selected stocks
#st.info("Downloading data for selected stocks...")
backtest_df = download_stock_data(st.session_state.final_selected_data, start_date, end_date)

# Download Nifty 50 data for the same time frame
nifty50_data = yf.download("^NSEI", start=start_date, end=end_date)
nifty50_data['returns'] = nifty50_data['Adj Close'].pct_change()

# Calculate cumulative returns for Nifty 50
cumulative_returns_nifty50 = (1 + nifty50_data['returns']).cumprod() - 1

# Define equal allocation for the portfolio
equal_weights = 1 / len(st.session_state.final_selected_data)

# Assuming you have the optimized weights from portfolio optimization
# Replace this with the actual optimized weights from your model
optimized_weights = [0.2] * len(st.session_state.final_selected_data)  # Example optimized weights (equal for now)

# Calculate portfolio returns with equal allocation
backtest_df['equal_allocation_returns'] = (backtest_df * equal_weights).sum(axis=1)

# Calculate cumulative returns for equal allocation
cumulative_returns_equal_allocation = (1 + backtest_df['equal_allocation_returns']).cumprod() - 1

# Calculate portfolio returns with optimized allocation
backtest_df['optimized_returns'] = (backtest_df[st.session_state.final_selected_data] * optimized_weights).sum(axis=1)

# Calculate cumulative returns for optimized allocation
cumulative_returns_optimized = (1 + backtest_df['optimized_returns']).cumprod() - 1

# Plotting the results
fig = go.Figure()

# Add Equal Allocation Portfolio
fig.add_trace(go.Scatter(x=cumulative_returns_equal_allocation.index,
                         y=cumulative_returns_equal_allocation * 100,  # Convert to percentage
                         mode='lines',
                         name='Equal Allocation Portfolio'))

# Add Optimized Portfolio
fig.add_trace(go.Scatter(x=cumulative_returns_optimized.index,
                         y=cumulative_returns_optimized * 100,  # Convert to percentage
                         mode='lines',
                         name='Optimized Portfolio'))

# Add Nifty 50 Benchmark
fig.add_trace(go.Scatter(x=cumulative_returns_nifty50.index,
                         y=cumulative_returns_nifty50 * 100,  # Convert to percentage
                         mode='lines',
                         name='Nifty 50 Benchmark'))

# Customize layout
fig.update_layout(
    title="Comparison of Equal Allocation vs Optimized Portfolio",
    xaxis_title="Date",
    yaxis_title="Cumulative Returns (%)",
    legend_title="Portfolio Type",
    template="plotly_white"
)

# Display the plot
st.plotly_chart(fig)

# Display the final returns of the portfolios and Nifty 50
st.write('Cumulative returns of a portfolio with equal allocations is %.3f%%' % (cumulative_returns_equal_allocation[-1] * 100))
st.write('Cumulative returns of a portfolio with optimal allocations is %.3f%%' % (cumulative_returns_optimized[-1] * 100))
st.write('Nifty 50 returns for the period is %.3f%%' % (cumulative_returns_nifty50[-1] * 100))

