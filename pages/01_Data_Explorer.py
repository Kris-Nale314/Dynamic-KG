"""
Dynamic-KG Data Explorer

A Streamlit application for exploring the collected financial and market data.
Place this file in the pages directory and run with:
streamlit run pages/01_data_explorer.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
import json

# Add the project root directory to the Python path for importing config
root_dir = str(Path(__file__).parents[1].absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the config from the root
from config import Config

# Initialize configuration
Config.initialize()

# Set page config
st.set_page_config(
    page_title="Dynamic-KG Data Explorer",
    page_icon="📊",
    layout="wide"
)

# Page header
st.title("Dynamic-KG Data Explorer")
st.markdown("""
This application allows you to explore the financial and market data collected for the Dynamic-KG project.
Select different data types and companies to visualize and analyze the data.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
data_type = st.sidebar.selectbox(
    "Select Data Type",
    ["Company Profiles", "Financial Statements", "Stock Prices", "Market Indicators", "Material Events (8-Ks)"]
)

# Get available companies
company_dirs = [d.name for d in Config.COMPANIES_DIR.glob("*") if d.is_dir()]
company_dirs.sort()

# Company explorer
if data_type == "Company Profiles":
    st.header("Company Profiles")
    
    # Select company
    selected_company = st.selectbox("Select Company", company_dirs)
    
    # Load company profile
    profile_path = Config.COMPANIES_DIR / selected_company / "profile.json"
    
    if profile_path.exists():
        with open(profile_path, "r") as f:
            profile = json.load(f)
        
        # Display company info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(profile.get("companyName", selected_company))
            st.markdown(f"**Exchange:** {profile.get('exchange', 'N/A')}")
            st.markdown(f"**Industry:** {profile.get('industry', 'N/A')}")
            st.markdown(f"**Sector:** {profile.get('sector', 'N/A')}")
            st.markdown(f"**Currency:** {profile.get('currency', 'USD')}")
            st.markdown(f"**Website:** [{profile.get('website', 'N/A')}]({profile.get('website', '#')})")
        
        with col2:
            st.markdown(f"**Market Cap:** ${profile.get('mktCap', 0):,.0f}")
            st.markdown(f"**Price:** ${profile.get('price', 0):,.2f}")
            st.markdown(f"**Volume:** {profile.get('volAvg', 0):,.0f}")
            st.markdown(f"**CEO:** {profile.get('ceo', 'N/A')}")
            st.markdown(f"**Employees:** {profile.get('fullTimeEmployees', 'N/A'):,}")
        
        # Company description
        st.subheader("Company Description")
        st.markdown(profile.get("description", "No description available."))
    else:
        st.error(f"No profile data found for {selected_company}")

# Financial statements explorer
elif data_type == "Financial Statements":
    st.header("Financial Statements")
    
    # Select company
    selected_company = st.selectbox("Select Company", company_dirs)
    
    # Load financial data
    financials_path = Config.COMPANIES_DIR / selected_company / "financials.parquet"
    
    if financials_path.exists():
        financials = pd.read_parquet(financials_path)
        
        # Select statement type and period
        col1, col2 = st.columns(2)
        
        with col1:
            statement_type = st.selectbox(
                "Select Statement Type",
                financials['statement_type'].unique()
            )
        
        with col2:
            period_type = st.selectbox(
                "Select Period Type",
                financials['period_type'].unique()
            )
        
        # Filter data
        filtered_data = financials[
            (financials['statement_type'] == statement_type) &
            (financials['period_type'] == period_type)
        ]
        
        if not filtered_data.empty:
            # Sort by date
            if 'date' in filtered_data.columns:
                filtered_data = filtered_data.sort_values('date')
            
            # Extract key metrics based on statement type
            if statement_type == "Income Statement":
                metrics = ["revenue", "grossProfit", "netIncome", "operatingIncome"]
            elif statement_type == "Balance Sheet":
                metrics = ["totalAssets", "totalLiabilities", "totalEquity", "cashAndCashEquivalents"]
            elif statement_type == "Cash Flow":
                metrics = ["netCashProvidedByOperatingActivities", "netCashUsedForInvestingActivites", "netCashUsedProvidedByFinancingActivities"]
            elif statement_type == "Key Metrics":
                metrics = ["revenuePerShare", "netIncomePerShare", "operatingCashFlowPerShare", "freeCashFlowPerShare"]
            elif statement_type == "Financial Ratios":
                metrics = ["currentRatio", "quickRatio", "debtToEquity", "returnOnEquity"]
            else:
                metrics = []
            
            # Filter available metrics
            available_metrics = [m for m in metrics if m in filtered_data.columns]
            
            if available_metrics:
                selected_metrics = st.multiselect(
                    "Select Metrics to Display",
                    available_metrics,
                    default=available_metrics[:2]
                )
                
                if selected_metrics:
                    # Prepare data for plotting
                    plot_data = filtered_data[['date'] + selected_metrics].copy()
                    plot_data.set_index('date', inplace=True)
                    
                    # Create line chart
                    fig = px.line(
                        plot_data,
                        x=plot_data.index,
                        y=selected_metrics,
                        title=f"{selected_company} - {statement_type} ({period_type})"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display data table
                    st.subheader("Data Table")
                    st.dataframe(plot_data[selected_metrics])
            else:
                st.warning("No suitable metrics found for selected statement type.")
        else:
            st.warning(f"No {statement_type} ({period_type}) data found for {selected_company}")
    else:
        st.error(f"No financial data found for {selected_company}")

# Stock prices explorer
elif data_type == "Stock Prices":
    st.header("Stock Prices")
    
    # Select company
    selected_company = st.selectbox("Select Company", company_dirs)
    
    # Load price data
    prices_path = Config.COMPANIES_DIR / selected_company / "prices.parquet"
    
    if prices_path.exists():
        prices = pd.read_parquet(prices_path)
        
        # Ensure date column is datetime
        if 'date' in prices.columns:
            prices['date'] = pd.to_datetime(prices['date'])
        
        # Date range selection
        min_date = prices['date'].min().date()
        max_date = prices['date'].max().date()
        
        date_range = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        
        # Filter data by date range
        filtered_prices = prices[
            (prices['date'].dt.date >= date_range[0]) &
            (prices['date'].dt.date <= date_range[1])
        ]
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=filtered_prices['date'],
            open=filtered_prices['open'],
            high=filtered_prices['high'],
            low=filtered_prices['low'],
            close=filtered_prices['close'],
            name='Price'
        )])
        
        fig.update_layout(
            title=f"{selected_company} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        volume_fig = px.bar(
            filtered_prices,
            x='date',
            y='volume',
            title=f"{selected_company} Trading Volume"
        )
        
        st.plotly_chart(volume_fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        # Calculate returns
        filtered_prices['daily_return'] = filtered_prices['close'].pct_change()
        
        with col1:
            st.metric(
                "Average Daily Volume",
                f"{filtered_prices['volume'].mean():,.0f}"
            )
        
        with col2:
            st.metric(
                "Average Daily Range",
                f"${(filtered_prices['high'] - filtered_prices['low']).mean():.2f}"
            )
        
        with col3:
            st.metric(
                "Average Daily Return",
                f"{filtered_prices['daily_return'].mean() * 100:.2f}%"
            )
        
        # Price table
        st.subheader("Price Data")
        st.dataframe(filtered_prices.sort_values('date', ascending=False).head(20))
    else:
        st.error(f"No price data found for {selected_company}")

# Market indicators explorer
elif data_type == "Market Indicators":
    st.header("Market Indicators")
    
    # Load economic data
    economic_path = Config.MARKET_DIR / "economic.parquet"
    treasury_path = Config.MARKET_DIR / "treasury.parquet"
    indices_path = Config.MARKET_DIR / "indices.parquet"
    
    tab1, tab2, tab3 = st.tabs(["Economic Indicators", "Treasury Rates", "Market Indices"])
    
    with tab1:
        if economic_path.exists():
            economic_data = pd.read_parquet(economic_path)
            
            # Get unique indicators
            indicators = economic_data['indicator'].unique().tolist()
            
            # Select indicators
            selected_indicators = st.multiselect(
                "Select Economic Indicators",
                indicators,
                default=indicators[:3] if indicators else []
            )
            
            if selected_indicators:
                # Prepare data
                plot_data = pd.DataFrame()
                
                for indicator in selected_indicators:
                    indicator_data = economic_data[economic_data['indicator'] == indicator].copy()
                    if 'date' in indicator_data.columns:
                        indicator_data['date'] = pd.to_datetime(indicator_data['date'])
                        indicator_data.sort_values('date', inplace=True)
                        
                        # Create new column with indicator name
                        if 'value' in indicator_data.columns:
                            plot_data[indicator] = indicator_data['value']
                            plot_data['date'] = indicator_data['date']
                
                if not plot_data.empty:
                    # Create plot
                    fig = px.line(
                        plot_data,
                        x='date',
                        y=selected_indicators,
                        title="Economic Indicators"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid data found for selected indicators.")
            else:
                st.info("Please select at least one indicator.")
        else:
            st.error("No economic data found.")
    
    with tab2:
        if treasury_path.exists():
            treasury_data = pd.read_parquet(treasury_path)
            
            # Convert date to datetime
            if 'date' in treasury_data.columns:
                treasury_data['date'] = pd.to_datetime(treasury_data['date'])
            
            # Date range selection
            date_range = st.slider(
                "Select Date Range",
                min_value=treasury_data['date'].min().date(),
                max_value=treasury_data['date'].max().date(),
                value=(
                    treasury_data['date'].max().date() - pd.Timedelta(days=365),
                    treasury_data['date'].max().date()
                ),
                format="YYYY-MM-DD",
                key="treasury_date_slider"
            )
            
            # Filter data
            filtered_treasury = treasury_data[
                (treasury_data['date'].dt.date >= date_range[0]) &
                (treasury_data['date'].dt.date <= date_range[1])
            ]
            
            # Get rate columns
            rate_columns = [col for col in treasury_data.columns if col not in ['date']]
            
            # Select rates
            selected_rates = st.multiselect(
                "Select Treasury Rates",
                rate_columns,
                default=rate_columns[:3] if rate_columns else []
            )
            
            if selected_rates:
                # Create plot
                fig = px.line(
                    filtered_treasury,
                    x='date',
                    y=selected_rates,
                    title="Treasury Rates"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display yield curve
                st.subheader("Treasury Yield Curve")
                
                # Get the latest date
                latest_date = filtered_treasury['date'].max()
                latest_data = filtered_treasury[filtered_treasury['date'] == latest_date]
                
                if not latest_data.empty:
                    # Map column names to maturities in months (approximate)
                    maturity_mapping = {
                        'year1': 12,
                        'year2': 24,
                        'year3': 36,
                        'year5': 60,
                        'year7': 84,
                        'year10': 120,
                        'year20': 240,
                        'year30': 360
                    }
                    
                    # Create data for yield curve
                    yield_data = []
                    for col, months in maturity_mapping.items():
                        if col in latest_data.columns:
                            yield_data.append({
                                'Maturity (Months)': months,
                                'Yield (%)': latest_data[col].iloc[0]
                            })
                    
                    yield_df = pd.DataFrame(yield_data)
                    
                    if not yield_df.empty:
                        yield_fig = px.line(
                            yield_df,
                            x='Maturity (Months)',
                            y='Yield (%)',
                            title=f"Treasury Yield Curve ({latest_date.date()})"
                        )
                        
                        st.plotly_chart(yield_fig, use_container_width=True)
            else:
                st.info("Please select at least one rate.")
        else:
            st.error("No treasury data found.")
    
    with tab3:
        if indices_path.exists():
            indices_data = pd.read_parquet(indices_path)
            
            # Convert date to datetime
            if 'date' in indices_data.columns:
                indices_data['date'] = pd.to_datetime(indices_data['date'])
            
            # Get unique indices
            indices = indices_data['symbol'].unique().tolist()
            
            # Select index
            selected_index = st.selectbox(
                "Select Market Index",
                indices,
                index=0 if indices else None
            )
            
            if selected_index:
                # Filter data
                index_data = indices_data[indices_data['symbol'] == selected_index].copy()
                index_data.sort_values('date', inplace=True)
                
                # Date range selection
                date_range = st.slider(
                    "Select Date Range",
                    min_value=index_data['date'].min().date(),
                    max_value=index_data['date'].max().date(),
                    value=(
                        index_data['date'].max().date() - pd.Timedelta(days=365),
                        index_data['date'].max().date()
                    ),
                    format="YYYY-MM-DD",
                    key="index_date_slider"
                )
                
                # Filter data
                filtered_index = index_data[
                    (index_data['date'].dt.date >= date_range[0]) &
                    (index_data['date'].dt.date <= date_range[1])
                ]
                
                # Create candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=filtered_index['date'],
                    open=filtered_index['open'],
                    high=filtered_index['high'],
                    low=filtered_index['low'],
                    close=filtered_index['close'],
                    name='Price'
                )])
                
                fig.update_layout(
                    title=f"{selected_index} Index Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Returns analysis
                filtered_index['daily_return'] = filtered_index['close'].pct_change()
                
                # Cumulative returns
                filtered_index['cumulative_return'] = (1 + filtered_index['daily_return']).cumprod() - 1
                
                # Plot cumulative returns
                cum_fig = px.line(
                    filtered_index,
                    x='date',
                    y='cumulative_return',
                    title=f"{selected_index} Cumulative Return"
                )
                
                st.plotly_chart(cum_fig, use_container_width=True)
            else:
                st.info("Please select an index.")
        else:
            st.error("No market index data found.")

# Material events explorer
elif data_type == "Material Events (8-Ks)":
    st.header("Material Events (8-Ks)")
    
    # Option to view by company or all events
    view_option = st.radio(
        "View Options",
        ["By Company", "All Material Events"]
    )
    
    if view_option == "By Company":
        # Select company
        selected_company = st.selectbox("Select Company", company_dirs)
        
        # Load 8-K filings
        filings_path = Config.COMPANIES_DIR / selected_company / "filings_8k.parquet"
        
        if filings_path.exists():
            filings = pd.read_parquet(filings_path)
            
            # Ensure date column is datetime
            if 'acceptedDate' in filings.columns:
                filings['acceptedDate'] = pd.to_datetime(filings['acceptedDate'])
                filings = filings.sort_values('acceptedDate', ascending=False)
            
            # Display 8-K filings
            st.subheader(f"8-K Filings for {selected_company}")
            
            if not filings.empty:
                # Create table for display
                display_columns = ['acceptedDate', 'type', 'title', 'finalLink']
                display_columns = [col for col in display_columns if col in filings.columns]
                
                if display_columns:
                    # Create a copyable DataFrame
                    display_df = filings[display_columns].copy()
                    
                    # Convert links to clickable format
                    if 'finalLink' in display_df.columns:
                        display_df['finalLink'] = display_df['finalLink'].apply(
                            lambda x: f'<a href="{x}" target="_blank">View Filing</a>' if pd.notnull(x) else ''
                        )
                    
                    st.dataframe(
                        display_df,
                        column_config={
                            "finalLink": st.column_config.Column(
                                "Filing Link",
                                width="small",
                                help="Link to the SEC filing",
                            )
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Filing frequency over time
                    if 'acceptedDate' in filings.columns:
                        # Group by month
                        filings['month'] = filings['acceptedDate'].dt.to_period('M')
                        monthly_counts = filings.groupby('month').size().reset_index(name='count')
                        monthly_counts['month'] = monthly_counts['month'].dt.to_timestamp()
                        
                        # Create bar chart
                        fig = px.bar(
                            monthly_counts,
                            x='month',
                            y='count',
                            title=f"8-K Filing Frequency for {selected_company}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No display columns available.")
            else:
                st.info(f"No 8-K filings found for {selected_company}.")
        else:
            st.error(f"No 8-K filing data found for {selected_company}")
    else:
        # Load all material events
        events_path = Config.EVENTS_DIR / "material_events.parquet"
        
        if events_path.exists():
            events = pd.read_parquet(events_path)
            
            # Ensure date column is datetime
            if 'date' in events.columns:
                events['date'] = pd.to_datetime(events['date'])
                events = events.sort_values('date', ascending=False)
            
            # Display material events
            st.subheader("Recent Material Events (8-K Filings)")
            
            if not events.empty:
                # Company filter
                companies = events['symbol'].unique().tolist() if 'symbol' in events.columns else []
                filtered_companies = st.multiselect(
                    "Filter by Companies",
                    companies,
                    default=[]
                )
                
                # Apply filters
                filtered_events = events
                if filtered_companies:
                    filtered_events = filtered_events[filtered_events['symbol'].isin(filtered_companies)]
                
                # Display events
                display_columns = ['date', 'symbol', 'title', 'link']
                display_columns = [col for col in display_columns if col in filtered_events.columns]
                
                if display_columns:
                    # Create a copyable DataFrame
                    display_df = filtered_events[display_columns].copy()
                    
                    # Convert links to clickable format
                    if 'link' in display_df.columns:
                        display_df['link'] = display_df['link'].apply(
                            lambda x: f'<a href="{x}" target="_blank">View Filing</a>' if pd.notnull(x) else ''
                        )
                    
                    st.dataframe(
                        display_df,
                        column_config={
                            "link": st.column_config.Column(
                                "Filing Link",
                                width="small",
                                help="Link to the SEC filing",
                            )
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Event frequency over time
                    if 'date' in filtered_events.columns and 'symbol' in filtered_events.columns:
                        # Group by month and company
                        filtered_events['month'] = filtered_events['date'].dt.to_period('M')
                        monthly_counts = filtered_events.groupby(['month', 'symbol']).size().reset_index(name='count')
                        monthly_counts['month'] = monthly_counts['month'].dt.to_timestamp()
                        
                        # Create line chart
                        fig = px.line(
                            monthly_counts,
                            x='month',
                            y='count',
                            color='symbol',
                            title="8-K Filing Frequency by Company"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No display columns available.")
            else:
                st.info("No material events found.")
        else:
            st.error("No material events data found.")

# Footer
st.markdown("---")
st.markdown("""
**Dynamic-KG Data Explorer** - This tool is part of the Dynamic-KG project, 
which creates intelligent digital twins of entities and their environments through evolving knowledge graphs.
""")