"""
03_Event_Analysis_Explorer.py

This page explores the relationship between 8-K filings and news articles,
analyzing patterns and correlations across different companies and event types.
It also incorporates price data to examine market impacts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import re
from collections import Counter

# Add the project root directory to the Python path for importing config
root_dir = str(Path(__file__).parents[1].absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the config and utilities
from config import Config
from utils.ui.styles import set_page_config, card, badge
from utils.ui.session import SessionState
from utils.data.processors import load_company_data

# Initialize configuration
Config.initialize()

# Set up logger
logger = Config.get_logger("event_analysis_explorer")
logger.info("Event Analysis Explorer page loaded")

# Set page configuration
set_page_config()

# Track page view
SessionState.track_page_view("Event Analysis Explorer")

# Title and description
st.title("8-K Filing & News Analysis")
st.markdown("""
This explorer analyzes the relationship between 8-K filings and surrounding news articles,
helping to identify patterns and potential leading indicators of material events.
""")

# Define target companies for initial focus
TARGET_COMPANIES = ["DELL", "NVDA", "TSLA"]

# Helper functions
def load_filing_dates():
    """Load the 8-K filing dates data."""
    filing_dates_path = Config.EVENTS_DIR / "filing_dates.parquet"
    
    if not filing_dates_path.exists():
        st.error("Filing dates data not found. Please run the extract_filing_dates.py script first.")
        return None
    
    return pd.read_parquet(filing_dates_path)

def load_filing_news():
    """Load the news articles around filing dates."""
    filing_news_path = Config.EVENTS_DIR / "filing_news.parquet"
    
    if not filing_news_path.exists():
        st.error("Filing news data not found. Please run the collect_filing_news.py script first.")
        return None
    
    return pd.read_parquet(filing_news_path)

def load_price_data(ticker):
    """Load stock price data for a company."""
    try:
        prices = load_company_data(ticker, "prices")
        if prices is not None:
            prices['date'] = pd.to_datetime(prices['date'])
            return prices
        return None
    except Exception as e:
        st.error(f"Error loading price data for {ticker}: {e}")
        return None

def match_prices_to_filings(filings_df, ticker):
    """Match stock prices to filings for a specific company."""
    # Load price data
    prices = load_price_data(ticker)
    if prices is None:
        return None
    
    # Filter filings for this company
    company_filings = filings_df[filings_df['ticker'] == ticker].copy()
    if company_filings.empty:
        return None
    
    # For each filing, find prices around the filing date
    results = []
    
    for _, filing in company_filings.iterrows():
        filing_date = filing['filing_date']
        filing_id = filing['filing_id']
        
        # Calculate date range
        start_date = filing_date - timedelta(days=5)
        end_date = filing_date + timedelta(days=10)
        
        # Get prices in range
        date_prices = prices[
            (prices['date'] >= start_date) &
            (prices['date'] <= end_date)
        ].sort_values('date')
        
        if len(date_prices) < 2:
            continue
        
        # Find pre-filing price (closest before filing date)
        pre_prices = date_prices[date_prices['date'] <= filing_date]
        if not pre_prices.empty:
            pre_price = pre_prices.iloc[-1]['close']
        else:
            pre_price = None
        
        # Find prices at different intervals after filing
        for days in [1, 3, 5, 10]:
            target_date = filing_date + timedelta(days=days)
            post_prices = date_prices[date_prices['date'] >= target_date]
            
            if not post_prices.empty:
                post_price = post_prices.iloc[0]['close']
                
                if pre_price is not None and pre_price > 0:
                    # Calculate price change
                    price_change = (post_price - pre_price) / pre_price
                else:
                    price_change = None
            else:
                post_price = None
                price_change = None
            
            # Add to results
            results.append({
                'filing_id': filing_id,
                'filing_date': filing_date,
                'days_after': days,
                'pre_price': pre_price,
                'post_price': post_price,
                'price_change': price_change
            })
    
    return pd.DataFrame(results)

def extract_item_numbers(filing_df):
    """Extract 8-K item numbers from filing data if available."""
    # Check if we already have item information
    if 'filing_item' in filing_df.columns:
        return filing_df
    
    # Try to extract from title or description
    if 'filing_title' in filing_df.columns:
        # Define patterns to look for
        patterns = [
            r"Item\s+(\d+\.\d+)",  # Standard format: "Item X.XX"
            r"ITEM\s+(\d+\.\d+)",  # All caps: "ITEM X.XX"
            r"Item\s+(\d+)",       # Just number: "Item X"
            r"ITEM\s+(\d+)"        # All caps, just number: "ITEM X"
        ]
        
        # Function to search for patterns
        def find_item_patterns(text, patterns):
            if pd.isna(text):
                return []
            
            results = []
            for pattern in patterns:
                matches = re.findall(pattern, str(text))
                results.extend(matches)
            return list(set(results))  # Remove duplicates
        
        # Apply to titles
        filing_df['extracted_items'] = filing_df['filing_title'].apply(
            lambda x: find_item_patterns(x, patterns)
        )
        
        # Try description if available and title didn't yield results
        if 'filing_description' in filing_df.columns:
            filing_df['extracted_items'] = filing_df.apply(
                lambda row: find_item_patterns(row['filing_description'], patterns) 
                if not row['extracted_items'] and not pd.isna(row['filing_description'])
                else row['extracted_items'],
                axis=1
            )
    
    return filing_df

def analyze_news_sentiment(news_df):
    """
    Simple sentiment analysis on news headlines.
    
    In a real implementation, you would use a proper NLP model.
    This is a simplified version for demonstration.
    """
    # Define simple positive and negative word lists
    positive_words = [
        'up', 'rise', 'bullish', 'gain', 'profit', 'success', 'positive', 'growth',
        'increase', 'higher', 'beat', 'exceeds', 'strong', 'soar', 'surge', 'jump',
        'record', 'good', 'great', 'excellent', 'impressive'
    ]
    
    negative_words = [
        'down', 'fall', 'bearish', 'loss', 'negative', 'decline', 'decrease', 'lower',
        'miss', 'weak', 'poor', 'bad', 'drop', 'plunge', 'slump', 'tumble', 'slide',
        'below', 'disappoint', 'concerned', 'worried', 'warning', 'risk'
    ]
    
    def simple_sentiment(text):
        if pd.isna(text):
            return 0
        
        text = text.lower()
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            return 1  # Positive
        elif neg_count > pos_count:
            return -1  # Negative
        else:
            return 0  # Neutral
    
    # Apply sentiment analysis to title and text
    news_df['title_sentiment'] = news_df['title'].apply(simple_sentiment)
    
    # If 'text' column exists (it might be 'content' or something else)
    if 'text' in news_df.columns:
        news_df['content_sentiment'] = news_df['text'].apply(simple_sentiment)
    elif 'content' in news_df.columns:
        news_df['content_sentiment'] = news_df['content'].apply(simple_sentiment)
    
    return news_df

def aggregate_news_sentiment(news_df):
    """Aggregate sentiment by filing and days from filing."""
    # Ensure we have sentiment columns
    if 'title_sentiment' not in news_df.columns:
        news_df = analyze_news_sentiment(news_df)
    
    # Group by filing_id and days_from_filing
    grouped = news_df.groupby(['filing_id', 'days_from_filing']).agg({
        'title_sentiment': ['mean', 'count'],
        'symbol': 'first',  # Keep ticker for reference
        'filing_date': 'first'  # Keep filing date for reference
    })
    
    # Flatten multi-index columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    return grouped

    # Load data
with st.spinner("Loading 8-K filing and news data..."):
    filings_df = load_filing_dates()
    news_df = load_filing_news()

# Check if data is available
if filings_df is None or news_df is None:
    st.stop()

# Handle column name differences in the news data
if news_df is not None and not news_df.empty:
    # Check if we need to create a ticker column from symbol
    if 'symbol' in news_df.columns and 'ticker' not in news_df.columns:
        news_df['ticker'] = news_df['symbol']
        
    # Handle publishedDate column
    if 'publishedDate' in news_df.columns and 'date' not in news_df.columns:
        news_df['date'] = news_df['publishedDate']

# Extract item numbers if needed
filings_df = extract_item_numbers(filings_df)

# Process news sentiment
news_df = analyze_news_sentiment(news_df)

# Sidebar filters
st.sidebar.header("Analysis Controls")

# Company selection
available_companies = sorted(filings_df['ticker'].unique())
selected_company = st.sidebar.selectbox(
    "Select Company",
    options=available_companies,
    index=0 if available_companies else None
)

# Date range for filtering
if 'filing_date' in filings_df.columns:
    min_date = filings_df['filing_date'].min().date()
    max_date = filings_df['filing_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Filing Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range[0]
        end_date = max_date

# Main content with tabs
tab1, tab2, tab3 = st.tabs([
    "Filing Overview", "News Analysis", "Price Impact"
])

# Create filtered data
if selected_company:
    company_filings = filings_df[filings_df['ticker'] == selected_company].copy()
    
    # Apply date filter if needed
    if 'start_date' in locals() and 'end_date' in locals():
        company_filings = company_filings[
            (company_filings['filing_date'].dt.date >= start_date) &
            (company_filings['filing_date'].dt.date <= end_date)
        ]
    
    # Get related news
    company_news = news_df[news_df['ticker'] == selected_company].copy()
    
    # Link to filings
    company_news = company_news[company_news['filing_id'].isin(company_filings['filing_id'])]

# Tab 1: Filing Overview
with tab1:
    st.header(f"8-K Filing Overview for {selected_company}")
    
    if 'company_filings' in locals() and not company_filings.empty:
        # Show basic stats
        st.subheader("Filing Statistics")
        
        filing_count = len(company_filings)
        date_range_str = f"{company_filings['filing_date'].min().strftime('%Y-%m-%d')} to {company_filings['filing_date'].max().strftime('%Y-%m-%d')}"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total 8-K Filings", filing_count)
        with col2:
            st.metric("Date Range", date_range_str)
        
        # Filing timeline
        st.subheader("Filing Timeline")
        
        # Create a time series of filings
        company_filings['month'] = company_filings['filing_date'].dt.to_period('M')
        monthly_counts = company_filings.groupby('month').size().reset_index(name='count')
        monthly_counts['month'] = monthly_counts['month'].astype(str)
        
        fig = px.line(
            monthly_counts,
            x='month',
            y='count',
            title=f"8-K Filings by Month for {selected_company}",
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Filings"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Item analysis if available
        if 'extracted_items' in company_filings.columns:
            st.subheader("Filing Item Analysis")
            
            # Flatten items
            all_items = []
            for items in company_filings['extracted_items']:
                if isinstance(items, list):
                    all_items.extend(items)
            
            # Count item frequencies
            item_counts = Counter(all_items)
            
            if item_counts:
                # Create a DataFrame for visualization
                item_df = pd.DataFrame({
                    'Item': list(item_counts.keys()),
                    'Count': list(item_counts.values())
                }).sort_values('Count', ascending=False)
                
                fig = px.bar(
                    item_df,
                    x='Item',
                    y='Count',
                    title=f"Most Common 8-K Items for {selected_company}",
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Item descriptions
                st.subheader("Common 8-K Item Descriptions")
                
                # Define common 8-K item descriptions
                item_descriptions = {
                    "1.01": "Entry into a Material Definitive Agreement",
                    "1.02": "Termination of a Material Definitive Agreement",
                    "2.01": "Completion of Acquisition or Disposition of Assets",
                    "2.02": "Results of Operations and Financial Condition",
                    "2.03": "Creation of a Direct Financial Obligation",
                    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
                    "2.05": "Costs Associated with Exit or Disposal Activities",
                    "2.06": "Material Impairments",
                    "3.01": "Notice of Delisting or Failure to Satisfy Listing Rule",
                    "3.02": "Unregistered Sales of Equity Securities",
                    "4.01": "Changes in Registrant's Certifying Accountant",
                    "4.02": "Non-Reliance on Previously Issued Financial Statements",
                    "5.01": "Changes in Control of Registrant",
                    "5.02": "Departure/Election of Directors or Principal Officers",
                    "5.03": "Amendments to Articles of Incorporation or Bylaws",
                    "5.07": "Submission of Matters to a Vote of Security Holders",
                    "7.01": "Regulation FD Disclosure",
                    "8.01": "Other Events",
                    "9.01": "Financial Statements and Exhibits"
                }
                
                # Create description table
                desc_data = []
                for item, count in item_counts.items():
                    # Extract item number without "Item" prefix
                    item_num = item.replace("Item ", "") if "Item " in item else item
                    
                    desc_data.append({
                        "Item": f"Item {item_num}",
                        "Description": item_descriptions.get(item_num, "Unknown"),
                        "Count": count
                    })
                
                st.dataframe(pd.DataFrame(desc_data).sort_values("Count", ascending=False))
        
        # Recent filings
        st.subheader("Recent Filings")
        
        # Display recent filings
        recent = company_filings.sort_values('filing_date', ascending=False).head(10)
        
        for _, filing in recent.iterrows():
            filing_date = filing['filing_date'].strftime('%Y-%m-%d')
            filing_id = filing['filing_id']
            
            # Try to get title
            if 'filing_title' in filing and pd.notna(filing['filing_title']):
                title = filing['filing_title']
            else:
                title = f"Filing {filing_id}"
            
            with st.expander(f"{filing_date} - {title}"):
                st.write(f"**Filing ID:** {filing_id}")
                
                # Show extracted items if available
                if 'extracted_items' in filing and filing['extracted_items']:
                    st.write("**Items:**")
                    items_str = ", ".join([f"Item {item}" for item in filing['extracted_items']])
                    st.write(items_str)
                
                # Show description if available
                if 'filing_description' in filing and pd.notna(filing['filing_description']):
                    st.write("**Description:**")
                    st.write(filing['filing_description'])
                
                # Show link if available
                if 'filing_link' in filing and pd.notna(filing['filing_link']):
                    st.markdown(f"[View Filing on SEC]({filing['filing_link']})")
                
                # Preview related news
                if 'company_news' in locals() and not company_news.empty:
                    filing_news = company_news[company_news['filing_id'] == filing_id]
                    
                    if not filing_news.empty:
                        st.write(f"**Related News ({len(filing_news)} articles):**")
                        
                        for _, news in filing_news.head(3).iterrows():
                            sentiment = "🟢" if news['title_sentiment'] > 0 else "🔴" if news['title_sentiment'] < 0 else "⚪"
                            days = news['days_from_filing']
                            days_str = f"{days} days before" if days < 0 else f"{days} days after" if days > 0 else "same day"
                            
                            st.markdown(f"{sentiment} **{days_str}**: {news['title']}")
                            
                        if len(filing_news) > 3:
                            st.write(f"...and {len(filing_news) - 3} more articles")
    else:
        st.info(f"No 8-K filings found for {selected_company} in the selected date range.")

# Tab 2: News Analysis
with tab2:
    st.header(f"News Analysis for {selected_company}")
    
    if 'company_news' in locals() and not company_news.empty:
        # Show basic stats
        st.subheader("News Statistics")
        
        news_count = len(company_news)
        filings_with_news = company_news['filing_id'].nunique()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total News Articles", news_count)
        with col2:
            st.metric("Filings with News", filings_with_news)
        with col3:
            avg_news = news_count / filings_with_news if filings_with_news > 0 else 0
            st.metric("Avg. News per Filing", f"{avg_news:.1f}")
        
        # News sentiment timeline
        st.subheader("News Sentiment Around Filings")
        
        # Aggregate sentiment
        sentiment_df = aggregate_news_sentiment(company_news)
        
        # Create a line chart of sentiment by days from filing
        days_sentiment = sentiment_df.groupby('days_from_filing').agg({
            'title_sentiment_mean': 'mean',
            'title_sentiment_count': 'sum'
        }).reset_index()
        
        fig = px.line(
            days_sentiment,
            x='days_from_filing',
            y='title_sentiment_mean',
            title=f"Average News Sentiment Around 8-K Filings ({selected_company})",
            markers=True,
            color_discrete_sequence=['blue']
        )
        
        # Add size as a marker property rather than as a parameter
        fig.update_traces(
            marker=dict(
                size=days_sentiment['title_sentiment_count'] / days_sentiment['title_sentiment_count'].max() * 15 + 5
            )
        )
        
        # Add zero line and filing date line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            xaxis_title="Days Relative to Filing (Negative = Before, Positive = After)",
            yaxis_title="Average Sentiment (-1 to 1)",
            xaxis=dict(
                tickmode='linear',
                tick0=-2,
                dtick=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # News volume
        st.subheader("News Volume Around Filings")
        
        # Count news by days from filing
        volume_df = company_news.groupby('days_from_filing').size().reset_index(name='count')
        
        fig = px.bar(
            volume_df,
            x='days_from_filing',
            y='count',
            title=f"News Volume Around 8-K Filings ({selected_company})",
            color='count',
            color_continuous_scale='Viridis'
        )
        
        # Add filing date line
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            xaxis_title="Days Relative to Filing (Negative = Before, Positive = After)",
            yaxis_title="Number of News Articles",
            xaxis=dict(
                tickmode='linear',
                tick0=-2,
                dtick=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # News content analysis
        st.subheader("News Content Analysis")
        
        # Extract common words/phrases from titles
        all_titles = company_news['title'].dropna().tolist()
        
        # Simple word frequency analysis
        stop_words = set([
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
            'at', 'from', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'to', 'of', 'in', 'on', 'is', 'are', 'was', 'were', 'be', 'been',
            'has', 'have', 'had', 'do', 'does', 'did', 'as', 'until', 'while',
            'his', 'her', 'its', 'their', 'our', 'my', 'your', 'that', 'this',
            'these', 'those'
        ])
        
        # Function to extract words
        def extract_words(text):
            if pd.isna(text):
                return []
            
            # Convert to lowercase and split
            words = re.findall(r'\w+', text.lower())
            
            # Filter out stop words and company name
            filtered = [word for word in words if word not in stop_words and word != selected_company.lower()]
            
            return filtered
        
        # Extract and count words
        all_words = []
        for title in all_titles:
            all_words.extend(extract_words(title))
        
        word_counts = Counter(all_words)
        
        # Create word frequency chart
        top_words = pd.DataFrame({
            'Word': list(word_counts.keys()),
            'Frequency': list(word_counts.values())
        }).sort_values('Frequency', ascending=False).head(20)
        
        fig = px.bar(
            top_words,
            x='Frequency',
            y='Word',
            title=f"Most Common Words in News Headlines ({selected_company})",
            orientation='h',
            color='Frequency',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title="Frequency",
            yaxis_title="Word",
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # News sources analysis
        if 'site' in company_news.columns:
            st.subheader("News Sources")
            
            source_counts = company_news['site'].value_counts().reset_index()
            source_counts.columns = ['Source', 'Count']
            
            fig = px.pie(
                source_counts.head(10),
                values='Count',
                names='Source',
                title=f"Top News Sources for {selected_company}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No news articles found for {selected_company} around filing dates.")

# Tab 3: Price Impact
with tab3:
    st.header(f"Price Impact Analysis for {selected_company}")
    
    # Load and process price data
    with st.spinner("Loading and processing price data..."):
        price_impacts = match_prices_to_filings(filings_df, selected_company)
    
    if price_impacts is not None and not price_impacts.empty:
        # Show average price changes
        st.subheader("Average Price Changes After Filings")
        
        # Calculate average changes by days after
        avg_changes = price_impacts.groupby('days_after').agg({
            'price_change': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten multi-index columns
        avg_changes.columns = ['_'.join(col).strip() for col in avg_changes.columns.values]
        
        # Create bar chart
        fig = px.bar(
            avg_changes,
            x='days_after_',
            y='price_change_mean',
            error_y='price_change_std',
            title=f"Average Price Change After 8-K Filings ({selected_company})",
            labels={
                'days_after_': 'Days After Filing',
                'price_change_mean': 'Average Price Change (%)'
            },
            color='price_change_mean',
            color_continuous_scale='RdBu_r',
            text=avg_changes['price_change_mean'].apply(lambda x: f"{x*100:.2f}%")
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=avg_changes['days_after_']
            ),
            yaxis=dict(
                tickformat=".1%"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # If we have item analysis, try to correlate with price changes
        if 'extracted_items' in company_filings.columns:
            st.subheader("Price Impact by Filing Item")
            
            # Create mapping from filing_id to items
            filing_items = {}
            for _, filing in company_filings.iterrows():
                if isinstance(filing['extracted_items'], list) and filing['extracted_items']:
                    filing_items[filing['filing_id']] = filing['extracted_items']
            
            # Add items to price impacts
            if filing_items:
                # Create item-specific datasets
                item_impacts = []
                
                for item in set([item for items in filing_items.values() for item in items]):
                    # Get filings with this item
                    item_filings = [fid for fid, items in filing_items.items() if item in items]
                    
                    # Get price impacts for these filings
                    impacts = price_impacts[price_impacts['filing_id'].isin(item_filings)]
                    
                    if not impacts.empty:
                        # Calculate average change for each time period
                        for days in impacts['days_after'].unique():
                            day_impacts = impacts[impacts['days_after'] == days]
                            avg_change = day_impacts['price_change'].mean()
                            
                            item_impacts.append({
                                'item': item,
                                'days_after': days,
                                'avg_change': avg_change,
                                'count': len(day_impacts)
                            })
                
                if item_impacts:
                    # Convert to DataFrame
                    item_df = pd.DataFrame(item_impacts)
                    
                    # Create heatmap
                    pivot_df = item_df.pivot(index='item', columns='days_after', values='avg_change')
                    
                    fig = px.imshow(
                        pivot_df,
                        title=f"Price Impact by Filing Item ({selected_company})",
                        labels=dict(x="Days After Filing", y="Item", color="Price Change"),
                        color_continuous_scale='RdBu_r',
                        aspect="auto",
                        zmin=-0.1,
                        zmax=0.1
                    )
                    
                    fig.update_layout(
                        xaxis=dict(
                            tickmode='array',
                            tickvals=list(pivot_df.columns)
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display items with the strongest impacts
                    st.subheader("Items with Strongest Price Impact")
                    
                    # Calculate absolute impact for each item
                    item_max_impact = item_df.groupby('item').agg({
                        'avg_change': lambda x: x.abs().max(),
                        'count': 'sum'
                    }).reset_index()
                    
                    # Get top items by impact
                    top_impact_items = item_max_impact.sort_values('avg_change', ascending=False).head(5)
                    
                    for _, row in top_impact_items.iterrows():
                        # Extract item number without "Item" prefix
                        item_num = row['item'].replace("Item ", "") if "Item " in row['item'] else row['item']
                        
                        # Get description
                        description = item_descriptions.get(item_num, "Unknown")
                        
                        st.markdown(f"**{row['item']}**: {description}")
                        st.markdown(f"Max Impact: {row['avg_change']*100:.2f}% ({row['count']} filings)")
                        
                        # Get specific examples
                        examples = price_impacts[
                            price_impacts['filing_id'].isin([
                                fid for fid, items in filing_items.items() 
                                if row['item'] in items
                            ])
                        ]
                        
                        if not examples.empty:
                            # Sort by absolute price change
                            examples['abs_change'] = examples['price_change'].abs()
                            top_examples = examples.sort_values('abs_change', ascending=False).head(3)
                            
                            st.markdown("**Notable examples:**")
                            for _, ex in top_examples.iterrows():
                                change_str = f"{ex['price_change']*100:.2f}% after {ex['days_after']} days"
                                filing_date = ex['filing_date'].strftime('%Y-%m-%d')
                                st.markdown(f"- {filing_date}: {change_str}")
                            
                            st.markdown("---")
        
        # Correlate price changes with news sentiment if available
        if 'company_news' in locals() and not company_news.empty:
            st.subheader("News Sentiment vs Price Changes")
            
            # Aggregate news sentiment by filing
            sentiment_by_filing = company_news.groupby('filing_id').agg({
                'title_sentiment': 'mean',
                'filing_date': 'first'
            }).reset_index()
            
            # Join with price impacts
            sentiment_price = pd.merge(
                sentiment_by_filing,
                price_impacts,
                on=['filing_id', 'filing_date']
            )
            
            if not sentiment_price.empty:
                # Create scatter plot for different day periods
                for days in sorted(sentiment_price['days_after'].unique()):
                    day_data = sentiment_price[sentiment_price['days_after'] == days]
                    
                    if len(day_data) >= 5:  # Need enough points
                        fig = px.scatter(
                            day_data,
                            x='title_sentiment',
                            y='price_change',
                            title=f"News Sentiment vs {days}-Day Price Change ({selected_company})",
                            trendline='ols',
                            labels={
                                'title_sentiment': 'Average News Sentiment',
                                'price_change': 'Price Change (%)'
                            }
                        )
                        
                        # Add zero lines
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.add_vline(x=0, line_dash="dash", line_color="gray")
                        
                        fig.update_layout(
                            yaxis=dict(
                                tickformat=".1%"
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate correlation
                        correlation = day_data['title_sentiment'].corr(day_data['price_change'])
                        st.write(f"Correlation: {correlation:.3f}")
    else:
        st.info(f"No price data available for {selected_company} or unable to match with filing dates.")

# Footer
st.markdown("---")
st.markdown("""
This analysis is part of the Dynamic-KG project, which aims to create evolving knowledge graphs
that capture temporal relationships between entities and events. The correlation between 8-K filings,
news sentiment, and price movements can help identify potential leading indicators and patterns.

To expand this analysis:
1. Run the extractor script for more companies
2. Collect news for more filing dates
3. Integrate additional data sources such as earnings calls and social media
""")