"""
collect_filing_news.py

This script collects news articles around 8-K filing dates using the FMP API.
It collects news from 2 days before to 5 days after each filing date.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta
import json
import os
import requests
import time

# Add the project root directory to the Python path for importing config
root_dir = str(Path(__file__).parents[2].absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the config
from config import Config

# Initialize configuration
Config.initialize()

# Set up logger
logger = Config.get_logger(__name__)
logger.info("Starting news collection around 8-K filing dates")

# Define the time window for news collection
DAYS_BEFORE = 2
DAYS_AFTER = 5

def load_filing_dates():
    """
    Load the previously extracted filing dates.
    
    Returns:
        DataFrame containing filing dates or None if file not found
    """
    filing_dates_path = Config.EVENTS_DIR / "filing_dates.parquet"
    
    if not filing_dates_path.exists():
        logger.error(f"Filing dates file not found: {filing_dates_path}")
        return None
    
    try:
        filings = pd.read_parquet(filing_dates_path)
        logger.info(f"Loaded {len(filings)} filing dates")
        return filings
    except Exception as e:
        logger.error(f"Error loading filing dates: {e}")
        return None

def collect_news_for_filing(ticker, filing_date, filing_id):
    """
    Collect news around a specific filing date using FMP API.
    
    Args:
        ticker: Company ticker symbol
        filing_date: Date of the filing
        filing_id: Unique identifier for the filing
        
    Returns:
        DataFrame containing news articles or None if API call fails
    """
    try:
        # Ensure filing_date is a datetime object
        if not isinstance(filing_date, datetime):
            filing_date = pd.to_datetime(filing_date)
        
        # Calculate the date range
        start_date = filing_date - timedelta(days=DAYS_BEFORE)
        end_date = filing_date + timedelta(days=DAYS_AFTER)
        
        # Format dates for API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Construct API URL
        api_key = Config.FMP_API_KEY
        if not api_key:
            logger.error("FMP API key not found in configuration")
            return None
        
        url = f"{Config.FMP_API_BASE_URL}/stock_news"
        params = {
            'tickers': ticker,
            'from': start_str,
            'to': end_str,
            'apikey': api_key
        }
        
        # Make the API request
        logger.info(f"Requesting news for {ticker} from {start_str} to {end_str}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse the response
        news_data = response.json()
        
        if not news_data:
            logger.warning(f"No news found for {ticker} from {start_str} to {end_str}")
            return None
        
        # Convert to DataFrame
        news_df = pd.DataFrame(news_data)
        
        # Ensure we have the expected columns
        if 'date' not in news_df.columns:
            logger.warning(f"Missing 'date' column in API response. Available columns: {news_df.columns.tolist()}")
            
            # Check for alternative date columns
            date_column = None
            for alt_col in ['publishedDate', 'published_date', 'timestamp', 'time']:
                if alt_col in news_df.columns:
                    date_column = alt_col
                    break
            
            if date_column:
                logger.info(f"Using alternative date column: {date_column}")
                news_df['date'] = news_df[date_column]
            else:
                # Create a placeholder date
                logger.warning("No date column found, using filing date as placeholder")
                news_df['date'] = filing_date.strftime('%Y-%m-%d')
        
        # Add reference to the filing
        news_df['filing_id'] = filing_id
        news_df['filing_date'] = filing_date
        
        # Convert API date strings to datetime objects more robustly
        def parse_date(date_str):
            try:
                # Try to parse ISO format
                if isinstance(date_str, str):
                    # Remove Z and replace with +00:00 for timezone handling
                    if date_str.endswith('Z'):
                        date_str = date_str.replace('Z', '+00:00')
                    return pd.to_datetime(date_str)
                else:
                    return pd.NaT
            except Exception as e:
                logger.debug(f"Date parsing error for '{date_str}': {e}")
                return pd.NaT
        
        # Apply date parsing and calculate days difference
        news_df['article_date'] = news_df['date'].apply(parse_date)
        
        # Handle date parsing errors
        if news_df['article_date'].isna().all():
            logger.warning(f"Failed to parse any dates from the 'date' column. Sample values: {news_df['date'].head(3).tolist()}")
            # Use a placeholder date equal to filing date
            news_df['article_date'] = filing_date
            news_df['days_from_filing'] = 0
        else:
            # Calculate days from filing
            news_df['days_from_filing'] = news_df.apply(
                lambda row: (row['article_date'] - filing_date).days 
                if pd.notna(row['article_date']) else 0, 
                axis=1
            )
        
        logger.info(f"Collected {len(news_df)} news articles for {ticker} around {filing_date.strftime('%Y-%m-%d')}")
        return news_df
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error for {ticker}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error collecting news for {ticker}: {e}")
        logger.debug(f"Error details: {traceback.format_exc()}")
        return None

def collect_all_news(filings_df, max_filings_per_company=None):
    """
    Collect news for all filings or a subset if specified.
    
    Args:
        filings_df: DataFrame containing filing dates
        max_filings_per_company: Maximum number of filings to process per company (for testing)
        
    Returns:
        Combined DataFrame with all news
    """
    all_news = []
    
    # Group by ticker to respect rate limits
    for ticker, group in filings_df.groupby('ticker'):
        logger.info(f"Processing news for {ticker} ({len(group)} filings)")
        
        # Limit the number of filings if specified
        if max_filings_per_company is not None:
            # Sort by date (most recent first) and take the limited number
            group = group.sort_values('filing_date', ascending=False).head(max_filings_per_company)
            logger.info(f"Limited to {len(group)} recent filings for {ticker}")
        
        # Process each filing
        for idx, filing in group.iterrows():
            filing_date = filing['filing_date']
            filing_id = filing['filing_id']
            
            news = collect_news_for_filing(ticker, filing_date, filing_id)
            
            if news is not None and not news.empty:
                all_news.append(news)
            
            # Respect API rate limits
            time.sleep(0.5)
    
    # Combine all news
    if all_news:
        combined = pd.concat(all_news, ignore_index=True)
        logger.info(f"Combined news contains {len(combined)} articles")
        return combined
    else:
        logger.warning("No news collected for any filing")
        return pd.DataFrame()

def save_news_data(news_df):
    """
    Save the collected news to multiple formats.
    
    Args:
        news_df: DataFrame containing news articles
    """
    # Ensure events directory exists
    events_dir = Config.EVENTS_DIR
    events_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we have necessary columns and create aliases if needed
    if 'symbol' in news_df.columns and 'ticker' not in news_df.columns:
        logger.info("Creating 'ticker' column from 'symbol' column")
        news_df['ticker'] = news_df['symbol']
    
    # Ensure we have ticker column
    if 'ticker' not in news_df.columns:
        logger.warning("No ticker or symbol column found, using filing_id to extract ticker")
        # Extract ticker from filing_id (format: TICKER_NUMBER)
        news_df['ticker'] = news_df['filing_id'].apply(
            lambda x: x.split('_')[0] if isinstance(x, str) and '_' in x else 'UNKNOWN'
        )
    
    # Save as parquet
    parquet_path = events_dir / "filing_news.parquet"
    news_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved news to {parquet_path}")
    
    # Save as CSV for readability
    csv_path = events_dir / "filing_news.csv"
    news_df.to_csv(csv_path, index=False)
    logger.info(f"Saved news to {csv_path}")
    
    # Save summary statistics
    companies = news_df['ticker'].unique()
    filings = news_df['filing_id'].unique()
    
    summary = {
        "companies": list(companies),
        "company_count": len(companies),
        "filing_count": len(filings),
        "news_count": len(news_df),
        "news_per_filing": {filing: int(count) for filing, count in news_df['filing_id'].value_counts().head(20).items()},
        "news_per_company": {ticker: int(count) for ticker, count in news_df['ticker'].value_counts().items()}
    }
    
    summary_path = events_dir / "filing_news_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")

def main():
    """Main function to run the news collection process."""
    try:
        # Load filing dates
        filings_df = load_filing_dates()
        
        if filings_df is None or filings_df.empty:
            logger.error("No filing dates available, cannot collect news")
            return
        
        # Check if we have the FMP API key
        if not Config.FMP_API_KEY:
            logger.error("FMP_API_KEY not set in configuration")
            return
        
        # Add debugging for date column
        logger.info(f"Filing date column type: {filings_df['filing_date'].dtype}")
        logger.info(f"Sample filing dates: {filings_df['filing_date'].head(3).tolist()}")
        
        # Convert filing_date to datetime if it's not already
        if not pd.api.types.is_datetime64_dtype(filings_df['filing_date']):
            logger.info("Converting filing_date column to datetime")
            filings_df['filing_date'] = pd.to_datetime(filings_df['filing_date'])
        
        # Collect news for all filings (limit per company for testing)
        # Adjust or remove the limit for production
        news_df = collect_all_news(filings_df, max_filings_per_company=5)
        
        if news_df is not None and not news_df.empty:
            # Save the collected news
            save_news_data(news_df)
            logger.info("News collection completed successfully")
        else:
            logger.error("No news collected, nothing to save")
            
            # Create an empty news file to prevent errors in the explorer
            empty_news = pd.DataFrame(columns=[
                'title', 'text', 'ticker', 'symbol', 'image', 'url', 'site', 
                'published_date', 'article_date', 'filing_id', 'filing_date', 'days_from_filing'
            ])
            
            # Ensure events directory exists
            events_dir = Config.EVENTS_DIR
            events_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet
            parquet_path = events_dir / "filing_news.parquet"
            empty_news.to_parquet(parquet_path, index=False)
            logger.info(f"Saved empty news dataframe to {parquet_path}")
            
            # Save as CSV for readability
            csv_path = events_dir / "filing_news.csv"
            empty_news.to_csv(csv_path, index=False)
            logger.info(f"Saved empty news dataframe to {csv_path}")
    
    except Exception as e:
        logger.error(f"Error in news collection process: {e}", exc_info=True)

if __name__ == "__main__":
    main()