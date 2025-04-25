"""
extract_filing_dates.py

This script extracts 8-K filing dates for all companies in the dataStore folder,
creating a master table of filing events to use for collecting related news.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta
import json
import os

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
logger.info("Starting 8-K filing date extraction")

# Define target companies for initial focus
INITIAL_FOCUS_COMPANIES = ["DELL", "NVDA", "TSLA"]

def extract_filing_dates_for_company(ticker):
    """
    Extract 8-K filing dates for a specific company.
    
    Args:
        ticker: Company ticker symbol
        
    Returns:
        DataFrame containing filing dates or None if no data found
    """
    company_dir = Config.COMPANIES_DIR / ticker
    filings_8k_path = company_dir / "filings_8k.parquet"
    filings_path = company_dir / "filings.parquet"
    
    # Try filings_8k.parquet first
    if filings_8k_path.exists():
        logger.info(f"Loading 8-K filings for {ticker} from filings_8k.parquet")
        try:
            filings = pd.read_parquet(filings_8k_path)
            return process_filings(filings, ticker, "filings_8k.parquet")
        except Exception as e:
            logger.error(f"Error loading filings_8k.parquet for {ticker}: {e}")
    
    # If filings_8k doesn't exist or fails, try general filings
    if filings_path.exists():
        logger.info(f"Loading general filings for {ticker} from filings.parquet")
        try:
            filings = pd.read_parquet(filings_path)
            
            # Filter for 8-K filings if possible
            if 'form_type' in filings.columns:
                filings = filings[filings['form_type'] == '8-K']
                
            if not filings.empty:
                return process_filings(filings, ticker, "filings.parquet")
            else:
                logger.warning(f"No 8-K filings found for {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error loading filings.parquet for {ticker}: {e}")
    
    logger.warning(f"No filing data found for {ticker}")
    return None

def process_filings(filings, ticker, source_file):
    """
    Process filings to extract relevant date information.
    
    Args:
        filings: DataFrame of filings
        ticker: Company ticker symbol
        source_file: Source file name for record-keeping
        
    Returns:
        Processed DataFrame with standardized columns
    """
    # Identify date column
    date_col = None
    for col in ['date', 'acceptedDate', 'filingDate']:
        if col in filings.columns:
            date_col = col
            break
    
    if date_col is None:
        logger.error(f"No date column found for {ticker}")
        return None
    
    # Identify item column if available
    item_col = None
    for col in ['item', 'items', 'formType', 'type']:
        if col in filings.columns:
            item_col = col
            break
    
    # Identify link column
    link_col = None
    for col in ['finalLink', 'link', 'url']:
        if col in filings.columns:
            link_col = col
            break
    
    # Create standardized DataFrame
    result = pd.DataFrame()
    result['ticker'] = [ticker] * len(filings)
    result['filing_date'] = pd.to_datetime(filings[date_col], errors='coerce')
    
    # Drop rows with invalid dates
    result = result.dropna(subset=['filing_date'])
    
    # Add filing item/type if available
    if item_col:
        result['filing_item'] = filings[item_col]
    
    # Add link if available
    if link_col:
        result['filing_link'] = filings[link_col]
    
    # Add title if available
    if 'title' in filings.columns:
        result['filing_title'] = filings['title']
    
    # Add description if available
    if 'description' in filings.columns:
        result['filing_description'] = filings['description']
    
    # Add source file for tracking
    result['source_file'] = source_file
    
    # Add filing_id for reference
    result['filing_id'] = [f"{ticker}_{i}" for i in range(len(result))]
    
    return result

def collect_all_filing_dates():
    """
    Extract 8-K filing dates for all available companies or focus list.
    
    Returns:
        Combined DataFrame with all filing dates
    """
    all_data = []
    
    # Check if companies directory exists
    if not Config.COMPANIES_DIR.exists():
        logger.error(f"Companies directory not found: {Config.COMPANIES_DIR}")
        return pd.DataFrame()
    
    # Get all company folders
    companies = [d.name for d in Config.COMPANIES_DIR.iterdir() if d.is_dir()]
    logger.info(f"Found {len(companies)} company directories")
    
    # Check if we have our focus companies
    focus_available = [c for c in INITIAL_FOCUS_COMPANIES if c in companies]
    if focus_available:
        logger.info(f"Using focus companies: {focus_available}")
        companies_to_process = focus_available
    else:
        logger.info("Focus companies not found, using all available companies")
        companies_to_process = companies
    
    # Process each company
    for ticker in companies_to_process:
        logger.info(f"Processing {ticker}")
        company_data = extract_filing_dates_for_company(ticker)
        
        if company_data is not None and not company_data.empty:
            logger.info(f"Found {len(company_data)} filings for {ticker}")
            all_data.append(company_data)
        else:
            logger.warning(f"No valid filing data found for {ticker}")
    
    # Combine all data
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data contains {len(combined)} filings across {len(all_data)} companies")
        return combined
    else:
        logger.warning("No filing data found for any company")
        return pd.DataFrame()

def save_filing_data(filings_df):
    """
    Save the combined filing dates to multiple formats.
    
    Args:
        filings_df: DataFrame containing filing dates
    """
    # Ensure events directory exists
    events_dir = Config.EVENTS_DIR
    events_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    parquet_path = events_dir / "filing_dates.parquet"
    filings_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved filing dates to {parquet_path}")
    
    # Save as CSV for readability
    csv_path = events_dir / "filing_dates.csv"
    filings_df.to_csv(csv_path, index=False)
    logger.info(f"Saved filing dates to {csv_path}")
    
    # Save as JSON for readability
    json_path = events_dir / "filing_dates.json"
    filings_df.to_json(json_path, orient='records', date_format='iso')
    logger.info(f"Saved filing dates to {json_path}")
    
    # Save summary statistics
    companies = filings_df['ticker'].unique()
    date_range = [filings_df['filing_date'].min(), filings_df['filing_date'].max()]
    
    summary = {
        "companies": list(companies),
        "company_count": len(companies),
        "filing_count": len(filings_df),
        "date_range": {
            "start": date_range[0].strftime('%Y-%m-%d'),
            "end": date_range[1].strftime('%Y-%m-%d')
        },
        "filings_per_company": {ticker: int(count) for ticker, count in filings_df['ticker'].value_counts().items()}
    }
    
    summary_path = events_dir / "filing_dates_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved summary to {summary_path}")

def main():
    """Main function to run the extraction process."""
    try:
        # Extract all filing dates
        all_filings = collect_all_filing_dates()
        
        if not all_filings.empty:
            # Save the combined data
            save_filing_data(all_filings)
            logger.info("Filing date extraction completed successfully")
        else:
            logger.error("No filing data collected, nothing to save")
    
    except Exception as e:
        logger.error(f"Error in extraction process: {e}", exc_info=True)

if __name__ == "__main__":
    main()