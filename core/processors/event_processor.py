"""
Event processor for analyzing 8-K filings and other material events.

This module provides functions for loading, categorizing, and preparing 8-K filings
and related data for analysis. It serves as the foundation for more advanced
event analytics in the Dynamic-KG project.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import sys
import networkx as nx

# Add the project root directory to the Python path for importing config
root_dir = str(Path(__file__).parents[2].absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the config
from config import Config

# Set up logger
logger = Config.get_logger(__name__)

# 8-K Item categories and descriptions
ITEM_CATEGORIES = {
    "Item 1.01": "Entry into a Material Definitive Agreement",
    "Item 1.02": "Termination of a Material Definitive Agreement",
    "Item 1.03": "Bankruptcy or Receivership",
    "Item 1.04": "Mine Safety - Reporting of Shutdowns and Patterns of Violations",
    "Item 2.01": "Completion of Acquisition or Disposition of Assets",
    "Item 2.02": "Results of Operations and Financial Condition",
    "Item 2.03": "Creation of a Direct Financial Obligation or an Obligation under an Off-Balance Sheet Arrangement",
    "Item 2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "Item 2.05": "Costs Associated with Exit or Disposal Activities",
    "Item 2.06": "Material Impairments",
    "Item 3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
    "Item 3.02": "Unregistered Sales of Equity Securities",
    "Item 3.03": "Material Modifications to Rights of Security Holders",
    "Item 4.01": "Changes in Registrant's Certifying Accountant",
    "Item 4.02": "Non-Reliance on Previously Issued Financial Statements",
    "Item 5.01": "Changes in Control of Registrant",
    "Item 5.02": "Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers",
    "Item 5.03": "Amendments to Articles of Incorporation or Bylaws",
    "Item 5.04": "Temporary Suspension of Trading Under Registrant's Employee Benefit Plans",
    "Item 5.05": "Amendments to the Registrant's Code of Ethics, or Waiver of a Provision",
    "Item 5.06": "Change in Shell Company Status",
    "Item 5.07": "Submission of Matters to a Vote of Security Holders",
    "Item 5.08": "Shareholder Director Nominations",
    "Item 6.01": "ABS Informational and Computational Material",
    "Item 6.02": "Change of Servicer or Trustee",
    "Item 6.03": "Change in Credit Enhancement or Other External Support",
    "Item 6.04": "Failure to Make a Required Distribution",
    "Item 6.05": "Securities Act Updating Disclosure",
    "Item 7.01": "Regulation FD Disclosure",
    "Item 8.01": "Other Events",
    "Item 9.01": "Financial Statements and Exhibits"
}

# Higher-level categorization for business impact
BUSINESS_IMPACT_CATEGORIES = {
    "Leadership Changes": ["Item 5.01", "Item 5.02"],
    "Financial Condition": ["Item 2.02", "Item 2.03", "Item 2.04", "Item 2.06"],
    "Material Agreements": ["Item 1.01", "Item 1.02"],
    "Corporate Actions": ["Item 5.03", "Item 5.07", "Item 5.08"],
    "Asset Transactions": ["Item 2.01"],
    "Accounting Issues": ["Item 4.01", "Item 4.02"],
    "Financial Restructuring": ["Item 1.03", "Item 2.05"],
    "Securities Matters": ["Item 3.01", "Item 3.02", "Item 3.03"],
    "Other Disclosures": ["Item 7.01", "Item 8.01", "Item 9.01"]
}

# Define the standard metrics we calculate for events
STANDARD_METRICS = {
    "price": [
        "1-day price change",
        "5-day price change", 
        "30-day price change",
        "30-day volatility",
        "trading volume change"
    ],
    "market": [
        "s&p 500 change",
        "sector index change",
        "10y treasury yield change",
        "yield curve change"
    ]
}

def load_8k_filings(ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Load 8-K filings from the dataStore.
    
    Args:
        ticker: Optional ticker symbol to filter by
        
    Returns:
        DataFrame containing 8-K filings
    """
    try:
        # If ticker provided, load company-specific filings
        if ticker:
            file_path = Config.COMPANIES_DIR / ticker.upper() / "filings_8k.parquet"
            if file_path.exists():
                filings = pd.read_parquet(file_path)
                logger.info(f"Loaded {len(filings)} 8-K filings for {ticker}")
                return filings
            else:
                logger.warning(f"No 8-K filings found for {ticker}")
                return pd.DataFrame()
        
        # Otherwise, load all material events
        file_path = Config.EVENTS_DIR / "material_events.parquet"
        if file_path.exists():
            filings = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(filings)} material events")
            return filings
        else:
            # Fallback: try to load and combine filings from all companies
            logger.warning("No material events found in events directory. Trying to combine company filings.")
            
            # Get all company directories
            company_dirs = [d for d in Config.COMPANIES_DIR.glob("*") if d.is_dir()]
            
            all_filings = []
            for company_dir in company_dirs:
                ticker = company_dir.name
                filing_path = company_dir / "filings_8k.parquet"
                
                if filing_path.exists():
                    try:
                        company_filings = pd.read_parquet(filing_path)
                        if not company_filings.empty:
                            all_filings.append(company_filings)
                    except Exception as e:
                        logger.error(f"Error loading filings for {ticker}: {e}")
            
            if all_filings:
                combined = pd.concat(all_filings, ignore_index=True)
                logger.info(f"Combined {len(combined)} 8-K filings from {len(all_filings)} companies")
                return combined
            else:
                logger.warning("No 8-K filings found")
                return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading 8-K filings: {e}", exc_info=True)
        return pd.DataFrame()

def extract_8k_items(filing_text: str) -> List[str]:
    """
    Extract 8-K item numbers from filing text.
    
    Args:
        filing_text: Text content of the 8-K filing
        
    Returns:
        List of item numbers found in the filing
    """
    # Return empty list if text is None or not a string
    if not isinstance(filing_text, str):
        return []
    
    # Pattern to match Item X.XX
    item_pattern = r"Item\s+(\d+\.\d+)"
    
    # Find all matches
    items = re.findall(item_pattern, filing_text)
    
    # Format as Item X.XX and remove duplicates
    formatted_items = [f"Item {item}" for item in items]
    unique_items = list(set(formatted_items))
    
    return unique_items

def categorize_8k_filing(filing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Categorize an 8-K filing based on its items and content.
    
    Args:
        filing: Dictionary containing filing information
        
    Returns:
        Dictionary with filing information and categorization
    """
    try:
        result = filing.copy()
        
        # Extract items if title or description is available
        items = []
        
        # Check title field
        if 'title' in filing and isinstance(filing['title'], str):
            items.extend(extract_8k_items(filing['title']))
        
        # Check description field if available
        if 'description' in filing and isinstance(filing['description'], str):
            items.extend(extract_8k_items(filing['description']))
        
        # Check form type field
        if 'form_type' in filing and filing['form_type'] == '8-K' and not items:
            # If we know it's an 8-K but couldn't find items, check for keywords
            if 'title' in filing and isinstance(filing['title'], str):
                title = filing['title'].lower()
                
                # Map keywords to likely item numbers
                keyword_mapping = {
                    "appoint": "Item 5.02",
                    "depart": "Item 5.02",
                    "resign": "Item 5.02",
                    "director": "Item 5.02",
                    "ceo": "Item 5.02",
                    "cfo": "Item 5.02",
                    "officer": "Item 5.02",
                    "material": "Item 1.01",
                    "agreement": "Item 1.01",
                    "contract": "Item 1.01",
                    "financial": "Item 2.02",
                    "results": "Item 2.02",
                    "earnings": "Item 2.02",
                    "amendment": "Item 5.03",
                    "acquisition": "Item 2.01",
                    "disposition": "Item 2.01",
                    "assets": "Item 2.01",
                    "impair": "Item 2.06",
                    "debt": "Item 2.03",
                    "credit": "Item 2.03",
                    "obligation": "Item 2.03",
                    "bankruptcy": "Item 1.03",
                    "delisting": "Item 3.01",
                    "accountant": "Item 4.01",
                    "auditor": "Item 4.01"
                }
                
                for keyword, item in keyword_mapping.items():
                    if keyword in title:
                        items.append(item)
        
        # Remove duplicates while preserving order
        unique_items = []
        for item in items:
            if item not in unique_items:
                unique_items.append(item)
        
        # Add items to result
        result['items'] = unique_items
        
        # Add item descriptions
        result['item_descriptions'] = [ITEM_CATEGORIES.get(item, "Unknown Item") for item in unique_items]
        
        # Add business impact category
        business_categories = set()
        for item in unique_items:
            for category, items in BUSINESS_IMPACT_CATEGORIES.items():
                if item in items:
                    business_categories.add(category)
        
        # If no categories matched, add "Other"
        if not business_categories and unique_items:
            business_categories.add("Other Disclosures")
        
        result['business_categories'] = list(business_categories)
        
        return result
    except Exception as e:
        logger.error(f"Error categorizing filing: {e}", exc_info=True)
        # Return original filing with empty categories as fallback
        filing['items'] = []
        filing['item_descriptions'] = []
        filing['business_categories'] = []
        return filing

def create_event_timeline(filings: pd.DataFrame, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None,
                         tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create a timeline of events from 8-K filings.
    
    Args:
        filings: DataFrame containing 8-K filings
        start_date: Optional start date for filtering (YYYY-MM-DD)
        end_date: Optional end date for filtering (YYYY-MM-DD)
        tickers: Optional list of ticker symbols to filter by
        
    Returns:
        DataFrame containing the event timeline
    """
    if filings.empty:
        return pd.DataFrame()
    
    try:
        # Make a copy to avoid modifying the original
        timeline = filings.copy()
        
        # Make sure date column is datetime
        date_column = None
        for col in ['date', 'acceptedDate', 'filingDate']:
            if col in timeline.columns:
                timeline[col] = pd.to_datetime(timeline[col], errors='coerce')
                date_column = col
                break
        
        if date_column is None:
            logger.error("No date column found in filings data")
            return pd.DataFrame()
        
        # Apply date filters
        if start_date:
            start_dt = pd.to_datetime(start_date, errors='coerce')
            if pd.notna(start_dt):
                timeline = timeline[timeline[date_column] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date, errors='coerce')
            if pd.notna(end_dt):
                timeline = timeline[timeline[date_column] <= end_dt]
        
        # Apply ticker filter
        if tickers:
            ticker_column = None
            for col in ['ticker', 'symbol']:
                if col in timeline.columns:
                    ticker_column = col
                    break
            
            if ticker_column:
                timeline = timeline[timeline[ticker_column].isin(tickers)]
        
        # Categorize each filing
        categorized_filings = []
        for _, filing in timeline.iterrows():
            categorized = categorize_8k_filing(filing.to_dict())
            categorized_filings.append(categorized)
        
        # Convert back to DataFrame
        result = pd.DataFrame(categorized_filings)
        
        # Sort by date
        if date_column in result.columns:
            result = result.sort_values(date_column)
        
        return result
    except Exception as e:
        logger.error(f"Error creating event timeline: {e}", exc_info=True)
        return pd.DataFrame()

def calculate_price_metrics(ticker: str, 
                           event_date: Union[str, datetime], 
                           window_days: int = 30) -> Dict[str, float]:
    """
    Calculate stock price metrics for a company around an event date.
    
    Args:
        ticker: Company ticker symbol
        event_date: Date of the event
        window_days: Number of days to analyze after the event
        
    Returns:
        Dictionary of price metrics
    """
    try:
        # Convert event_date to datetime if it's a string
        if isinstance(event_date, str):
            event_date = pd.to_datetime(event_date)
        
        # Load price data
        price_file = Config.COMPANIES_DIR / ticker.upper() / "prices.parquet"
        if not price_file.exists():
            logger.debug(f"No price data found for {ticker}")
            return {}
        
        prices = pd.read_parquet(price_file)
        
        # Ensure date is datetime
        prices['date'] = pd.to_datetime(prices['date'], errors='coerce')
        
        # Drop rows with invalid dates
        prices = prices.dropna(subset=['date'])
        
        # Filter to time window
        start_dt = event_date - timedelta(days=window_days)
        end_dt = event_date + timedelta(days=window_days)
        
        window_prices = prices[
            (prices['date'] >= start_dt) &
            (prices['date'] <= end_dt)
        ].sort_values('date')
        
        if len(window_prices) < 2:
            logger.debug(f"Insufficient price data for {ticker} around {event_date}")
            return {}
        
        # Find pre-event and post-event prices
        pre_event = window_prices[window_prices['date'] < event_date]
        post_event = window_prices[window_prices['date'] >= event_date]
        
        if pre_event.empty or post_event.empty:
            logger.debug(f"Missing pre-event or post-event data for {ticker}")
            return {}
        
        # Get base price (last price before event)
        base_price = pre_event.iloc[-1]['close']
        
        # Calculate metrics
        metrics = {}
        
        # 1-day price change
        if len(post_event) >= 1:
            day1_price = post_event.iloc[0]['close']
            metrics["1-day price change"] = (day1_price - base_price) / base_price
        
        # 5-day price change
        if len(post_event) >= 5:
            day5_idx = min(4, len(post_event) - 1)
            day5_price = post_event.iloc[day5_idx]['close']
            metrics["5-day price change"] = (day5_price - base_price) / base_price
        
        # 30-day price change (or as far as data allows)
        if len(post_event) >= 1:
            last_idx = min(window_days - 1, len(post_event) - 1)
            last_price = post_event.iloc[last_idx]['close']
            metrics["30-day price change"] = (last_price - base_price) / base_price
        
        # Calculate volatility
        if len(post_event) >= 5:
            price_returns = post_event['close'].pct_change().dropna()
            if not price_returns.empty:
                metrics["30-day volatility"] = price_returns.std() * np.sqrt(252)  # Annualized
        
        # Volume change
        if 'volume' in pre_event.columns and 'volume' in post_event.columns:
            pre_volume = pre_event['volume'].mean()
            post_volume = post_event['volume'].mean()
            if pd.notna(pre_volume) and pd.notna(post_volume) and pre_volume > 0:
                metrics["trading volume change"] = (post_volume - pre_volume) / pre_volume
        
        # Store raw price data for reference
        metrics["base_price"] = base_price
        if not post_event.empty:
            metrics["last_price"] = post_event.iloc[-1]['close']
        
        # Add event date for reference
        metrics["event_date"] = event_date.strftime("%Y-%m-%d")
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating price metrics for {ticker}: {e}", exc_info=True)
        return {}

def calculate_market_metrics(event_date: Union[str, datetime], 
                            window_days: int = 30) -> Dict[str, float]:
    """
    Calculate market-wide metrics around an event date.
    
    Args:
        event_date: Date of the event
        window_days: Number of days to analyze
        
    Returns:
        Dictionary of market metrics
    """
    try:
        # Convert event_date to datetime if it's a string
        if isinstance(event_date, str):
            event_date = pd.to_datetime(event_date)
        
        metrics = {}
        
        # Load market index data (S&P 500)
        index_file = Config.MARKET_DIR / "indices.parquet"
        if index_file.exists():
            indices = pd.read_parquet(index_file)
            indices['date'] = pd.to_datetime(indices['date'], errors='coerce')
            
            # Filter S&P 500
            sp500 = indices[indices['symbol'] == '^GSPC']
            
            if not sp500.empty:
                # Filter to time window
                start_dt = event_date - timedelta(days=window_days)
                end_dt = event_date + timedelta(days=window_days)
                
                window_sp500 = sp500[
                    (sp500['date'] >= start_dt) &
                    (sp500['date'] <= end_dt)
                ].sort_values('date')
                
                # Calculate S&P 500 change
                if len(window_sp500) >= 2:
                    pre_event = window_sp500[window_sp500['date'] < event_date]
                    post_event = window_sp500[window_sp500['date'] >= event_date]
                    
                    if not pre_event.empty and not post_event.empty:
                        base_price = pre_event.iloc[-1]['close']
                        last_idx = min(window_days - 1, len(post_event) - 1)
                        end_price = post_event.iloc[last_idx]['close']
                        
                        metrics["s&p 500 change"] = (end_price - base_price) / base_price
        
        # Load treasury data
        treasury_file = Config.MARKET_DIR / "treasury.parquet"
        if treasury_file.exists():
            treasury = pd.read_parquet(treasury_file)
            treasury['date'] = pd.to_datetime(treasury['date'], errors='coerce')
            
            # Filter to time window
            start_dt = event_date - timedelta(days=window_days)
            end_dt = event_date + timedelta(days=window_days)
            
            window_treasury = treasury[
                (treasury['date'] >= start_dt) &
                (treasury['date'] <= end_dt)
            ].sort_values('date')
            
            if len(window_treasury) >= 2:
                pre_event = window_treasury[window_treasury['date'] < event_date]
                post_event = window_treasury[window_treasury['date'] >= event_date]
                
                if not pre_event.empty and not post_event.empty:
                    # 10Y Treasury yield
                    if 'year10' in pre_event.columns and 'year10' in post_event.columns:
                        base_10y = pre_event.iloc[-1]['year10']
                        last_idx = min(window_days - 1, len(post_event) - 1)
                        end_10y = post_event.iloc[last_idx]['year10']
                        
                        metrics["10y treasury yield change"] = end_10y - base_10y
                    
                    # Yield curve (10Y-3M)
                    if 'year10' in pre_event.columns and 'month3' in pre_event.columns:
                        base_spread = pre_event.iloc[-1]['year10'] - pre_event.iloc[-1]['month3']
                        last_idx = min(window_days - 1, len(post_event) - 1)
                        end_spread = post_event.iloc[last_idx]['year10'] - post_event.iloc[last_idx]['month3']
                        
                        metrics["yield curve change"] = end_spread - base_spread
        
        # Add event date for reference
        metrics["event_date"] = event_date.strftime("%Y-%m-%d")
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating market metrics: {e}", exc_info=True)
        return {}

def calculate_sector_metrics(ticker: str, 
                            event_date: Union[str, datetime],
                            window_days: int = 30) -> Dict[str, float]:
    """
    Calculate sector-specific metrics around an event date.
    
    Args:
        ticker: Company ticker symbol
        event_date: Date of the event
        window_days: Number of days to analyze
        
    Returns:
        Dictionary of sector metrics
    """
    try:
        # This is a placeholder for sector analysis
        # In a more complete implementation, we would:
        # 1. Determine the company's sector
        # 2. Find peer companies in the same sector
        # 3. Calculate average price changes across peers
        
        # For now, return an empty dict
        return {"event_date": event_date if isinstance(event_date, str) else event_date.strftime("%Y-%m-%d")}
    except Exception as e:
        logger.error(f"Error calculating sector metrics for {ticker}: {e}", exc_info=True)
        return {}

def detect_event_clusters(timeline: pd.DataFrame, 
                          date_window: int = 7,
                          min_cluster_size: int = 2) -> List[Dict[str, Any]]:
    """
    Detect clusters of similar events across companies.
    
    Args:
        timeline: DataFrame containing the event timeline
        date_window: Number of days to consider for clustering
        min_cluster_size: Minimum number of events to form a cluster
        
    Returns:
        List of event clusters
    """
    if timeline.empty:
        return []
    
    try:
        # Determine date column
        date_column = None
        for col in ['date', 'acceptedDate', 'filingDate']:
            if col in timeline.columns:
                date_column = col
                break
        
        if date_column is None:
            logger.error("No date column found in timeline data")
            return []
        
        # Ensure date is datetime
        timeline[date_column] = pd.to_datetime(timeline[date_column], errors='coerce')
        
        # Sort by date
        timeline = timeline.sort_values(date_column)
        
        # Find ticker/symbol column
        ticker_column = None
        for col in ['ticker', 'symbol']:
            if col in timeline.columns:
                ticker_column = col
                break
        
        if ticker_column is None:
            logger.error("No ticker column found in timeline data")
            return []
        
        # Group events by business category
        clusters = []
        
        # Check if business_categories column exists
        if 'business_categories' not in timeline.columns:
            # Try to categorize if not already done
            categorized_filings = []
            for _, filing in timeline.iterrows():
                categorized = categorize_8k_filing(filing.to_dict())
                categorized_filings.append(categorized)
            
            timeline = pd.DataFrame(categorized_filings)
        
        # For each business category, find clusters
        all_categories = set()
        for cats in timeline['business_categories']:
            if isinstance(cats, list):
                all_categories.update(cats)
        
        for category in all_categories:
            # Filter timeline for this category
            category_events = timeline[timeline['business_categories'].apply(
                lambda x: category in x if isinstance(x, list) else False
            )]
            
            if len(category_events) < min_cluster_size:
                continue
            
            # Find date clusters
            date_clusters = []
            current_cluster = []
            
            for i, (_, event) in enumerate(category_events.iterrows()):
                if i == 0:
                    current_cluster.append(event)
                    continue
                
                # Check if this event is within the window of the first event in cluster
                first_event_date = current_cluster[0][date_column]
                current_event_date = event[date_column]
                
                if (current_event_date - first_event_date).days <= date_window:
                    current_cluster.append(event)
                else:
                    # Save current cluster if it's large enough and start a new one
                    if len(current_cluster) >= min_cluster_size:
                        date_clusters.append(current_cluster)
                    current_cluster = [event]
            
            # Add the last cluster if it's large enough
            if len(current_cluster) >= min_cluster_size:
                date_clusters.append(current_cluster)
            
            # Convert clusters to dictionary format
            for cluster in date_clusters:
                # Get unique companies in cluster
                companies = list(set([event[ticker_column] for event in cluster]))
                
                # Get date range
                start_date = min([event[date_column] for event in cluster])
                end_date = max([event[date_column] for event in cluster])
                
                clusters.append({
                    'category': category,
                    'companies': companies,
                    'event_count': len(cluster),
                    'start_date': start_date,
                    'end_date': end_date,
                    'date_range_days': (end_date - start_date).days,
                    'events': cluster
                })
        
        # Sort clusters by event count (descending)
        clusters.sort(key=lambda x: x['event_count'], reverse=True)
        
        return clusters
    except Exception as e:
        logger.error(f"Error detecting event clusters: {e}", exc_info=True)
        return []

def build_event_correlation_network(timeline: pd.DataFrame,
                                  min_correlation: float = 0.3,
                                  time_window: int = 90) -> Dict[str, Any]:
    """
    Build a network of companies based on correlated event patterns.
    
    Args:
        timeline: DataFrame containing the event timeline
        min_correlation: Minimum correlation to consider a connection
        time_window: Time window in days to consider for correlation
        
    Returns:
        Dictionary containing network data
    """
    if timeline.empty:
        return {"nodes": [], "edges": []}
    
    try:
        # Find date and ticker columns
        date_column = None
        for col in ['date', 'acceptedDate', 'filingDate']:
            if col in timeline.columns:
                date_column = col
                break
        
        ticker_column = None
        for col in ['ticker', 'symbol']:
            if col in timeline.columns:
                ticker_column = col
                break
        
        if date_column is None or ticker_column is None:
            logger.error("Missing required columns in timeline data")
            return {"nodes": [], "edges": []}
        
        # Ensure date is datetime
        timeline[date_column] = pd.to_datetime(timeline[date_column], errors='coerce')
        
        # Get unique companies
        companies = timeline[ticker_column].unique()
        
        # Create time bins (e.g., weeks)
        min_date = timeline[date_column].min()
        max_date = timeline[date_column].max()
        
        if pd.isna(min_date) or pd.isna(max_date):
            logger.error("Invalid date range in timeline data")
            return {"nodes": [], "edges": []}
        
        # Create weekly bins
        bins = pd.date_range(start=min_date, end=max_date, freq='W')
        
        # Create a company-time matrix
        company_time_matrix = {}
        
        for company in companies:
            company_events = timeline[timeline[ticker_column] == company]
            
            # Create a vector of event counts per time bin
            time_vector = []
            
            for i in range(len(bins) - 1):
                start_bin = bins[i]
                end_bin = bins[i + 1]
                
                # Count events in this bin
                bin_events = company_events[
                    (company_events[date_column] >= start_bin) &
                    (company_events[date_column] < end_bin)
                ]
                
                time_vector.append(len(bin_events))
            
            company_time_matrix[company] = time_vector
        
        # Calculate correlations between companies
        correlations = {}
        
        for i, company1 in enumerate(companies):
            vector1 = company_time_matrix.get(company1, [])
            
            if not vector1:
                continue
            
            for j, company2 in enumerate(companies):
                if i >= j:  # Avoid duplicate calculations
                    continue
                
                vector2 = company_time_matrix.get(company2, [])
                
                if not vector2 or len(vector1) != len(vector2):
                    continue
                
                # Calculate correlation
                correlation = np.corrcoef(vector1, vector2)[0, 1]
                
                if pd.notna(correlation) and abs(correlation) >= min_correlation:
                    correlations[(company1, company2)] = correlation
        
        # Create network
        G = nx.Graph()
        
        # Add nodes (companies)
        for company in companies:
            # Get company events
            company_events = timeline[timeline[ticker_column] == company]
            
            # Get business categories
            categories = set()
            if 'business_categories' in company_events.columns:
                for cats in company_events['business_categories']:
                    if isinstance(cats, list):
                        categories.update(cats)
            
            # Add node
            G.add_node(company, categories=list(categories), event_count=len(company_events))
        
        # Add edges (correlations)
        for (company1, company2), correlation in correlations.items():
            G.add_edge(company1, company2, weight=correlation)
        
        # Convert to dictionary
        nodes = [
            {
                "id": node,
                "categories": G.nodes[node].get("categories", []),
                "event_count": G.nodes[node].get("event_count", 0)
            }
            for node in G.nodes
        ]
        
        edges = [
            {
                "source": source,
                "target": target,
                "correlation": G.edges[source, target]["weight"]
            }
            for source, target in G.edges
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "min_correlation": min_correlation,
            "time_window": time_window
        }
    except Exception as e:
        logger.error(f"Error building event correlation network: {e}", exc_info=True)
        return {"nodes": [], "edges": []}

def prepare_event_data_for_analysis(timeline: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """
    Prepare event data with all necessary metrics for advanced analysis.
    
    Args:
        timeline: DataFrame containing event timeline
        window_days: Number of days to analyze around each event
        
    Returns:
        DataFrame with events and their associated metrics
    """
    if timeline.empty:
        return pd.DataFrame()
    
    try:
        # Find date and ticker columns
        date_column = None
        for col in ['date', 'acceptedDate', 'filingDate']:
            if col in timeline.columns:
                date_column = col
                break
        
        ticker_column = None
        for col in ['ticker', 'symbol']:
            if col in timeline.columns:
                ticker_column = col
                break
        
        if date_column is None or ticker_column is None:
            logger.error("Missing required columns in timeline data")
            return pd.DataFrame()
        
        # Ensure date is datetime
        timeline[date_column] = pd.to_datetime(timeline[date_column], errors='coerce')
        
        # Create a new DataFrame to store enriched event data
        enriched_events = []
        
        # Process each event
        for i, (_, event) in enumerate(timeline.iterrows()):
            # Show progress for large datasets
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i} events out of {len(timeline)}")
            
            ticker = event[ticker_column]
            event_date = event[date_column]
            
            if pd.isna(event_date):
                continue
            
            # Get base event data
            enriched_event = event.to_dict()
            
            # Calculate metrics
            price_metrics = calculate_price_metrics(ticker, event_date, window_days)
            market_metrics = calculate_market_metrics(event_date, window_days)
            sector_metrics = calculate_sector_metrics(ticker, event_date, window_days)
            
            # Combine metrics
            metrics = {}
            metrics.update(price_metrics)
            metrics.update(market_metrics)
            metrics.update(sector_metrics)
            
            # Add metrics to event
            enriched_event['metrics'] = metrics
            
            # Add a metrics prefix to each metric for easy filtering
            for metric_name, value in metrics.items():
                enriched_event[f"metric_{metric_name}"] = value
            
            enriched_events.append(enriched_event)
        
        # Convert to DataFrame
        result = pd.DataFrame(enriched_events)
        
        # Sort by date
        if date_column in result.columns:
            result = result.sort_values(date_column)
        
        return result
    except Exception as e:
        logger.error(f"Error preparing event data for analysis: {e}", exc_info=True)
        return pd.DataFrame()

# Utility functions
def get_event_impact(ticker: str, 
                   event_date: Union[str, datetime], 
                   window_days: int = 30) -> Dict[str, Any]:
    """
    Get the impact of an event on a company's stock price and market context.
    
    Args:
        ticker: Company ticker symbol
        event_date: Date of the event
        window_days: Number of days to analyze around the event
        
    Returns:
        Dictionary containing impact analysis
    """
    try:
        # Convert event_date to datetime if it's a string
        if isinstance(event_date, str):
            event_date = pd.to_datetime(event_date)
        
        # Get metrics
        price_metrics = calculate_price_metrics(ticker, event_date, window_days)
        market_metrics = calculate_market_metrics(event_date, window_days)
        
        # Combine results
        results = {
            'ticker': ticker,
            'event_date': event_date.strftime("%Y-%m-%d"),
            'price_metrics': price_metrics,
            'market_metrics': market_metrics,
            'status': 'success' if price_metrics else 'error'
        }
        
        if not price_metrics:
            results['message'] = 'No price data available or insufficient data'
        
        return results
    except Exception as e:
        logger.error(f"Error getting event impact for {ticker}: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'ticker': ticker,
            'event_date': event_date if isinstance(event_date, str) else event_date.strftime("%Y-%m-%d")
        }

def extract_event_timeline_stats(timeline: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract statistics about the event timeline.
    
    Args:
        timeline: DataFrame containing event timeline
        
    Returns:
        Dictionary containing timeline statistics
    """
    if timeline.empty:
        return {
            "event_count": 0,
            "company_count": 0,
            "category_counts": {},
            "date_range": None
        }
    
    try:
        # Find date and ticker columns
        date_column = None
        for col in ['date', 'acceptedDate', 'filingDate']:
            if col in timeline.columns:
                date_column = col
                break
        
        ticker_column = None
        for col in ['ticker', 'symbol']:
            if col in timeline.columns:
                ticker_column = col
                break
        
        stats = {}
        
        # Event count
        stats["event_count"] = len(timeline)
        
        # Company count
        if ticker_column:
            stats["company_count"] = timeline[ticker_column].nunique()
            
            # Companies by event count
            company_counts = timeline[ticker_column].value_counts().to_dict()
            stats["companies_by_event_count"] = company_counts
        
        # Category counts
        if 'business_categories' in timeline.columns:
            category_counts = {}
            
            for cats in timeline['business_categories']:
                if isinstance(cats, list):
                    for cat in cats:
                        if cat in category_counts:
                            category_counts[cat] += 1
                        else:
                            category_counts[cat] = 1
            
            stats["category_counts"] = category_counts
        
        # Date range
        if date_column:
            stats["date_range"] = {
                "start": timeline[date_column].min().strftime("%Y-%m-%d"),
                "end": timeline[date_column].max().strftime("%Y-%m-%d")
            }
        
        return stats
    except Exception as e:
        logger.error(f"Error extracting timeline stats: {e}", exc_info=True)
        return {
            "event_count": len(timeline),
            "error": str(e)
        }