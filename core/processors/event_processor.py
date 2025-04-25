"""
Event processor for analyzing 8-K filings and other material events.

This module provides functions for categorizing, analyzing, and extracting insights
from 8-K filings and other material events. It serves as the foundation for the
event timeline and impact analysis features of Dynamic-KG.
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

def load_8k_filings(ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Load 8-K filings from the dataStore.
    
    Args:
        ticker: Optional ticker symbol to filter by
        
    Returns:
        DataFrame containing 8-K filings
    """
    # If ticker provided, load company-specific filings
    if ticker:
        file_path = Config.COMPANIES_DIR / ticker.upper() / "filings_8k.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        else:
            logger.warning(f"No 8-K filings found for {ticker}")
            return pd.DataFrame()
    
    # Otherwise, load all material events
    file_path = Config.EVENTS_DIR / "material_events.parquet"
    if file_path.exists():
        return pd.read_parquet(file_path)
    else:
        logger.warning("No material events found")
        return pd.DataFrame()

def extract_8k_items(filing_text: str) -> List[str]:
    """
    Extract 8-K item numbers from filing text.
    
    Args:
        filing_text: Text content of the 8-K filing
        
    Returns:
        List of item numbers found in the filing
    """
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
    result = filing.copy()
    
    # Extract items if title or description is available
    items = []
    
    # Check title field
    if 'title' in filing and isinstance(filing['title'], str):
        items.extend(extract_8k_items(filing['title']))
    
    # Check description field if available
    if 'description' in filing and isinstance(filing['description'], str):
        items.extend(extract_8k_items(filing['description']))
    
    # If no items found, check for keywords in title
    if not items and 'title' in filing and isinstance(filing['title'], str):
        title = filing['title'].lower()
        
        # Map keywords to likely item numbers
        keyword_mapping = {
            "appoint": "Item 5.02",
            "depart": "Item 5.02",
            "resign": "Item 5.02",
            "material": "Item 1.01",
            "agreement": "Item 1.01",
            "financial": "Item 2.02",
            "results": "Item 2.02",
            "earnings": "Item 2.02",
            "amendment": "Item 5.03",
            "acquisition": "Item 2.01",
            "disposition": "Item 2.01",
            "impair": "Item 2.06"
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
    
    result['business_categories'] = list(business_categories)
    
    return result

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
    
    # Make a copy to avoid modifying the original
    timeline = filings.copy()
    
    # Make sure date column is datetime
    date_column = None
    for col in ['date', 'acceptedDate', 'filingDate']:
        if col in timeline.columns:
            timeline[col] = pd.to_datetime(timeline[col])
            date_column = col
            break
    
    if date_column is None:
        logger.error("No date column found in filings data")
        return pd.DataFrame()
    
    # Apply date filters
    if start_date:
        start_dt = pd.to_datetime(start_date)
        timeline = timeline[timeline[date_column] >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
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
    timeline[date_column] = pd.to_datetime(timeline[date_column])
    
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

def analyze_event_impact(ticker: str, 
                        event_date: str, 
                        window_days: int = 7) -> Dict[str, Any]:
    """
    Analyze the impact of an event on a company's stock price and peers.
    
    Args:
        ticker: Company ticker symbol
        event_date: Date of the event (YYYY-MM-DD)
        window_days: Number of days to analyze before and after the event
        
    Returns:
        Dictionary containing impact analysis
    """
    # Load company price data
    price_file = Config.COMPANIES_DIR / ticker.upper() / "prices.parquet"
    if not price_file.exists():
        logger.warning(f"No price data found for {ticker}")
        return {'status': 'error', 'message': 'No price data available'}
    
    prices = pd.read_parquet(price_file)
    
    # Ensure date is datetime
    prices['date'] = pd.to_datetime(prices['date'])
    
    # Filter to time window
    event_dt = pd.to_datetime(event_date)
    start_dt = event_dt - timedelta(days=window_days)
    end_dt = event_dt + timedelta(days=window_days)
    
    window_prices = prices[
        (prices['date'] >= start_dt) &
        (prices['date'] <= end_dt)
    ].sort_values('date')
    
    if len(window_prices) < 2:
        return {'status': 'error', 'message': 'Insufficient price data for analysis'}
    
    # Find pre-event and post-event prices
    pre_event = window_prices[window_prices['date'] < event_dt]
    post_event = window_prices[window_prices['date'] >= event_dt]
    
    if pre_event.empty or post_event.empty:
        return {'status': 'error', 'message': 'Missing pre-event or post-event data'}
    
    # Calculate price changes
    pre_price = pre_event.iloc[-1]['close']
    post_price = post_event.iloc[0]['close']
    
    price_change = post_price - pre_price
    price_change_pct = (price_change / pre_price) * 100
    
    # Calculate volume changes
    pre_volume = pre_event['volume'].mean()
    post_volume = post_event['volume'].mean()
    
    volume_change = post_volume - pre_volume
    volume_change_pct = (volume_change / pre_volume) * 100 if pre_volume > 0 else 0
    
    # Gather results
    results = {
        'ticker': ticker,
        'event_date': event_date,
        'pre_event_price': pre_price,
        'post_event_price': post_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'pre_event_volume': pre_volume,
        'post_event_volume': post_volume,
        'volume_change': volume_change,
        'volume_change_pct': volume_change_pct,
        'price_series': window_prices[['date', 'close', 'volume']].to_dict('records'),
        'status': 'success'
    }
    
    return results

def analyze_peer_reactions(ticker: str,
                         event_date: str,
                         peers: Optional[List[str]] = None,
                         window_days: int = 7) -> Dict[str, Any]:
    """
    Analyze how peers react to a company's material event.
    
    Args:
        ticker: Company ticker symbol
        event_date: Date of the event (YYYY-MM-DD)
        peers: Optional list of peer ticker symbols
        window_days: Number of days to analyze before and after the event
        
    Returns:
        Dictionary containing peer reaction analysis
    """
    # If peers not provided, try to load from relationships data
    if peers is None:
        peers_file = Config.RELATIONSHIPS_DIR / "peers.parquet"
        if peers_file.exists():
            peer_data = pd.read_parquet(peers_file)
            if 'source_ticker' in peer_data.columns:
                company_peers = peer_data[peer_data['source_ticker'] == ticker]
                if not company_peers.empty and 'ticker' in company_peers.columns:
                    peers = company_peers['ticker'].tolist()
    
    # If still no peers, return error
    if not peers:
        return {'status': 'error', 'message': 'No peer companies specified or found'}
    
    # Analyze the main company's event impact
    company_impact = analyze_event_impact(ticker, event_date, window_days)
    
    # Analyze each peer's reaction
    peer_impacts = []
    for peer in peers:
        peer_impact = analyze_event_impact(peer, event_date, window_days)
        if peer_impact['status'] == 'success':
            peer_impacts.append(peer_impact)
    
    # Calculate correlation between company and peers
    correlations = {}
    if company_impact['status'] == 'success' and peer_impacts:
        # Create DataFrames for each company's price series
        company_prices = pd.DataFrame(company_impact['price_series']).set_index('date')['close']
        
        for peer_impact in peer_impacts:
            peer_prices = pd.DataFrame(peer_impact['price_series']).set_index('date')['close']
            
            # Align the two series
            combined = pd.concat([company_prices, peer_prices], axis=1)
            combined.columns = [ticker, peer_impact['ticker']]
            
            # Drop rows with missing values
            combined = combined.dropna()
            
            # Calculate correlation if enough data points
            if len(combined) >= 3:
                corr = combined.corr().iloc[0, 1]
                correlations[peer_impact['ticker']] = corr
    
    # Gather results
    results = {
        'company': ticker,
        'event_date': event_date,
        'company_impact': company_impact,
        'peer_impacts': peer_impacts,
        'correlations': correlations,
        'status': 'success'
    }
    
    return results

def identify_leading_indicators(timeline: pd.DataFrame,
                               performance_data: pd.DataFrame,
                               lag_days: int = 90,
                               min_occurrences: int = 3) -> List[Dict[str, Any]]:
    """
    Identify event types that tend to precede performance changes.
    
    Args:
        timeline: DataFrame containing event timeline
        performance_data: DataFrame with company performance metrics
        lag_days: Maximum number of days to look for a relationship
        min_occurrences: Minimum number of occurrences to consider an indicator
        
    Returns:
        List of potential leading indicators
    """
    if timeline.empty or performance_data.empty:
        return []
    
    # Find date columns
    timeline_date_col = None
    for col in ['date', 'acceptedDate', 'filingDate']:
        if col in timeline.columns:
            timeline_date_col = col
            break
    
    perf_date_col = None
    for col in ['date', 'report_date', 'filing_date']:
        if col in performance_data.columns:
            perf_date_col = col
            break
    
    if timeline_date_col is None or perf_date_col is None:
        logger.error("Missing date columns")
        return []
    
    # Ensure dates are datetime
    timeline[timeline_date_col] = pd.to_datetime(timeline[timeline_date_col])
    performance_data[perf_date_col] = pd.to_datetime(performance_data[perf_date_col])
    
    # Find ticker columns
    timeline_ticker_col = None
    for col in ['ticker', 'symbol']:
        if col in timeline.columns:
            timeline_ticker_col = col
            break
    
    perf_ticker_col = None
    for col in ['ticker', 'symbol', 'company_id']:
        if col in performance_data.columns:
            perf_ticker_col = col
            break
    
    if timeline_ticker_col is None or perf_ticker_col is None:
        logger.error("Missing ticker columns")
        return []
    
    # Get performance metrics columns
    perf_metric_cols = [col for col in performance_data.columns 
                       if col not in [perf_date_col, perf_ticker_col]]
    
    if not perf_metric_cols:
        logger.error("No performance metric columns found")
        return []
    
    # Check for business categories
    if 'business_categories' not in timeline.columns:
        # Categorize events
        categorized_filings = []
        for _, filing in timeline.iterrows():
            categorized = categorize_8k_filing(filing.to_dict())
            categorized_filings.append(categorized)
        
        timeline = pd.DataFrame(categorized_filings)
    
    # For each company, event category, and performance metric, look for relationships
    indicators = []
    
    companies = timeline[timeline_ticker_col].unique()
    
    for company in companies:
        # Filter to this company
        company_events = timeline[timeline[timeline_ticker_col] == company]
        company_perf = performance_data[performance_data[perf_ticker_col] == company]
        
        if company_events.empty or company_perf.empty:
            continue
        
        # Get all business categories
        all_categories = set()
        for cats in company_events['business_categories']:
            if isinstance(cats, list):
                all_categories.update(cats)
        
        for category in all_categories:
            # Filter events to this category
            category_events = company_events[company_events['business_categories'].apply(
                lambda x: category in x if isinstance(x, list) else False
            )]
            
            if len(category_events) < min_occurrences:
                continue
            
            for metric in perf_metric_cols:
                # For each event, find the next performance change
                event_perf_changes = []
                
                for _, event in category_events.iterrows():
                    event_date = event[timeline_date_col]
                    
                    # Find performance data after this event within lag window
                    next_perf = company_perf[
                        (company_perf[perf_date_col] > event_date) &
                        (company_perf[perf_date_col] <= event_date + timedelta(days=lag_days))
                    ].sort_values(perf_date_col)
                    
                    if next_perf.empty:
                        continue
                    
                    # Calculate change
                    current_perf = company_perf[company_perf[perf_date_col] <= event_date]
                    if current_perf.empty:
                        continue
                    
                    current_value = current_perf.sort_values(perf_date_col).iloc[-1][metric]
                    next_value = next_perf.iloc[0][metric]
                    
                    if pd.isna(current_value) or pd.isna(next_value) or current_value == 0:
                        continue
                    
                    change = next_value - current_value
                    change_pct = change / current_value
                    
                    event_perf_changes.append({
                        'event_date': event_date,
                        'perf_date': next_perf.iloc[0][perf_date_col],
                        'lag_days': (next_perf.iloc[0][perf_date_col] - event_date).days,
                        'current_value': current_value,
                        'next_value': next_value,
                        'change': change,
                        'change_pct': change_pct
                    })
                
                if len(event_perf_changes) < min_occurrences:
                    continue
                
                # Calculate summary statistics
                changes = [c['change_pct'] for c in event_perf_changes]
                avg_change = sum(changes) / len(changes)
                median_change = sorted(changes)[len(changes) // 2]
                
                # Check if changes are consistently in one direction
                positive_changes = sum(1 for c in changes if c > 0)
                negative_changes = sum(1 for c in changes if c < 0)
                
                consistency = max(positive_changes, negative_changes) / len(changes)
                
                # Only consider as indicator if there's consistency
                if consistency >= 0.7:
                    indicators.append({
                        'company': company,
                        'category': category,
                        'metric': metric,
                        'occurrences': len(event_perf_changes),
                        'avg_lag_days': sum(c['lag_days'] for c in event_perf_changes) / len(event_perf_changes),
                        'avg_change': avg_change,
                        'median_change': median_change,
                        'consistency': consistency,
                        'direction': 'positive' if positive_changes > negative_changes else 'negative',
                        'events': event_perf_changes
                    })
    
    # Sort indicators by consistency and occurrences
    indicators.sort(key=lambda x: (x['consistency'], x['occurrences']), reverse=True)
    
    return indicators