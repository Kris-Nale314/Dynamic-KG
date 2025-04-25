# utils/data/processors.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

# Add the project root directory to the Python path for importing config
root_dir = str(Path(__file__).parents[2].absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the config from the root
from config import Config

def load_company_data(ticker, data_type="profile"):
    """Load data for a specific company and type."""
    company_dir = Config.get_company_dir(ticker)
    
    if data_type == "profile":
        profile_path = company_dir / "profile.json"
        if profile_path.exists():
            with open(profile_path, "r") as f:
                return json.load(f)
        return None
    
    elif data_type in ["financials", "prices", "earnings", "filings", "filings_8k", "news", "transcripts"]:
        file_path = company_dir / f"{data_type}.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        return None
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def load_market_data(data_type="economic"):
    """Load market-wide data for the specified type."""
    if data_type in ["economic", "treasury", "indices", "sector_pe"]:
        file_path = Config.MARKET_DIR / f"{data_type}.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        return None
    
    else:
        raise ValueError(f"Unknown market data type: {data_type}")

def load_event_data(data_type="material_events"):
    """Load event data for the specified type."""
    if data_type in ["material_events", "insider_trading"]:
        file_path = Config.EVENTS_DIR / f"{data_type}.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        return None
    
    else:
        raise ValueError(f"Unknown event data type: {data_type}")

def load_synthetic_data(data_type="loan_applications"):
    """Load synthetic data for the specified type."""
    file_path = Config.SYNTHETIC_DIR / f"{data_type}.parquet"
    if file_path.exists():
        return pd.read_parquet(file_path)
    return None

def get_financial_ratio(ticker, ratio_name, period_type="quarterly", periods=8):
    """Extract a specific financial ratio for a company over time."""
    financials = load_company_data(ticker, "financials")
    if financials is None:
        return None
    
    # Filter for ratio data
    ratio_data = financials[
        (financials["statement_type"] == "Financial Ratios") &
        (financials["period_type"] == period_type)
    ]
    
    if ratio_data.empty or ratio_name not in ratio_data.columns:
        return None
    
    # Sort and limit to specified number of periods
    ratio_data = ratio_data.sort_values("date", ascending=False).head(periods)
    
    # Extract the ratio values and dates
    result = ratio_data[["date", ratio_name]].copy()
    result["ticker"] = ticker
    
    return result

def compare_companies(tickers, metric, data_type="financials", period_type="quarterly", periods=8):
    """Compare multiple companies on a specific metric."""
    results = []
    
    for ticker in tickers:
        if data_type == "financials":
            # Special handling for financial data
            financials = load_company_data(ticker, data_type)
            if financials is None:
                continue
            
            # Filter for appropriate data
            filtered_data = financials[
                (financials["period_type"] == period_type)
            ]
            
            # Skip if metric not found
            if metric not in filtered_data.columns:
                continue
            
            # Sort and get recent periods
            filtered_data = filtered_data.sort_values("date", ascending=False).head(periods)
            
            # Extract data
            for _, row in filtered_data.iterrows():
                results.append({
                    "ticker": ticker,
                    "date": row["date"],
                    "metric": metric,
                    "value": row[metric]
                })
        
        elif data_type == "prices":
            # Handle price data
            prices = load_company_data(ticker, data_type)
            if prices is None:
                continue
            
            # Sort and get recent periods
            prices = prices.sort_values("date", ascending=False).head(periods)
            
            # Skip if metric not found
            if metric not in prices.columns:
                continue
            
            # Extract data
            for _, row in prices.iterrows():
                results.append({
                    "ticker": ticker,
                    "date": row["date"],
                    "metric": metric,
                    "value": row[metric]
                })
    
    if not results:
        return None
    
    return pd.DataFrame(results)

def get_market_context(date, window_days=30):
    """Get market context around a specific date."""
    # Load economic data
    economic = load_market_data("economic")
    treasury = load_market_data("treasury")
    indices = load_market_data("indices")
    
    context = {}
    
    if economic is not None:
        # Convert date strings to datetime
        economic["date"] = pd.to_datetime(economic["date"])
        
        # Filter to get data around the target date
        start_date = pd.to_datetime(date) - timedelta(days=window_days)
        end_date = pd.to_datetime(date) + timedelta(days=window_days)
        
        filtered_economic = economic[
            (economic["date"] >= start_date) &
            (economic["date"] <= end_date)
        ]
        
        # Group by indicator and get the most recent value
        if not filtered_economic.empty:
            context["economic"] = {}
            
            for indicator in filtered_economic["indicator"].unique():
                indicator_data = filtered_economic[filtered_economic["indicator"] == indicator]
                if not indicator_data.empty:
                    latest = indicator_data.sort_values("date", ascending=False).iloc[0]
                    context["economic"][indicator] = {
                        "date": latest["date"],
                        "value": latest["value"] if "value" in latest else None
                    }
    
    if treasury is not None:
        # Convert date to datetime
        treasury["date"] = pd.to_datetime(treasury["date"])
        
        # Filter to get data around the target date
        start_date = pd.to_datetime(date) - timedelta(days=window_days)
        end_date = pd.to_datetime(date) + timedelta(days=window_days)
        
        filtered_treasury = treasury[
            (treasury["date"] >= start_date) &
            (treasury["date"] <= end_date)
        ]
        
        # Get the closest treasury data
        if not filtered_treasury.empty:
            closest = filtered_treasury.iloc[(filtered_treasury['date'] - pd.to_datetime(date)).abs().argsort()[0]]
            context["treasury"] = {col: closest[col] for col in treasury.columns if col != "date"}
            context["treasury"]["date"] = closest["date"]
    
    if indices is not None:
        # Convert date to datetime
        indices["date"] = pd.to_datetime(indices["date"])
        
        # Filter to get data around the target date
        start_date = pd.to_datetime(date) - timedelta(days=window_days)
        end_date = pd.to_datetime(date) + timedelta(days=window_days)
        
        filtered_indices = indices[
            (indices["date"] >= start_date) &
            (indices["date"] <= end_date)
        ]
        
        # Group by symbol and get the most recent value
        if not filtered_indices.empty:
            context["indices"] = {}
            
            for symbol in filtered_indices["symbol"].unique():
                symbol_data = filtered_indices[filtered_indices["symbol"] == symbol]
                if not symbol_data.empty:
                    latest = symbol_data.sort_values("date", ascending=False).iloc[0]
                    context["indices"][symbol] = {
                        "date": latest["date"],
                        "close": latest["close"],
                        "volume": latest["volume"] if "volume" in latest else None
                    }
    
    return context