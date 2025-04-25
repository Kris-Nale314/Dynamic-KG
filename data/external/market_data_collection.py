"""
Dynamic-KG Data Collection Script

This script collects financial and economic data from the Financial Modeling Prep API
and organizes it in the dataStore structure for the Dynamic-KG project.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import requests
from tqdm import tqdm
import sys

# Add the project root directory to the Python path for importing config
root_dir = str(Path(__file__).parents[2].absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the config from the root
from config import Config

# Set up logger
logger = Config.get_logger(__name__)

class FMPDataCollector:
    """Collects and organizes data from the Financial Modeling Prep API."""
    
    def __init__(self, api_key=None):
        """Initialize the collector with API key and base URL."""
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP API key is required. Set FMP_API_KEY environment variable or pass as parameter.")
        
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.base_url_v4 = "https://financialmodelingprep.com/api/v4"
        self.request_delay = 0.5  # Delay between API requests to avoid rate limiting
    
    def api_request(self, endpoint, params=None, version="v3"):
        """Make a request to the FMP API with error handling and rate limiting."""
        if params is None:
            params = {}
        
        params["apikey"] = self.api_key
        
        # Select base URL based on version
        base_url = self.base_url if version == "v3" else self.base_url_v4
        url = f"{base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Simple rate limiting
            time.sleep(self.request_delay)
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in API request to {endpoint}: {e}")
            return None
    
    def collect_company_profile(self, ticker):
        """Collect and save company profile data."""
        logger.info(f"Collecting profile data for {ticker}")
        
        data = self.api_request(f"profile/{ticker}")
        if data and len(data) > 0:
            # Create output directory
            company_dir = Config.get_company_dir(ticker)
            
            # Save as JSON for human readability
            with open(company_dir / "profile.json", "w") as f:
                json.dump(data[0], f, indent=2)
            
            logger.info(f"Profile data saved for {ticker}")
            return True
        return False
    
    def collect_company_financials(self, ticker, years=3):
        """Collect and save company financial statements."""
        logger.info(f"Collecting financial data for {ticker}")
        
        # Define statement types to collect
        statement_types = {
            "income-statement": "Income Statement",
            "balance-sheet-statement": "Balance Sheet",
            "cash-flow-statement": "Cash Flow",
            "key-metrics": "Key Metrics",
            "ratios": "Financial Ratios"
        }
        
        # Create DataFrame to store all financial data
        all_financials = pd.DataFrame()
        
        for endpoint, name in statement_types.items():
            # Collect quarterly data
            quarterly = self.api_request(f"{endpoint}/{ticker}", {"period": "quarter", "limit": years * 4})
            
            if quarterly:
                # Convert to DataFrame
                df = pd.DataFrame(quarterly)
                df['statement_type'] = name
                df['period_type'] = 'quarterly'
                
                # Append to main DataFrame
                all_financials = pd.concat([all_financials, df])
            
            # Collect annual data
            annual = self.api_request(f"{endpoint}/{ticker}", {"limit": years})
            
            if annual:
                # Convert to DataFrame
                df = pd.DataFrame(annual)
                df['statement_type'] = name
                df['period_type'] = 'annual'
                
                # Append to main DataFrame
                all_financials = pd.concat([all_financials, df])
        
        # Save all financial data as parquet file
        if not all_financials.empty:
            company_dir = Config.get_company_dir(ticker)
            all_financials.to_parquet(company_dir / "financials.parquet", engine='pyarrow')
            logger.info(f"Financial data saved for {ticker}")
            return True
        
        return False
    
    def collect_historical_prices(self, ticker, years=3):
        """Collect and save historical stock prices."""
        logger.info(f"Collecting price data for {ticker}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        data = self.api_request(f"historical-price-full/{ticker}", {
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d")
        })
        
        if data and "historical" in data:
            # Convert to DataFrame
            df = pd.DataFrame(data["historical"])
            df['ticker'] = ticker
            
            # Save as parquet file
            company_dir = Config.get_company_dir(ticker)
            df.to_parquet(company_dir / "prices.parquet", engine='pyarrow')
            
            logger.info(f"Price data saved for {ticker}")
            return True
        
        return False
    
    def collect_earnings_data(self, ticker, quarters=12):
        """Collect earnings history and call transcripts."""
        logger.info(f"Collecting earnings data for {ticker}")
        
        # Collect earnings history
        earnings_history = self.api_request(f"historical/earning_calendar/{ticker}", {"limit": quarters})
        
        if earnings_history:
            # Convert to DataFrame
            df = pd.DataFrame(earnings_history)
            
            # Save as parquet file
            company_dir = Config.get_company_dir(ticker)
            df.to_parquet(company_dir / "earnings.parquet", engine='pyarrow')
        
        # Collect earnings call transcripts
        transcripts = []
        
        for i in range(min(quarters, 12)):  # Limit to reasonable number
            transcript = self.api_request(f"earning_call_transcript/{ticker}", {"quarter": i + 1})
            
            if transcript and len(transcript) > 0:
                # Extract transcript data
                transcripts.append({
                    'ticker': ticker,
                    'quarter': transcript[0].get('quarter'),
                    'year': transcript[0].get('year'),
                    'date': transcript[0].get('date'),
                    'content': transcript[0].get('content', '')
                })
        
        if transcripts:
            # Convert to DataFrame
            df = pd.DataFrame(transcripts)
            
            # Save as parquet file
            company_dir = Config.get_company_dir(ticker)
            df.to_parquet(company_dir / "transcripts.parquet", engine='pyarrow')
        
        logger.info(f"Earnings data saved for {ticker}")
        return True
    
    def collect_sec_filings(self, ticker, years=3):
        """Collect SEC filing metadata."""
        logger.info(f"Collecting SEC filing data for {ticker}")
        
        # Get filing metadata
        filings = self.api_request(f"sec_filings/{ticker}", {"limit": years * 20})  # Approximation
        
        if filings:
            # Convert to DataFrame
            df = pd.DataFrame(filings)
            
            # Save as parquet file
            company_dir = Config.get_company_dir(ticker)
            df.to_parquet(company_dir / "filings.parquet", engine='pyarrow')
            
            logger.info(f"SEC filing data saved for {ticker}")
            return True
        
        return False
    
    def collect_8k_filings(self, ticker, years=3):
        """Collect 8-K filings specifically."""
        logger.info(f"Collecting 8-K filings for {ticker}")
        
        # Get 8-K filing metadata
        all_filings = self.api_request(f"sec_filings/{ticker}", {"limit": years * 50})  # Get more to filter
        
        if all_filings:
            # Filter for 8-K filings
            filings_8k = [filing for filing in all_filings if filing.get('type') == '8-K']
            
            if filings_8k:
                # Convert to DataFrame
                df = pd.DataFrame(filings_8k)
                
                # Save as parquet file
                company_dir = Config.get_company_dir(ticker)
                df.to_parquet(company_dir / "filings_8k.parquet", engine='pyarrow')
                
                logger.info(f"8-K filings saved for {ticker} ({len(filings_8k)} filings)")
                return True
        
        logger.warning(f"No 8-K filings found for {ticker}")
        return False
    
    def collect_company_news(self, ticker, limit=25):
        """Collect a limited set of company news articles."""
        logger.info(f"Collecting news data for {ticker}")
        
        # Get news articles
        news = self.api_request("stock_news", {"tickers": ticker, "limit": limit})
        
        if news:
            # Convert to DataFrame
            df = pd.DataFrame(news)
            
            # Save as parquet file
            company_dir = Config.get_company_dir(ticker)
            df.to_parquet(company_dir / "news.parquet", engine='pyarrow')
            
            logger.info(f"News data saved for {ticker}")
            return True
        
        return False
    
    def collect_economic_indicators(self, years=3):
        """Collect economic indicators."""
        logger.info("Collecting economic indicators")
        
        # Define indicators to collect
        indicators = [
            "GDP", "realGDP", "nominalPotentialGDP", "realGDPPerCapita", 
            "federalFunds", "CPI", "inflationRate", "inflation", 
            "retailSales", "consumerSentiment", "durableGoods", 
            "unemploymentRate", "totalNonfarmPayroll", "initialClaims", 
            "industrialProductionTotalIndex", "newPrivatelyOwnedHousingUnitsStartedTotalUnits", 
            "totalVehicleSales", "retailMoneyFunds", "smoothedUSRecessionProbabilities"
        ]
        
        # Collect data for each indicator
        all_data = pd.DataFrame()
        
        for indicator in tqdm(indicators, desc="Economic Indicators"):
            data = self.api_request(f"economic/{indicator}", {"limit": years * 12})  # Monthly data approximation
            
            if data:
                # Convert to DataFrame
                df = pd.DataFrame(data)
                df['indicator'] = indicator
                
                # Append to main DataFrame
                all_data = pd.concat([all_data, df])
        
        # Save as parquet file
        if not all_data.empty:
            os.makedirs(Config.MARKET_DIR, exist_ok=True)
            all_data.to_parquet(Config.MARKET_DIR / "economic.parquet", engine='pyarrow')
            
            logger.info("Economic indicators saved")
            return True
        
        return False
    
    def collect_treasury_rates(self, years=3):
        """Collect treasury rates."""
        logger.info("Collecting treasury rates")
        
        # Get treasury rates
        data = self.api_request("treasury", {"limit": years * 250})  # Daily data approximation
        
        if data:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Save as parquet file
            os.makedirs(Config.MARKET_DIR, exist_ok=True)
            df.to_parquet(Config.MARKET_DIR / "treasury.parquet", engine='pyarrow')
            
            logger.info("Treasury rates saved")
            return True
        
        return False
    
    def collect_market_indexes(self, years=3):
        """Collect market index data."""
        logger.info("Collecting market index data")
        
        # Define indexes to collect
        indexes = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow Jones, NASDAQ
        
        # Collect data for each index
        all_data = pd.DataFrame()
        
        for index_symbol in indexes:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            
            data = self.api_request(f"historical-price-full/{index_symbol}", {
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d")
            })
            
            if data and "historical" in data:
                # Convert to DataFrame
                df = pd.DataFrame(data["historical"])
                df['symbol'] = index_symbol
                
                # Append to main DataFrame
                all_data = pd.concat([all_data, df])
        
        # Save as parquet file
        if not all_data.empty:
            os.makedirs(Config.MARKET_DIR, exist_ok=True)
            all_data.to_parquet(Config.MARKET_DIR / "indices.parquet", engine='pyarrow')
            
            logger.info("Market index data saved")
            return True
        
        return False
    
    
    
    def collect_material_events(self):
        """Collect 8-K filings (material events) across companies."""
        logger.info("Collecting material events (8-K filings)")
        
        # Get RSS feed of 8-K filings
        data = self.api_request("rss_feed_8k", version="v4")
        
        if data:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Save as parquet file
            os.makedirs(Config.EVENTS_DIR, exist_ok=True)
            df.to_parquet(Config.EVENTS_DIR / "material_events.parquet", engine='pyarrow')
            
            logger.info("Material events saved")
            return True
        
        return False
    
    def collect_insider_trading(self, limit=1000):
        """Collect insider trading data across companies."""
        logger.info("Collecting insider trading data")
        
        # Get insider trading data
        data = self.api_request("insider-trading", {"limit": limit})
        
        if data:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Save as parquet file
            os.makedirs(Config.EVENTS_DIR, exist_ok=True)
            df.to_parquet(Config.EVENTS_DIR / "insider_trading.parquet", engine='pyarrow')
            
            logger.info("Insider trading data saved")
            return True
        
        return False
    
    def collect_company_data(self, ticker, years=3):
        """Collect all data for a specific company."""
        logger.info(f"Starting data collection for {ticker}")
        
        # Collect company-specific data
        self.collect_company_profile(ticker)
        self.collect_company_financials(ticker, years)
        self.collect_historical_prices(ticker, years)
        self.collect_earnings_data(ticker, years * 4)  # Quarterly data
        self.collect_sec_filings(ticker, years)
        self.collect_8k_filings(ticker, years)
        self.collect_company_news(ticker, 25)  # Limited news collection
        
        logger.info(f"Completed data collection for {ticker}")
        return True
    
    def collect_market_data(self, years=3):
        """Collect market-wide data."""
        logger.info("Starting market data collection")
        
        # Collect market data
        self.collect_economic_indicators(years)
        self.collect_treasury_rates(years)
        self.collect_market_indexes(years)
        self.collect_sector_pe_ratios()
        
        logger.info("Completed market data collection")
        return True
    
    def collect_event_data(self):
        """Collect event-based data."""
        logger.info("Starting event data collection")
        
        # Collect event data
        self.collect_material_events()
        self.collect_insider_trading()
        
        logger.info("Completed event data collection")
        return True
    
    def collect_all_data(self, tickers, years=3):
        """Collect all data for the Dynamic-KG POC."""
        logger.info("Starting comprehensive data collection")
        
        # Ensure data structure exists
        Config.initialize()
        
        # Collect company-specific data
        for ticker in tqdm(tickers, desc="Companies"):
            self.collect_company_data(ticker, years)
        
        # Collect market-wide data
        self.collect_market_data(years)
        
        # Collect event-based data
        self.collect_event_data()
        
        logger.info("Completed comprehensive data collection")
        return True


# Target companies to collect data for
TARGET_COMPANIES = [
    # Technology
    "GOOGL", "MSFT", "DELL", "NVDA", "TSM",
    # Consumer
    "TSLA", "AMZN", "WMT", "MCD", "NKE",
    # Industrial/Energy
    "SHEL", "CAT", "BA", "GE", "UNP", "CEG",
    # Financial
    "JPM", "GS", "V", "AXP", "BRK-A",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Professional Services
    "BAH", "ACN",
    # Real Estate
    "PLD", "SPG"
]

def main():
    """Main function to run data collection."""
    logger.info("Starting Dynamic-KG Data Collection")
    
    try:
        # Create FMP data collector
        collector = FMPDataCollector()
        
        # Collect all data
        collector.collect_all_data(TARGET_COMPANIES, years=3)
        
        logger.info("Data collection completed successfully")
    except Exception as e:
        logger.error(f"Error in data collection: {e}", exc_info=True)

if __name__ == "__main__":
    main()