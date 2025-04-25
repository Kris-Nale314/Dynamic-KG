"""
03_8K_Raw_Explorer.py

This page directly examines 8-K filing data from the company folders in the dataStore,
focusing on raw data exploration before applying event processing logic.
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
import traceback
import os

# Add the project root directory to the Python path for importing config
root_dir = str(Path(__file__).parents[1].absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the config and utilities
from config import Config
from utils.ui.styles import set_page_config, card, badge
from utils.ui.session import SessionState

# Initialize configuration
Config.initialize()

# Set up logger
logger = Config.get_logger("8k_raw_explorer")
logger.info("8-K Raw Explorer page loaded")

# Set page configuration
set_page_config()

# Track page view
SessionState.track_page_view("8K Raw Explorer")

# Title and description
st.title("8-K Raw Data Explorer")
st.markdown("""
This page directly examines the raw 8-K filing data from the company folders in the dataStore.
It helps understand the data structure before applying event processing logic.
""")

# Define target companies
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

# Function to load raw 8-K data
def load_raw_8k_data(ticker):
    """Load raw 8-K filings for a specific ticker."""
    company_dir = Config.COMPANIES_DIR / ticker
    filings_path = company_dir / "filings_8k.parquet"
    
    if not filings_path.exists():
        return None
    
    try:
        return pd.read_parquet(filings_path)
    except Exception as e:
        st.error(f"Error loading 8-K filings for {ticker}: {str(e)}")
        return None

# Function to explore company directory
def explore_company_directory(ticker):
    """Explore the company directory structure and files."""
    company_dir = Config.COMPANIES_DIR / ticker
    
    if not company_dir.exists():
        return None
    
    result = {
        "directory": str(company_dir),
        "exists": True,
        "files": []
    }
    
    try:
        for file_path in company_dir.iterdir():
            file_info = {
                "name": file_path.name,
                "type": file_path.suffix,
                "size": file_path.stat().st_size if file_path.is_file() else None,
                "is_file": file_path.is_file()
            }
            result["files"].append(file_info)
    except Exception as e:
        st.error(f"Error exploring {ticker} directory: {str(e)}")
    
    return result

# Check if dataStore directory exists
if not Config.DATASTORE_DIR.exists():
    st.error("DataStore directory not found.")
    st.stop()

# Check if companies directory exists
if not Config.COMPANIES_DIR.exists():
    st.error("Companies directory not found in dataStore.")
    st.stop()

# Scan for available companies
available_companies = []
for company in TARGET_COMPANIES:
    company_dir = Config.COMPANIES_DIR / company
    if company_dir.exists():
        available_companies.append(company)

if not available_companies:
    st.error("No target companies found in the companies directory.")
    st.stop()

# Sidebar for company selection
st.sidebar.header("Company Selection")
selected_company = st.sidebar.selectbox(
    "Select a Company",
    options=available_companies,
    index=0
)

# Main content
st.header(f"Exploring 8-K Data for {selected_company}")

# Explore company directory
dir_info = explore_company_directory(selected_company)

if dir_info:
    with st.expander("Company Directory Structure", expanded=False):
        st.write(f"Directory: {dir_info['directory']}")
        
        # Create a table of files
        file_data = []
        for file in dir_info["files"]:
            size_kb = file["size"] / 1024 if file["size"] else None
            size_str = f"{size_kb:.2f} KB" if size_kb else "N/A"
            
            file_data.append({
                "Name": file["name"],
                "Type": file["type"],
                "Size": size_str,
                "Is File": "Yes" if file["is_file"] else "No"
            })
        
        if file_data:
            st.dataframe(pd.DataFrame(file_data))
        else:
            st.info("No files found in this directory.")

# Load 8-K filings data
filings_data = load_raw_8k_data(selected_company)

if filings_data is None:
    st.warning(f"No 8-K filings data found for {selected_company} (looking for filings_8k.parquet).")
    
    # Try to look for alternative files
    st.header("Alternative Files")
    alt_files = []
    company_dir = Config.COMPANIES_DIR / selected_company
    if company_dir.exists():
        for file in company_dir.glob("*.parquet"):
            alt_files.append(str(file.name))
    
    if alt_files:
        st.write("Found these parquet files instead:")
        for file in alt_files:
            st.write(f"- {file}")
        
        # Try to load filings.parquet if available
        if "filings.parquet" in alt_files:
            st.subheader("Examining filings.parquet")
            try:
                filings = pd.read_parquet(company_dir / "filings.parquet")
                # Display basic info
                st.write(f"Shape: {filings.shape}")
                st.write("Columns:", filings.columns.tolist())
                
                # Filter for 8-K filings if form_type column exists
                if 'form_type' in filings.columns:
                    eight_k_filings = filings[filings['form_type'] == '8-K']
                    if not eight_k_filings.empty:
                        st.write(f"Found {len(eight_k_filings)} 8-K filings in the general filings file.")
                        filings_data = eight_k_filings  # Use this as our data
                    else:
                        st.warning("No 8-K filings found in the general filings file.")
                else:
                    st.write("No form_type column found to filter for 8-K filings.")
            except Exception as e:
                st.error(f"Error examining filings.parquet: {str(e)}")
    else:
        st.info("No alternative parquet files found.")
    
    if filings_data is None:
        st.stop()

# Display basic information about the filings data
st.subheader("Basic Information")
st.write(f"Number of 8-K filings: {len(filings_data)}")
st.write(f"Columns: {filings_data.columns.tolist()}")

# Show sample of the data
st.subheader("Data Sample")
st.dataframe(filings_data.head(5))

# Determine date column
date_col = None
for col in ['date', 'acceptedDate', 'filingDate']:
    if col in filings_data.columns:
        date_col = col
        filings_data[date_col] = pd.to_datetime(filings_data[date_col], errors='coerce')
        break

if date_col:
    # Show filing timeline
    st.subheader("Filing Timeline")
    
    # Group by month
    filings_data['month'] = filings_data[date_col].dt.to_period('M')
    monthly_counts = filings_data.groupby('month').size().reset_index(name='count')
    monthly_counts['month'] = monthly_counts['month'].astype(str)
    
    fig = px.line(
        monthly_counts,
        x='month',
        y='count',
        title="8-K Filings by Month",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Filings",
        xaxis=dict(tickangle=45)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Check for relevant columns for item extraction
item_text_cols = []
for col in ['title', 'description', 'text', 'content', 'form_type']:
    if col in filings_data.columns:
        item_text_cols.append(col)

if item_text_cols:
    st.subheader("Content Analysis")
    st.write(f"Columns potentially containing 8-K item information: {', '.join(item_text_cols)}")
    
    # Sample content from these columns
    sample_idx = min(len(filings_data) - 1, 0)  # First filing or none
    if 'title' in filings_data.columns:
        # Try to find a filing with a non-empty title
        non_empty_titles = filings_data.index[filings_data['title'].notna() & (filings_data['title'] != '')].tolist()
        if non_empty_titles:
            sample_idx = non_empty_titles[0]
    
    sample_filing = filings_data.iloc[sample_idx]
    
    for col in item_text_cols:
        if col in sample_filing and pd.notna(sample_filing[col]) and sample_filing[col]:
            st.write(f"**Sample {col}:**")
            st.text(str(sample_filing[col])[:500] + ("..." if len(str(sample_filing[col])) > 500 else ""))
    
    # Custom Item Pattern Analysis
    st.subheader("Item Pattern Analysis")
    st.write("Analyzing for potential 8-K item patterns...")
    
    # Define patterns to look for
    patterns = [
        r"Item\s+(\d+\.\d+)",  # Standard format: "Item X.XX"
        r"ITEM\s+(\d+\.\d+)",  # All caps: "ITEM X.XX"
        r"Item\s+(\d+)",       # Just number: "Item X"
        r"ITEM\s+(\d+)"        # All caps, just number: "ITEM X"
    ]
    
    import re
    
    # Function to search for patterns
    def find_item_patterns(text, patterns):
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, str(text))
            results.extend(matches)
        return list(set(results))  # Remove duplicates
    
    # Try to find patterns in each content column
    pattern_counts = {pattern: 0 for pattern in patterns}
    
    for col in item_text_cols:
        pattern_found = False
        
        for pattern in patterns:
            # Test on a sample of rows
            for idx, row in filings_data.head(50).iterrows():
                if pd.notna(row[col]) and row[col]:
                    matches = re.findall(pattern, str(row[col]))
                    if matches:
                        pattern_counts[pattern] += 1
                        pattern_found = True
        
        if pattern_found:
            st.write(f"Found potential item patterns in the '{col}' column!")
    
    # Show pattern detection results
    successful_patterns = [p for p, c in pattern_counts.items() if c > 0]
    if successful_patterns:
        st.write("**Successful pattern matches:**")
        for pattern in successful_patterns:
            st.code(pattern, language="text")
            st.write(f"Matched {pattern_counts[pattern]} times")
    else:
        st.warning("No standard 8-K item patterns found in the text columns.")
        
    # Display occurrences of specific common items
    common_items = [
        "Item 1.01", "Item 2.01", "Item 2.02", "Item 5.02", "Item 7.01", "Item 8.01", "Item 9.01"
    ]
    
    st.subheader("Common Item Search")
    
    # Combine text from relevant columns
    combined_text_samples = []
    for idx, row in filings_data.head(50).iterrows():
        text = ""
        for col in item_text_cols:
            if col in row and pd.notna(row[col]):
                text += str(row[col]) + " "
        combined_text_samples.append(text)
    
    item_occurrences = {}
    for item in common_items:
        count = sum(1 for text in combined_text_samples if item in text)
        item_occurrences[item] = count
    
    # Create bar chart of occurrences
    if any(count > 0 for count in item_occurrences.values()):
        item_df = pd.DataFrame({
            "Item": list(item_occurrences.keys()),
            "Count": list(item_occurrences.values())
        }).sort_values("Count", ascending=False)
        
        fig = px.bar(
            item_df,
            x="Item",
            y="Count",
            title="Common 8-K Items Found in Samples",
            color="Count",
            color_continuous_scale="Viridis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("None of the common 8-K items were found in the text samples.")

# Column content exploration
st.subheader("Column Content Exploration")

# Create tabs for each important column group
tab1, tab2, tab3 = st.tabs(["Basic Info", "Content", "Metadata"])

with tab1:
    basic_cols = ['ticker', 'symbol', 'cik', 'date', 'filingDate', 'acceptedDate', 'form_type']
    available_cols = [col for col in basic_cols if col in filings_data.columns]
    
    if available_cols:
        st.dataframe(filings_data[available_cols].head(10))
    else:
        st.write("No basic information columns found.")

with tab2:
    content_cols = ['title', 'description', 'text', 'content', 'item']
    available_cols = [col for col in content_cols if col in filings_data.columns]
    
    if available_cols:
        # Show non-null counts for content columns
        null_counts = filings_data[available_cols].isnull().sum()
        non_null_counts = len(filings_data) - null_counts
        
        null_df = pd.DataFrame({
            'Column': null_counts.index,
            'Non-null Count': non_null_counts.values,
            'Non-null Percentage': (non_null_counts / len(filings_data) * 100).values
        })
        
        st.write("Content column completeness:")
        st.dataframe(null_df)
        
        # Sample content for each column
        st.write("Sample content:")
        for col in available_cols:
            # Get a sample with non-null value
            sample = filings_data[filings_data[col].notna()].sample(1) if not filings_data[filings_data[col].notna()].empty else None
            
            if sample is not None:
                with st.expander(f"{col} sample", expanded=False):
                    content = sample[col].values[0]
                    if isinstance(content, str) and len(content) > 1000:
                        # Truncate long text
                        st.text(content[:1000] + "...")
                    else:
                        st.text(str(content))
    else:
        st.write("No content columns found.")

with tab3:
    meta_cols = ['finalLink', 'link', 'url', 'type', 'form', 'id', 'filerId']
    available_cols = [col for col in meta_cols if col in filings_data.columns]
    
    if available_cols:
        st.dataframe(filings_data[available_cols].head(10))
    else:
        st.write("No metadata columns found.")

# Analyze columns to help with event processor integration
st.header("Recommendations for Event Processor")

recommendations = []

# Check for date column
if date_col:
    recommendations.append(f"Use '{date_col}' as the date column in your event processor.")
else:
    recommendations.append("⚠️ No standard date column found. Your event processor needs custom date handling.")

# Check for company identifier
ticker_col = None
for col in ['ticker', 'symbol']:
    if col in filings_data.columns:
        ticker_col = col
        break

if ticker_col:
    recommendations.append(f"Use '{ticker_col}' as the company identifier column.")
else:
    recommendations.append("⚠️ No standard ticker column found. Your event processor needs custom company identification.")

# Check for content columns
if item_text_cols:
    text_cols_str = ", ".join([f"'{col}'" for col in item_text_cols])
    recommendations.append(f"Search these columns for 8-K items: {text_cols_str}")
else:
    recommendations.append("⚠️ No content columns found for item extraction.")

# Add pattern recommendations
if 'successful_patterns' in locals() and successful_patterns:
    pattern_str = " | ".join([f"'{p}'" for p in successful_patterns])
    recommendations.append(f"Use regex pattern(s): {pattern_str}")
else:
    recommendations.append("⚠️ No successful item patterns detected. Try a custom pattern.")

# Display recommendations
for i, rec in enumerate(recommendations):
    st.write(f"{i+1}. {rec}")

# Generate code snippet for event processor
if item_text_cols and ('successful_patterns' in locals() and successful_patterns):
    st.subheader("Suggested Code Snippet for Item Extraction")
    
    code = f"""
def extract_8k_items(filing_text):
    \"\"\"
    Extract 8-K item numbers from filing text.
    
    Args:
        filing_text: Text content of the 8-K filing
        
    Returns:
        List of item numbers found in the filing
    \"\"\"
    # Return empty list if text is None or not a string
    if not isinstance(filing_text, str):
        return []
    
    # Pattern to match Item X.XX with variations
    item_patterns = [
        {", ".join([f'r"{p}"' for p in successful_patterns])}
    ]
    
    # Find all matches from all patterns
    all_matches = []
    for pattern in item_patterns:
        matches = re.findall(pattern, filing_text)
        all_matches.extend(matches)
    
    # Format as Item X.XX and remove duplicates
    formatted_items = []
    for item in all_matches:
        # Handle items that might just be numbers without decimals
        if '.' not in item:
            item = item + '.01'  # Add default subsection
        formatted_items.append(f"Item {{item}}")
    
    unique_items = list(set(formatted_items))
    
    return unique_items
"""
    
    st.code(code, language='python')

# Footer
st.markdown("---")
st.markdown("""
Use this analysis to improve your event processor implementation for the Dynamic-KG project.
The key is ensuring your item extraction logic matches the actual data format in your files.
""")