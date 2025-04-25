"""
Advanced event analytics for 8-K filings and market data.

This module provides functions for analyzing relationships between material events
and financial/economic metrics, including correlation analysis, signal propagation,
and predictive power assessment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import networkx as nx
import json
import sys
from pathlib import Path
import plotly.graph_objects as go
from pyvis.network import Network
import random
import traceback

# Add the project root directory to the Python path for importing config
root_dir = str(Path(__file__).parents[2].absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the config
from config import Config

# Import the basic event processor
import event_processor as ep

# Set up logger
logger = Config.get_logger(__name__)

# Define the key metrics categories for analysis
KEY_METRICS = {
    "Stock Price": [
        "1-day price change",
        "5-day price change", 
        "30-day price change",
        "30-day volatility",
        "trading volume change"
    ],
    "Company Financial": [
        "revenue growth",
        "net income growth",
        "debt-to-equity ratio",
        "current ratio",
        "gross margin",
        "operating margin"
    ],
    "Market": [
        "s&p 500 change",
        "industry sector change",
        "10y treasury yield change",
        "yield curve change"
    ],
    "Economic": [
        "gdp growth rate",
        "inflation rate change",
        "unemployment rate change",
        "consumer sentiment change",
        "industrial production change",
        "housing starts change"
    ]
}

def analyze_event_metric_relationships(
    events_df: pd.DataFrame,
    market_df: Optional[pd.DataFrame] = None,
    economic_df: Optional[pd.DataFrame] = None,
    company_df: Optional[pd.DataFrame] = None,
    financial_df: Optional[pd.DataFrame] = None,
    window_days: int = 30
) -> Dict[str, Any]:
    """
    Analyze relationships between events and various metrics.
    
    Args:
        events_df: DataFrame containing event data
        market_df: DataFrame containing market data (optional)
        economic_df: DataFrame containing economic indicators (optional)
        company_df: DataFrame containing company data (optional)
        financial_df: DataFrame containing financial data (optional)
        window_days: Number of days after an event to analyze
        
    Returns:
        Dictionary containing relationship analysis results
    """
    if events_df.empty:
        logger.warning("No events provided for analysis")
        return {}
    
    try:
        # Find date column and ticker column
        date_col = None
        for col in ['date', 'acceptedDate', 'filingDate']:
            if col in events_df.columns:
                date_col = col
                events_df[col] = pd.to_datetime(events_df[col], errors='coerce')
                break
        
        ticker_col = None
        for col in ['ticker', 'symbol']:
            if col in events_df.columns:
                ticker_col = col
                break
        
        if date_col is None or ticker_col is None:
            logger.error("Required date or ticker column not found in events data")
            return {}
        
        # Check if the DataFrame already has metrics
        metric_prefix = "metric_"
        has_metrics = any(col.startswith(metric_prefix) for col in events_df.columns)
        
        # If metrics are not present, calculate them
        if not has_metrics:
            logger.info("Calculating metrics for events...")
            events_df = ep.prepare_event_data_for_analysis(events_df, window_days=window_days)
            
            # Check if metrics were successfully calculated
            has_metrics = any(col.startswith(metric_prefix) for col in events_df.columns)
            
            if not has_metrics:
                logger.warning("Failed to calculate metrics for events")
                return {}
        
        # Get unique event types from business categories
        event_types = set()
        if 'business_categories' in events_df.columns:
            for cats in events_df['business_categories']:
                if isinstance(cats, list):
                    event_types.update(cats)
        
        event_types = sorted(list(event_types))
        
        if not event_types:
            logger.warning("No event types found in data")
            return {}
        
        # Get metric columns
        metric_columns = [col for col in events_df.columns if col.startswith(metric_prefix)]
        metric_names = [col[len(metric_prefix):] for col in metric_columns]
        
        if not metric_columns:
            logger.warning("No metric columns found in data")
            return {}
        
        # Initialize results structure
        results = {
            'event_types': event_types,
            'metric_names': metric_names,
            'correlation_matrix': [],
            'avg_changes': {},
            'lag_analysis': []
        }
        
        # Initialize correlation matrix
        correlation_matrix = np.zeros((len(event_types), len(metric_names)))
        
        # Calculate correlations and average changes
        for i, event_type in enumerate(event_types):
            # Filter events for this type
            type_events = events_df[events_df['business_categories'].apply(
                lambda x: event_type in x if isinstance(x, list) else False
            )]
            
            if len(type_events) < 3:  # Need at least a few events for meaningful analysis
                continue
                
            # Calculate average changes for this event type
            avg_changes = {}
            for j, metric in enumerate(metric_names):
                metric_col = f"{metric_prefix}{metric}"
                if metric_col in type_events.columns:
                    metric_values = type_events[metric_col].dropna()
                    
                    if len(metric_values) >= 3:
                        # Calculate average change
                        avg_change = metric_values.mean()
                        avg_changes[metric] = avg_change
                        
                        # Calculate correlation strength
                        # For binary event presence, we use the mean as a simple correlation measure
                        correlation_matrix[i, j] = avg_change
            
            results['avg_changes'][event_type] = avg_changes
        
        # Normalize correlation matrix to [-1, 1] range
        max_abs_corr = np.max(np.abs(correlation_matrix)) if correlation_matrix.size > 0 else 1.0
        if max_abs_corr > 0:
            correlation_matrix = correlation_matrix / max_abs_corr
        
        results['correlation_matrix'] = correlation_matrix.tolist()
        
        # Calculate lag analysis (timing of impacts)
        lag_analysis = []
        
        # For each event type and metric, analyze timing of impacts
        for event_type in event_types:
            # Filter events for this type
            type_events = events_df[events_df['business_categories'].apply(
                lambda x: event_type in x if isinstance(x, list) else False
            )]
            
            if len(type_events) < 5:  # Need several events for lag analysis
                continue
            
            for metric in metric_names:
                # Check if we have 1-day, 5-day, and 30-day versions of this metric
                # This is primarily for price metrics
                day_metrics = []
                
                for day in [1, 5, 30]:
                    day_metric = f"{metric_prefix}{day}-day {metric}"
                    if day_metric in type_events.columns:
                        day_metrics.append((day, day_metric))
                
                if day_metrics:
                    # Compare impacts at different time lags
                    changes = []
                    days = []
                    
                    for day, day_metric in day_metrics:
                        values = type_events[day_metric].dropna()
                        if len(values) >= 5:
                            changes.append(values.mean())
                            days.append(day)
                    
                    if changes:
                        # Find the peak impact day
                        if abs(max(changes, key=abs)) > 0:
                            peak_day = days[changes.index(max(changes, key=abs))]
                            avg_change = max(changes, key=abs)
                            
                            lag_analysis.append({
                                'event_type': event_type,
                                'metric': metric,
                                'avg_days': peak_day,
                                'correlation': avg_change / max_abs_corr if max_abs_corr > 0 else 0,
                                'occurrence_count': len(type_events)
                            })
                else:
                    # For metrics without day variations, use window_days as the lag
                    metric_col = f"{metric_prefix}{metric}"
                    if metric_col in type_events.columns:
                        values = type_events[metric_col].dropna()
                        if len(values) >= 5:
                            avg_change = values.mean()
                            
                            lag_analysis.append({
                                'event_type': event_type,
                                'metric': metric,
                                'avg_days': window_days // 2,  # Use half the window as a reasonable estimate
                                'correlation': avg_change / max_abs_corr if max_abs_corr > 0 else 0,
                                'occurrence_count': len(type_events)
                            })
        
        results['lag_analysis'] = lag_analysis
        
        return results
    except Exception as e:
        logger.error(f"Error analyzing event-metric relationships: {e}", exc_info=True)
        return {}

def analyze_signal_propagation(
    events_df: pd.DataFrame,
    market_df: Optional[pd.DataFrame] = None,
    company_df: Optional[pd.DataFrame] = None,
    window_days: int = 30
) -> Dict[str, Any]:
    """
    Analyze how signals from events propagate across companies.
    
    Args:
        events_df: DataFrame containing event data
        market_df: DataFrame containing market data (optional)
        company_df: DataFrame containing company data (optional)
        window_days: Number of days to track propagation
        
    Returns:
        Dictionary containing signal propagation analysis
    """
    if events_df.empty:
        logger.warning("No events provided for analysis")
        return {}
    
    try:
        # Find date column and ticker column
        date_col = None
        for col in ['date', 'acceptedDate', 'filingDate']:
            if col in events_df.columns:
                date_col = col
                events_df[col] = pd.to_datetime(events_df[col], errors='coerce')
                break
        
        ticker_col = None
        for col in ['ticker', 'symbol']:
            if col in events_df.columns:
                ticker_col = col
                break
        
        if date_col is None or ticker_col is None:
            logger.error("Required date or ticker column not found in events data")
            return {}
        
        # Get unique companies
        companies = events_df[ticker_col].unique()
        
        # Build event correlation network
        network = ep.build_event_correlation_network(events_df, min_correlation=0.2, time_window=window_days*3)
        
        # Get unique event types
        event_types = set()
        if 'business_categories' in events_df.columns:
            for cats in events_df['business_categories']:
                if isinstance(cats, list):
                    event_types.update(cats)
        
        event_types = sorted(list(event_types))
        
        # Initialize results
        results = {
            'timeline': [],
            'network_html': None,
            'industry_impact': []
        }
        
        # Process Timeline: For each company and event type, analyze how it affects other companies
        for event_type in event_types:
            # Filter events of this type
            type_events = events_df[events_df['business_categories'].apply(
                lambda x: event_type in x if isinstance(x, list) else False
            )]
            
            if len(type_events) < 3:  # Need at least a few events
                continue
            
            # For each event, find price changes in other companies
            for _, event in type_events.iterrows():
                source_company = event[ticker_col]
                event_date = event[date_col]
                
                # Skip events with missing dates
                if pd.isna(event_date):
                    continue
                
                # Find other companies that might be affected
                affected_companies = [c for c in companies if c != source_company]
                
                # Use network to prioritize connected companies
                connected_companies = []
                for edge in network.get('edges', []):
                    if edge['source'] == source_company:
                        connected_companies.append(edge['target'])
                    elif edge['target'] == source_company:
                        connected_companies.append(edge['source'])
                
                # Prioritize connected companies
                if connected_companies:
                    affected_companies = connected_companies + [c for c in affected_companies if c not in connected_companies]
                
                # Limit to a reasonable number
                affected_companies = affected_companies[:min(10, len(affected_companies))]
                
                # Analyze impact on affected companies
                for target_company in affected_companies:
                    # Calculate impact using event_processor
                    impact = ep.get_event_impact(target_company, event_date, window_days)
                    
                    if impact['status'] != 'success':
                        continue
                    
                    # Get price metrics
                    price_metrics = impact.get('price_metrics', {})
                    
                    if not price_metrics:
                        continue
                    
                    # Get price changes at different time lags
                    for lag_metric in ['1-day price change', '5-day price change', '30-day price change']:
                        if lag_metric in price_metrics:
                            impact_strength = price_metrics[lag_metric]
                            days_after = int(lag_metric.split('-')[0])
                            
                            # Estimate significance by change magnitude
                            # In a more sophisticated implementation, we'd compare to normal volatility
                            significance = abs(impact_strength) * 5  # Scale for visualization
                            
                            # Add to timeline
                            results['timeline'].append({
                                'event_type': event_type,
                                'source_company': source_company,
                                'company': target_company,
                                'days_after_event': days_after,
                                'impact_strength': impact_strength,
                                'significance': significance
                            })
        
        # Create network visualization
        if network.get('nodes') and network.get('edges'):
            try:
                net = Network(notebook=False, height="600px", width="100%", directed=True)
                
                # Add nodes
                for node in network['nodes']:
                    # Estimate node size based on event count
                    size = min(30, max(10, node.get('event_count', 0) / 2))
                    
                    # Add node to visualization
                    net.add_node(
                        node['id'],
                        label=node['id'],
                        size=size,
                        title=f"{node['id']}: {node['event_count']} events",
                        group=node.get('categories', ['Other'])[0] if node.get('categories') else 'Other'
                    )
                
                # Add edges
                for edge in network['edges']:
                    # Scale for visualization
                    width = abs(edge['correlation']) * 5
                    
                    # Add edge to visualization
                    net.add_edge(
                        edge['source'],
                        edge['target'],
                        width=width,
                        title=f"Correlation: {edge['correlation']:.2f}"
                    )
                
                # Set physics options for better layout
                net.set_options("""
                var options = {
                    "physics": {
                        "forceAtlas2Based": {
                            "gravitationalConstant": -50,
                            "centralGravity": 0.01,
                            "springLength": 100,
                            "springConstant": 0.08
                        },
                        "solver": "forceAtlas2Based",
                        "stabilization": {
                            "iterations": 100
                        }
                    }
                }
                """)
                
                # Generate HTML
                html = net.generate_html()
                results['network_html'] = html
            except Exception as e:
                logger.error(f"Error generating network visualization: {e}", exc_info=True)
        
        # Calculate industry impact
        # In a real implementation, we'd use company industry data
        # For this POC, we'll simulate industry data
        
        # Create simulated industry mappings if actual data isn't available
        industry_mapping = {}
        industries = ["Technology", "Financial", "Healthcare", "Consumer", "Industrial", "Energy"]
        
        for company in companies:
            # Try to extract industry from event data (if categorized)
            company_events = events_df[events_df[ticker_col] == company]
            
            if 'business_categories' in company_events.columns:
                # Look for events that might indicate industry
                industry = None
                
                for _, event in company_events.iterrows():
                    if isinstance(event.get('business_categories'), list):
                        cats = event['business_categories']
                        if "Financial Condition" in cats:
                            industry = "Financial"
                            break
                        elif "Material Agreements" in cats and "technology" in str(event.get('title', '')).lower():
                            industry = "Technology"
                            break
                
                if industry:
                    industry_mapping[company] = industry
                else:
                    # Assign a random industry if we couldn't determine one
                    industry_mapping[company] = random.choice(industries)
            else:
                # Assign a random industry
                industry_mapping[company] = random.choice(industries)
        
        # Calculate average impact by industry and event type
        industry_impacts = {}
        
        for event_type in event_types:
            # Get all impacts for this event type
            type_timeline = [t for t in results['timeline'] if t['event_type'] == event_type]
            
            for item in type_timeline:
                company = item['company']
                industry = industry_mapping.get(company, "Other")
                impact = item['impact_strength']
                
                key = (event_type, industry)
                if key not in industry_impacts:
                    industry_impacts[key] = []
                
                industry_impacts[key].append(impact)
        
        # Calculate averages
        for (event_type, industry), impacts in industry_impacts.items():
            if impacts:
                results['industry_impact'].append({
                    'event_type': event_type,
                    'industry': industry,
                    'avg_impact': sum(impacts) / len(impacts),
                    'sample_size': len(impacts)
                })
        
        return results
    except Exception as e:
        logger.error(f"Error analyzing signal propagation: {e}", exc_info=True)
        return {}

def analyze_predictive_power(
    events_df: pd.DataFrame,
    market_df: Optional[pd.DataFrame] = None,
    economic_df: Optional[pd.DataFrame] = None,
    company_df: Optional[pd.DataFrame] = None,
    financial_df: Optional[pd.DataFrame] = None,
    target_metric: str = "5-day price change",
    window_days: int = 30
) -> Dict[str, Any]:
    """
    Analyze the predictive power of events for a specific metric.
    
    Args:
        events_df: DataFrame containing event data
        market_df: DataFrame containing market data (optional)
        economic_df: DataFrame containing economic indicators (optional)
        company_df: DataFrame containing company data (optional)
        financial_df: DataFrame containing financial data (optional)
        target_metric: Metric to predict
        window_days: Number of days to analyze
        
    Returns:
        Dictionary containing predictive power analysis
    """
    if events_df.empty:
        logger.warning("No events provided for analysis")
        return {}
    
    try:
        # Find date column and ticker column
        date_col = None
        for col in ['date', 'acceptedDate', 'filingDate']:
            if col in events_df.columns:
                date_col = col
                events_df[col] = pd.to_datetime(events_df[col], errors='coerce')
                break
        
        ticker_col = None
        for col in ['ticker', 'symbol']:
            if col in events_df.columns:
                ticker_col = col
                break
        
        if date_col is None or ticker_col is None:
            logger.error("Required date or ticker column not found in events data")
            return {}
        
        # Check if the DataFrame already has metrics
        metric_prefix = "metric_"
        full_target_metric = f"{metric_prefix}{target_metric}"
        
        # If the target metric is not present, calculate metrics
        if full_target_metric not in events_df.columns:
            logger.info(f"Target metric {target_metric} not found. Calculating metrics...")
            events_df = ep.prepare_event_data_for_analysis(events_df, window_days=window_days)
            
            # Check if metrics were successfully calculated
            if full_target_metric not in events_df.columns:
                logger.warning(f"Failed to calculate target metric {target_metric}")
                return {}
        
        # Get unique event types from business categories
        event_types = set()
        if 'business_categories' in events_df.columns:
            for cats in events_df['business_categories']:
                if isinstance(cats, list):
                    event_types.update(cats)
        
        event_types = sorted(list(event_types))
        
        if not event_types:
            logger.warning("No event types found in data")
            return {}
        
        # Filter to events with target metric
        events_df = events_df.dropna(subset=[full_target_metric])
        
        if events_df.empty:
            logger.warning(f"No events with valid {target_metric} values")
            return {}
        
        # Initialize results
        results = {
            'accuracy_by_event': {},
            'information_value': {},
            'feature_importance': []
        }
        
        # Calculate prediction accuracy by event type
        for event_type in event_types:
            # Filter events for this type
            type_events = events_df[events_df['business_categories'].apply(
                lambda x: event_type in x if isinstance(x, list) else False
            )]
            
            if len(type_events) < 5:  # Need enough samples
                continue
            
            # Extract target values
            target_values = type_events[full_target_metric]
            
            # Calculate predictive accuracy
            # In a more sophisticated implementation, we'd use cross-validation
            # For this POC, we'll use a simpler approach
            
            # Calculate mean value
            mean_value = target_values.mean()
            
            # Calculate accuracy as consistency in direction
            if abs(mean_value) > 0.001:  # Avoid division by zero or tiny values
                # Count values with same sign as mean
                same_direction = sum(1 for v in target_values if (v > 0 and mean_value > 0) or (v < 0 and mean_value < 0))
                accuracy = same_direction / len(target_values)
            else:
                accuracy = 0.5  # Random guess for near-zero mean
            
            results['accuracy_by_event'][event_type] = accuracy
            
            # Calculate information value
            # Information value is a measure of predictive power
            # For this POC, we'll use a simplified approach
            
            # Calculate variance of target metric overall
            overall_variance = events_df[full_target_metric].var()
            
            if overall_variance > 0:
                # Calculate variance within this event type
                group_variance = target_values.var()
                
                # Information value as variance reduction
                info_value = 1 - (group_variance / overall_variance)
                info_value = max(0, min(1, info_value))  # Bound between 0 and 1
                
                results['information_value'][event_type] = info_value
        
        # Calculate feature importance using a machine learning model
        try:
            # Prepare features using one-hot encoding of event types
            X_data = []
            y_data = []
            
            # Group by company and date to avoid duplication
            for ticker in events_df[ticker_col].unique():
                company_events = events_df[events_df[ticker_col] == ticker]
                
                # Group by date (to handle multiple events on same day)
                for date, date_events in company_events.groupby(date_col):
                    # Create feature vector (event types present)
                    features = np.zeros(len(event_types))
                    
                    for _, event in date_events.iterrows():
                        if isinstance(event.get('business_categories'), list):
                            for i, event_type in enumerate(event_types):
                                if event_type in event['business_categories']:
                                    features[i] = 1
                    
                    # Get target value (average if multiple events)
                    target = date_events[full_target_metric].mean()
                    
                    X_data.append(features)
                    y_data.append(target)
            
            if len(X_data) >= 10:  # Need enough samples for meaningful analysis
                X = np.array(X_data)
                y = np.array(y_data)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                # Train a random forest model
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                
                # Get feature importance
                importances = model.feature_importances_
                
                # Create feature importance list
                for i, event_type in enumerate(event_types):
                    results['feature_importance'].append({
                        'feature': event_type,
                        'importance': float(importances[i])
                    })
                
                # Sort by importance
                results['feature_importance'].sort(key=lambda x: x['importance'], reverse=True)
                
                # Calculate model performance
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                results['model_performance'] = {
                    'r2_score': float(r2),
                    'mean_absolute_error': float(mae),
                    'test_size': len(y_test)
                }
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}", exc_info=True)
        
        return results
    except Exception as e:
        logger.error(f"Error analyzing predictive power: {e}", exc_info=True)
        return {}

def calculate_information_value(
    events: pd.DataFrame,
    target_values: pd.Series,
    event_column: str,
    bins: int = 10
) -> Dict[str, float]:
    """
    Calculate the information value of event types for predicting a target.
    
    Args:
        events: DataFrame containing events
        target_values: Series containing target values
        event_column: Column containing event types
        bins: Number of bins for discretizing target
        
    Returns:
        Dictionary mapping event types to information value
    """
    try:
        if events.empty or target_values.empty:
            return {}
        
        # Check if event_column exists
        if event_column not in events.columns:
            logger.error(f"Event column {event_column} not found in events DataFrame")
            return {}
        
        # Discretize target values
        target_bins = pd.qcut(target_values, bins, duplicates='drop')
        
        # Get unique event types
        event_types = events[event_column].unique()
        
        # Calculate information value for each event type
        info_values = {}
        
        for event_type in event_types:
            # Create binary indicator for this event type
            event_indicator = (events[event_column] == event_type)
            
            # Calculate information value
            # (This is a simplified approach - real implementation would use weight of evidence)
            
            info_value = 0
            for bin_val in target_bins.cat.categories:
                # Get counts with and without event
                with_event = ((target_bins == bin_val) & event_indicator).sum()
                without_event = ((target_bins == bin_val) & ~event_indicator).sum()
                
                # Get total counts
                total_with_event = event_indicator.sum()
                total_without_event = (~event_indicator).sum()
                
                # Calculate proportions
                if total_with_event > 0 and total_without_event > 0:
                    prop_with_event = with_event / total_with_event
                    prop_without_event = without_event / total_without_event
                    
                    # Avoid division by zero
                    if prop_with_event > 0 and prop_without_event > 0:
                        # Weight of evidence
                        woe = np.log(prop_with_event / prop_without_event)
                        
                        # Information value contribution
                        iv_contribution = (prop_with_event - prop_without_event) * woe
                        
                        info_value += iv_contribution
            
            info_values[event_type] = abs(info_value)
        
        return info_values
    except Exception as e:
        logger.error(f"Error calculating information value: {e}", exc_info=True)
        return {}

def create_propagation_network(
    propagation_data: List[Dict[str, Any]],
    min_impact: float = 0.1
) -> str:
    """
    Create a network visualization of signal propagation.
    
    Args:
        propagation_data: List of propagation data points
        min_impact: Minimum impact to include in the network
        
    Returns:
        HTML string containing the network visualization
    """
    try:
        if not propagation_data:
            return ""
        
        # Create network
        net = Network(notebook=False, height="600px", width="100%", directed=True)
        
        # Track nodes and edges
        nodes = set()
        edges = {}
        
        # Process propagation data
        for item in propagation_data:
            source = item.get('source_company')
            target = item.get('company')
            impact = item.get('impact_strength', 0)
            event_type = item.get('event_type', 'Unknown')
            
            if not source or not target or abs(impact) < min_impact:
                continue
            
            # Add nodes
            nodes.add(source)
            nodes.add(target)
            
            # Add or update edge
            edge_key = (source, target, event_type)
            if edge_key in edges:
                # Update existing edge
                edges[edge_key]['impact'] += impact
                edges[edge_key]['count'] += 1
            else:
                # Add new edge
                edges[edge_key] = {
                    'impact': impact,
                    'count': 1,
                    'event_type': event_type
                }
        
        # Add nodes to network
        for node in nodes:
            net.add_node(node, label=node, title=node)
        
        # Add edges to network
        for (source, target, event_type), edge_data in edges.items():
            # Calculate average impact
            avg_impact = edge_data['impact'] / edge_data['count']
            
            # Set edge properties
            width = min(10, max(1, abs(avg_impact) * 20))  # Scale impact to reasonable width
            color = 'green' if avg_impact > 0 else 'red'
            
            # Add edge
            net.add_edge(
                source, 
                target, 
                width=width, 
                color=color, 
                title=f"{event_type}: {avg_impact:.2%} (n={edge_data['count']})"
            )
        
        # Set physics options
        net.set_options("""
        var options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "iterations": 100
                }
            }
        }
        """)
        
        # Generate HTML
        html = net.generate_html()
        return html
    except Exception as e:
        logger.error(f"Error creating propagation network: {e}", exc_info=True)
        return ""

def run_event_analysis(
    output_dir: Optional[Union[str, Path]] = None,
    window_days: int = 30
) -> Dict[str, Any]:
    """
    Run a complete event analysis and save results.
    
    Args:
        output_dir: Directory to save results (default to Config.OUTPUT_DIR)
        window_days: Number of days to analyze
        
    Returns:
        Dictionary containing all analysis results
    """
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting comprehensive event analysis")
    
    # Load event data
    events_data = ep.load_8k_filings()
    
    if events_data.empty:
        logger.error("No event data found. Please load 8-K filings first.")
        return {}
    
    # Create event timeline
    timeline = ep.create_event_timeline(events_data)
    
    if timeline.empty:
        logger.error("Could not create event timeline")
        return {}
    
    logger.info(f"Loaded {len(timeline)} events for analysis")
    
    # Initialize results
    results = {
        'relationship_analysis': None,
        'propagation_analysis': None,
        'predictive_analysis': {}
    }
    
    # Run relationship analysis
    logger.info("Running event-metric relationship analysis")
    relationship_results = analyze_event_metric_relationships(timeline, window_days=window_days)
    results['relationship_analysis'] = relationship_results
    
    # Save relationship results
    relationship_file = output_path / "event_relationships.json"
    try:
        with open(relationship_file, 'w') as f:
            json.dump(relationship_results, f, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        logger.info(f"Saved relationship analysis to {relationship_file}")
    except Exception as e:
        logger.error(f"Error saving relationship analysis: {e}")
    
    # Run propagation analysis
    logger.info("Running signal propagation analysis")
    propagation_results = analyze_signal_propagation(timeline, window_days=window_days)
    results['propagation_analysis'] = propagation_results
    
    # Save propagation results (excluding HTML content)
    prop_results_save = {k: v for k, v in propagation_results.items() if k != 'network_html'}
    prop_file = output_path / "signal_propagation.json"
    try:
        with open(prop_file, 'w') as f:
            json.dump(prop_results_save, f)
        logger.info(f"Saved propagation analysis to {prop_file}")
        
        # Save network HTML separately
        if 'network_html' in propagation_results and propagation_results['network_html']:
            html_file = output_path / "propagation_network.html"
            with open(html_file, 'w') as f:
                f.write(propagation_results['network_html'])
            logger.info(f"Saved propagation network to {html_file}")
    except Exception as e:
        logger.error(f"Error saving propagation analysis: {e}")
    
    # Run predictive analysis for key metrics
    for category, metrics in KEY_METRICS.items():
        for metric in metrics[:2]:  # Analyze first two metrics in each category
            logger.info(f"Running predictive analysis for {metric}")
            prediction_results = analyze_predictive_power(
                timeline, 
                target_metric=metric,
                window_days=window_days
            )
            
            if prediction_results:
                results['predictive_analysis'][metric] = prediction_results
                
                # Save prediction results
                pred_file = output_path / f"predictive_{metric.replace(' ', '_')}.json"
                try:
                    with open(pred_file, 'w') as f:
                        json.dump(prediction_results, f)
                    logger.info(f"Saved predictive analysis for {metric} to {pred_file}")
                except Exception as e:
                    logger.error(f"Error saving predictive analysis for {metric}: {e}")
    
    logger.info("Event analysis complete")
    return results

if __name__ == "__main__":
    # Initialize configuration
    Config.initialize()
    
    # Run analysis with default parameters
    try:
        results = run_event_analysis(window_days=30)
        
        if results:
            logger.info("Analysis completed successfully")
        else:
            logger.error("Analysis failed or returned no results")
    except Exception as e:
        logger.error(f"Error running event analysis: {e}", exc_info=True)