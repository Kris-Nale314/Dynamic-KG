# Dynamic-KG: Data Strategy

## Overview

This document outlines the data strategy for the Dynamic-KG project, focusing on how we collect, process, and integrate various data sources to create evolving knowledge graphs for loan risk assessment. Our approach combines synthetic data generation with external market data to demonstrate how knowledge graphs can evolve over time and provide context-aware risk insights.

## Core Data Philosophy

Our data strategy is built on three key principles:

1. **Temporal Evolution**: All data is time-stamped and versioned to track how entities and relationships change over time.
2. **Contextual Enrichment**: Individual loan applications are positioned within their broader market and industry context.
3. **Signal Amplification**: By combining multiple weak signals from different sources, we can generate stronger predictive insights.

## Data Sources

### 1. Synthetic Loan Data

We generate synthetic loan applications with realistic properties to serve as the foundation for our knowledge graph. This data includes:

- Company profiles with industry classification
- Owner information with credit scores
- Financial statements and ratios
- Loan applications with specific terms and purposes
- Risk metrics with confidence scoring

This synthetic data is designed to evolve over multiple time periods, demonstrating how the knowledge graph adapts to changing circumstances.

### 2. Public Company Financial Data

We collect financial data from 30 selected public companies that serve as "bellwethers" for broader economic and industry trends. These companies span multiple sectors:

**Technology**: Google, Microsoft, Dell, NVIDIA, TSMC  
**Consumer**: Tesla, Amazon, Walmart, McDonald's, Nike  
**Industrial/Energy**: Shell, Caterpillar, Boeing, General Electric, Union Pacific, Constellation Energy  
**Financial**: JPMorgan Chase, Goldman Sachs, Visa, American Express, Berkshire Hathaway  
**Healthcare**: Johnson & Johnson, UnitedHealth Group, Pfizer  
**Professional Services**: Booz Allen Hamilton, Accenture  
**Real Estate**: Prologis, Simon Property Group  

For each company, we collect:
- Quarterly financial statements (income statement, balance sheet, cash flow)
- Key financial ratios
- Stock price and trading volume
- 8-K filings (material events)

### 3. Market and Economic Data

To provide broader context, we capture:
- Sector performance metrics
- Economic indicators (GDP, inflation, unemployment)
- Interest rates and yield curves
- Industry-specific growth rates
- Market sentiment indicators

### 4. International Economic Context

Selected international indicators to capture global economic impacts:
- Currency exchange rates (USD/CNY, USD/EUR)
- Trade relationship metrics
- Global economic health indicators

### 5. Material Events (8-K Filings)

Special focus is placed on 8-K filings as early indicators of significant changes:
- Management changes
- Material agreements (entry or termination)
- Acquisitions and divestitures
- Restructuring events
- Financial restatements

## Signal Amplification Strategy

The power of our approach comes from combining multiple signals to create more reliable risk indicators:

### Industry Context Signals

By monitoring public companies in the same industry as a loan applicant, we can identify:
- **Industry Growth Trends**: Declining growth in public companies can be an early warning for smaller businesses
- **Margin Pressure**: If major players are seeing margin contraction, smaller companies likely face similar pressures
- **Capital Investment Patterns**: Changes in CapEx among industry leaders often signal broader industry shifts

### Material Event Propagation

8-K filings from public companies can provide early warning signals:
- **Leadership Changes**: Departures of key executives in major companies often precede industry-wide challenges
- **Contract Terminations**: Cancellation of major contracts can indicate shifting market dynamics
- **Restructuring Events**: Cost-cutting measures in large firms may indicate industry-wide financial stress

### Economic Context Integration

Broader economic indicators provide essential context:
- **Interest Rate Sensitivity**: Companies in debt-heavy industries become higher risk as rates rise
- **Supply Chain Vulnerability**: International economic tensions can create cascading effects through supply chains
- **Consumer Sentiment Impact**: Changes in consumer behavior affect different industries with varying time lags

### Financial Ratio Correlations

By analyzing patterns between public company financial ratios and loan performance:
- **Leading Indicators**: Some financial metrics in public companies may precede similar changes in private firms
- **Sector-Specific Thresholds**: Different industries have different tolerance levels for metrics like debt-to-equity
- **Temporal Patterns**: The time lag between public and private company effects varies by industry and metric

## Data Collection Strategy

### Collection Frequency

- **Core Financial Data**: Quarterly updates aligned with reporting seasons
- **8-K Filings**: Near real-time collection as filings occur
- **Market Data**: Daily closing prices and volumes
- **Economic Indicators**: As released (typically monthly or quarterly)

### Historical Range

- Initial collection spans 3 years of historical data
- Synthetic loan applications are "backdated" to allow for simulation of knowledge graph evolution
- Multiple timeframes (24 months ago, 18 months ago) serve as starting points to test predictive capabilities

### Implementation Approach

We use a modular data collection system with components for:
1. Core financial data collection
2. 8-K processing and categorization
3. Market data aggregation
4. Economic indicator integration
5. Relationship mapping

All data is stored in the dataStore structure with clear separation between:
- Raw collected data
- Processed knowledge graph entities
- Temporal snapshots for historical analysis

## Signal Integration and Knowledge Graph Evolution

The knowledge graph integrates these diverse signals by:

1. **Representing entities and relationships** with temporal attributes
2. **Calculating confidence scores** for each piece of information
3. **Propagating changes** through the graph when new information arrives
4. **Detecting patterns** across different entity types
5. **Identifying anomalies** that may indicate changing risk profiles

As new data arrives, the knowledge graph evolves through:
- Addition of new nodes and relationships
- Modification of confidence scores for existing information
- Decay of older information's relevance
- Formation of new inferred relationships based on pattern recognition

## Experimental Approach

This data strategy supports our experimental approach by:
1. Using historical data to simulate how the knowledge graph would have evolved over time
2. Testing predictive capabilities by "backdating" loan applications
3. Comparing signal strength in different combinations of data sources
4. Measuring the impact of adding contextual information on risk assessment accuracy

The modular nature of our data collection allows for incremental enhancement and focused experimentation on specific signal types.

## Future Enhancements

Potential extensions to this data strategy include:
- **News sentiment analysis**: Incorporating news about companies and industries
- **Supply chain mapping**: More detailed relationship mapping between entities
- **Regulatory event tracking**: Monitoring changes in regulations affecting industries
- **Alternative data sources**: Adding non-traditional data like satellite imagery, social media trends, etc.

These enhancements would be prioritized based on their demonstrated value in improving risk assessment accuracy.