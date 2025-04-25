"""
Synthetic loan application generator for Dynamic-KG.

This module generates realistic synthetic loan applications with temporal
attributes, relationship information, and confidence scoring - designed to
demonstrate the core innovations of the Dynamic-KG project.

Place this file in the 'data/synthetic' folder and run directly.
"""

import os
import sys
import json
import random
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker
from tqdm import tqdm

# Add the project root directory to the Python path for importing config
root_dir = str(Path(__file__).parents[2].absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the config from the root
from config import Config

# Initialize generators
fake = Faker()
Faker.seed(Config.SYNTHETIC_COMPANY_COUNT)  # For reproducibility
random.seed(Config.SYNTHETIC_COMPANY_COUNT)
np.random.seed(Config.SYNTHETIC_COMPANY_COUNT)

# Set up logger
logger = Config.get_logger(__name__)

def assign_entity_id(entity_type, identifier):
    """
    Create a consistent entity ID for the knowledge graph.
    
    Args:
        entity_type: Type of entity (company, owner, loan, etc.)
        identifier: Unique identifier for this entity
        
    Returns:
        String ID formatted for the knowledge graph
    """
    return f"{entity_type}_{identifier}".replace(" ", "_").lower()

def generate_temporal_attributes(base_date=None, future=False):
    """
    Generate temporal attributes for a knowledge graph entity.
    
    Args:
        base_date: Base date to use (defaults to today)
        future: Whether to generate future dates (for end dates)
        
    Returns:
        Dictionary with temporal attributes
    """
    if base_date is None:
        base_date = datetime.now()
    
    # Generate valid_from date (slightly in the past)
    if not future:
        days_offset = random.randint(1, 30)
        valid_from = base_date - timedelta(days=days_offset)
    else:
        valid_from = base_date
    
    # Either generate valid_to date or set to None (still valid)
    if random.random() < 0.2:  # 20% chance of having an end date
        days_offset = random.randint(30, 180)
        valid_to = valid_from + timedelta(days=days_offset)
    else:
        valid_to = None
    
    # Generate collection timestamp (when the data was "collected")
    collection_timestamp = valid_from + timedelta(minutes=random.randint(10, 120))
    
    return {
        "valid_from": valid_from.isoformat(),
        "valid_to": valid_to.isoformat() if valid_to else None,
        "collection_timestamp": collection_timestamp.isoformat()
    }

def generate_confidence_attributes(source_type="synthetic"):
    """
    Generate confidence scoring attributes for a knowledge graph entity.
    
    Args:
        source_type: Type of data source
        
    Returns:
        Dictionary with confidence attributes
    """
    # Base confidence from source reliability
    base_confidence = 0.8  # Default for synthetic data
    
    # Add some randomness
    extraction_confidence = round(base_confidence * random.uniform(0.9, 1.0), 3)
    
    # Generate random corroboration level
    corroboration_sources = random.randint(1, 3)
    
    return {
        "source": source_type,
        "confidence": extraction_confidence,
        "corroboration_level": corroboration_sources,
        "evidence_id": f"SYN-{int(datetime.now().timestamp())}-{random.randint(1000, 9999)}"
    }

def generate_company_profile(industry_sector=None, time_period=0):
    """
    Generate a synthetic company profile with temporal attributes.
    
    Args:
        industry_sector: Optional industry to use
        time_period: Time period offset (0 for current, positive for future periods)
        
    Returns:
        Dictionary containing company profile data
    """
    # Set realistic industry sectors
    industries = [
        "Manufacturing", "Construction", "Retail", "Wholesale", 
        "Professional Services", "Healthcare", "Technology", 
        "Food Service", "Transportation", "Real Estate"
    ]
    
    # Generate base date for temporal attributes
    base_date = datetime.now() + timedelta(days=90 * time_period)
    
    # Use provided industry or select random
    industry = industry_sector if industry_sector else random.choice(industries)
    
    # Set realistic sizes based on industry
    if industry in ["Manufacturing", "Construction", "Healthcare"]:
        employee_range = (15, 200)
        revenue_range = (1000000, 20000000)
        years_in_business_range = (3, 30)
    elif industry in ["Retail", "Food Service", "Technology"]:
        employee_range = (5, 100)
        revenue_range = (500000, 10000000)
        years_in_business_range = (2, 15)
    else:
        employee_range = (3, 50)
        revenue_range = (250000, 5000000)
        years_in_business_range = (1, 20)
    
    # Create company profile with realistic correlations
    company_name = fake.company()
    company_id = assign_entity_id("company", company_name)
    employees = random.randint(*employee_range)
    
    # Correlate revenue with employee count
    base_revenue = random.randint(*revenue_range)
    revenue_modifier = 0.8 + (employees / employee_range[1]) * 0.4
    annual_revenue = int(base_revenue * revenue_modifier)
    
    # Years in business - older companies tend to have more employees
    years_in_business_base = random.randint(*years_in_business_range)
    years_in_business_modifier = 0.7 + (employees / employee_range[1]) * 0.6
    years_in_business = max(1, int(years_in_business_base * years_in_business_modifier))
    
    business_start_date = base_date - timedelta(days=365 * years_in_business)
    
    # Create the basic profile
    profile = {
        "entity_id": company_id,
        "entity_type": "Company",
        "company_name": company_name,
        "industry": industry,
        "industry_id": assign_entity_id("industry", industry),
        "employees": employees,
        "annual_revenue": annual_revenue,
        "years_in_business": years_in_business,
        "business_start_date": business_start_date.strftime("%Y-%m-%d"),
        "business_address": fake.address(),
        "business_city": fake.city(),
        "business_state": fake.state(),
        "business_zip": fake.zipcode(),
        "business_type": random.choice(["LLC", "Corporation", "Partnership", "Sole Proprietorship"]),
    }
    
    # Add temporal attributes
    profile.update(generate_temporal_attributes(base_date))
    
    # Add confidence attributes
    profile.update(generate_confidence_attributes())
    
    return profile

def generate_owner_profile(company_profile, time_period=0):
    """
    Generate owner profile with realistic correlations to company.
    
    Args:
        company_profile: The company profile to correlate with
        time_period: Time period offset (0 for current, positive for future periods)
        
    Returns:
        Dictionary containing owner profile data
    """
    # Generate base date for temporal attributes
    base_date = datetime.now() + timedelta(days=90 * time_period)
    
    # More experienced owners for larger companies
    experience_years = min(45, max(company_profile["years_in_business"] + random.randint(-5, 10), 2))
    
    # Realistic age based on experience (assume career starts ~22 years old)
    min_age = experience_years + 22
    age = random.randint(min_age, min(min_age + 20, 75))
    
    # Credit score correlated with business success
    base_credit_score = 650
    business_success_modifier = (company_profile["annual_revenue"] / 10000000) * 50
    experience_modifier = experience_years * 1.5
    
    credit_score = int(min(850, max(580, base_credit_score + business_success_modifier + experience_modifier + random.randint(-30, 30))))
    
    # Create owner name and ID
    owner_name = fake.name()
    owner_id = assign_entity_id("owner", owner_name)
    
    # Create the basic profile
    profile = {
        "entity_id": owner_id,
        "entity_type": "Owner",
        "owner_name": owner_name,
        "owner_title": random.choice(["CEO", "President", "Owner", "Managing Partner"]),
        "owner_age": age,
        "owner_experience_years": experience_years,
        "owner_credit_score": credit_score,
        "owner_email": fake.email(),
        "owner_phone": fake.phone_number(),
        "ownership_percentage": random.randint(51, 100),
        "owns_company_id": company_profile["entity_id"],
        "company_name": company_profile["company_name"]
    }
    
    # Add temporal attributes
    profile.update(generate_temporal_attributes(base_date))
    
    # Add confidence attributes
    profile.update(generate_confidence_attributes())
    
    return profile

def generate_financial_data(company_profile, time_period=0):
    """
    Generate realistic financial data based on company profile.
    
    Args:
        company_profile: The company profile to correlate with
        time_period: Time period offset (0 for current, positive for future periods)
        
    Returns:
        Dictionary containing financial data
    """
    # Generate base date for temporal attributes
    base_date = datetime.now() + timedelta(days=90 * time_period)
    
    annual_revenue = company_profile["annual_revenue"]
    industry = company_profile["industry"]
    years_in_business = company_profile["years_in_business"]
    
    # Set industry-specific margin ranges
    if industry in ["Manufacturing", "Construction"]:
        gross_margin_range = (0.15, 0.35)
        net_margin_range = (0.03, 0.12)
    elif industry in ["Retail", "Wholesale"]:
        gross_margin_range = (0.20, 0.45)
        net_margin_range = (0.02, 0.08)
    elif industry in ["Technology", "Professional Services"]:
        gross_margin_range = (0.40, 0.75)
        net_margin_range = (0.08, 0.25)
    else:
        gross_margin_range = (0.25, 0.50)
        net_margin_range = (0.04, 0.15)
    
    # Adjust margins based on years in business (more established = more efficient)
    maturity_factor = min(1.0, years_in_business / 10) * 0.2
    gross_margin = random.uniform(gross_margin_range[0], gross_margin_range[1]) + maturity_factor
    net_margin = random.uniform(net_margin_range[0], net_margin_range[1]) + (maturity_factor / 2)
    
    # Calculate financial metrics
    cogs = annual_revenue * (1 - gross_margin)
    gross_profit = annual_revenue - cogs
    operating_expenses = gross_profit - (annual_revenue * net_margin)
    net_income = annual_revenue * net_margin
    
    # Balance sheet items
    cash_on_hand = annual_revenue * random.uniform(0.05, 0.15)
    accounts_receivable = annual_revenue * random.uniform(0.08, 0.25)
    inventory = cogs * random.uniform(0.1, 0.3)
    
    fixed_assets = annual_revenue * random.uniform(0.2, 0.5)
    depreciation = fixed_assets * random.uniform(0.1, 0.3)
    net_fixed_assets = fixed_assets - depreciation
    
    total_assets = cash_on_hand + accounts_receivable + inventory + net_fixed_assets
    
    # Liabilities
    accounts_payable = cogs * random.uniform(0.05, 0.2)
    existing_debt = total_assets * random.uniform(0.1, 0.4)
    accrued_expenses = operating_expenses * random.uniform(0.05, 0.15)
    
    total_liabilities = accounts_payable + existing_debt + accrued_expenses
    equity = total_assets - total_liabilities
    
    # Calculate key ratios
    current_ratio = (cash_on_hand + accounts_receivable + inventory) / (accounts_payable + accrued_expenses) if (accounts_payable + accrued_expenses) > 0 else 2.0
    debt_to_equity = total_liabilities / equity if equity > 0 else 3.0
    debt_service_coverage_ratio = net_income / (existing_debt * 0.1) if existing_debt > 0 else 1.5  # Assuming ~10% debt service
    
    # Create financial data with entity ID
    financial_id = f"{company_profile['entity_id']}_financials_t{time_period}"
    
    # Create the financial data dict
    financial_data = {
        "entity_id": financial_id,
        "entity_type": "FinancialData",
        "company_id": company_profile["entity_id"],
        "company_name": company_profile["company_name"],
        "time_period": time_period,
        "annual_revenue": annual_revenue,
        "cogs": cogs,
        "gross_profit": gross_profit,
        "operating_expenses": operating_expenses,
        "net_income": net_income,
        "cash_on_hand": cash_on_hand,
        "accounts_receivable": accounts_receivable,
        "inventory": inventory,
        "fixed_assets": fixed_assets,
        "total_assets": total_assets,
        "accounts_payable": accounts_payable,
        "existing_debt": existing_debt,
        "total_liabilities": total_liabilities,
        "equity": equity,
        "current_ratio": current_ratio,
        "debt_to_equity": debt_to_equity,
        "debt_service_coverage_ratio": debt_service_coverage_ratio,
        "gross_margin": gross_margin,
        "net_margin": net_margin,
    }
    
    # Add temporal attributes
    financial_data.update(generate_temporal_attributes(base_date))
    
    # Add confidence attributes
    financial_data.update(generate_confidence_attributes())
    
    return financial_data

def generate_market_conditions(time_period=0):
    """
    Generate synthetic market conditions for a specific time period.
    
    Args:
        time_period: Time period offset (0 for current, positive for future periods)
        
    Returns:
        Dictionary containing market condition data
    """
    # Generate base date for temporal attributes
    base_date = datetime.now() + timedelta(days=90 * time_period)
    
    # Generate market ID
    market_id = f"market_conditions_t{time_period}"
    
    # Base metrics
    base_interest_rate = 0.04  # 4% base
    base_unemployment = 0.05   # 5% base
    base_gdp_growth = 0.025    # 2.5% base
    base_inflation = 0.03      # 3% base
    
    # Add time-based trends and randomness
    # Simulate market cycles over time periods
    cycle_position = (time_period % 8) / 8  # Position in an 8-quarter cycle
    cycle_factor = np.sin(cycle_position * 2 * np.pi)
    
    interest_rate = max(0.02, base_interest_rate + (0.02 * cycle_factor) + random.uniform(-0.005, 0.005))
    unemployment = max(0.03, base_unemployment + (0.02 * cycle_factor) + random.uniform(-0.01, 0.01))
    gdp_growth = base_gdp_growth - (0.03 * cycle_factor) + random.uniform(-0.005, 0.005)
    inflation = max(0.01, base_inflation + (0.015 * cycle_factor) + random.uniform(-0.005, 0.005))
    
    # Industry-specific conditions
    industry_conditions = {}
    industries = [
        "Manufacturing", "Construction", "Retail", "Wholesale", 
        "Professional Services", "Healthcare", "Technology", 
        "Food Service", "Transportation", "Real Estate"
    ]
    
    for industry in industries:
        # Base industry growth
        if industry in ["Technology", "Healthcare"]:
            base_growth = 0.05  # 5% base growth for strong industries
        elif industry in ["Retail", "Food Service"]:
            base_growth = 0.03  # 3% base growth for moderate industries
        else:
            base_growth = 0.02  # 2% base growth for other industries
        
        # Add cyclical and random factors
        if industry in ["Construction", "Real Estate"]:
            # More sensitive to interest rates
            industry_growth = base_growth - (0.1 * interest_rate) + (0.03 * cycle_factor)
        elif industry in ["Retail", "Food Service"]:
            # More sensitive to unemployment
            industry_growth = base_growth - (0.2 * unemployment) + (0.02 * cycle_factor)
        elif industry == "Technology":
            # Less sensitive to economic cycles
            industry_growth = base_growth + random.uniform(-0.01, 0.03)
        else:
            # General economic sensitivity
            industry_growth = base_growth + (0.5 * gdp_growth) - (0.5 * inflation) + (0.02 * cycle_factor)
        
        # Add some randomness
        industry_growth += random.uniform(-0.01, 0.01)
        
        # Create industry condition entry
        industry_conditions[industry] = {
            "growth_rate": round(industry_growth, 4),
            "risk_score": round(random.uniform(0.2, 0.8), 2),
            "sentiment": random.choice(["positive", "neutral", "negative"]),
            "trend": random.choice(["improving", "stable", "declining"])
        }
    
    # Calculate market stability score
    stability_factors = [
        (1 - abs(interest_rate - base_interest_rate) / 0.03),
        (1 - abs(unemployment - base_unemployment) / 0.03),
        (1 - abs(gdp_growth - base_gdp_growth) / 0.02),
        (1 - abs(inflation - base_inflation) / 0.02)
    ]
    market_stability = sum(stability_factors) / len(stability_factors)
    market_stability = max(0.1, min(0.9, market_stability))  # Bound between 0.1 and 0.9
    
    # Create market conditions dict
    market_data = {
        "entity_id": market_id,
        "entity_type": "MarketConditions",
        "time_period": time_period,
        "date": base_date.strftime("%Y-%m-%d"),
        "interest_rate": round(interest_rate, 4),
        "unemployment_rate": round(unemployment, 4),
        "gdp_growth_rate": round(gdp_growth, 4),
        "inflation_rate": round(inflation, 4),
        "market_stability": round(market_stability, 2),
        "industry_conditions": industry_conditions
    }
    
    # Add temporal attributes
    market_data.update(generate_temporal_attributes(base_date))
    
    # Add confidence attributes
    market_data.update(generate_confidence_attributes())
    
    return market_data

def generate_loan_application(company_profile, owner_profile, financial_data, market_conditions, time_period=0):
    """
    Generate a loan application based on company, owner, financial data and market conditions.
    
    Args:
        company_profile: Company entity data
        owner_profile: Owner entity data
        financial_data: Financial data entity
        market_conditions: Market conditions entity
        time_period: Time period offset
        
    Returns:
        Dictionary containing loan application data
    """
    # Generate base date for temporal attributes
    base_date = datetime.now() + timedelta(days=90 * time_period)
    
    # Extract key information
    industry = company_profile["industry"]
    total_assets = financial_data["total_assets"]
    annual_revenue = financial_data["annual_revenue"]
    market_stability = market_conditions["market_stability"]
    interest_rate_base = market_conditions["interest_rate"]
    industry_growth = market_conditions["industry_conditions"][industry]["growth_rate"]
    
    # Determine loan purpose based on industry
    if industry in ["Manufacturing", "Construction"]:
        purposes = ["Equipment Purchase", "Facility Expansion", "Working Capital"]
    elif industry in ["Retail", "Wholesale"]:
        purposes = ["Inventory Purchase", "Store Renovation", "Working Capital"]
    elif industry in ["Technology", "Professional Services"]:
        purposes = ["Office Expansion", "Equipment Purchase", "Acquisition"]
    else:
        purposes = ["Working Capital", "Refinance", "Expansion"]
    
    loan_purpose = random.choice(purposes)
    
    # Determine realistic loan amount based on company size and purpose
    if loan_purpose in ["Equipment Purchase", "Inventory Purchase"]:
        base_amount = min(total_assets * 0.3, annual_revenue * 0.5)
    elif loan_purpose in ["Facility Expansion", "Store Renovation", "Office Expansion"]:
        base_amount = min(total_assets * 0.7, annual_revenue * 0.8)
    elif loan_purpose == "Acquisition":
        base_amount = min(total_assets * 1.2, annual_revenue * 1.5)
    else:  # Working Capital, Refinance
        base_amount = min(total_assets * 0.4, annual_revenue * 0.3)
    
    # Add some randomness but keep reasonable
    loan_amount = int(base_amount * random.uniform(0.8, 1.2))
    
    # Term length based on purpose and amount
    if loan_purpose in ["Working Capital", "Inventory Purchase"]:
        term_years = random.choice([1, 2, 3])
    elif loan_purpose in ["Equipment Purchase"]:
        term_years = random.choice([3, 5, 7])
    else:  # Real estate, expansion, acquisition
        term_years = random.choice([5, 7, 10, 15])
    
    # Collateral based on purpose and financials
    if loan_purpose in ["Facility Expansion", "Store Renovation"]:
        collateral_type = "Real Estate"
        collateral_value = loan_amount * random.uniform(1.2, 1.5)
    elif loan_purpose == "Equipment Purchase":
        collateral_type = "Equipment"
        collateral_value = loan_amount * random.uniform(1.0, 1.2)
    elif loan_purpose == "Acquisition":
        collateral_type = "Business Assets"
        collateral_value = loan_amount * random.uniform(1.1, 1.3)
    else:
        collateral_type = random.choice(["Accounts Receivable", "Inventory", "Personal Guarantee", "None"])
        if collateral_type == "None":
            collateral_value = 0
        else:
            collateral_value = loan_amount * random.uniform(0.7, 1.1)
    
    # Interest rate based on market conditions, credit score, and collateral
    risk_premium = 0
    
    # Add premium for market instability
    risk_premium += (1 - market_stability) * 0.04
    
    # Add premium for poor credit
    if owner_profile["owner_credit_score"] < 650:
        risk_premium += 0.03
    elif owner_profile["owner_credit_score"] < 700:
        risk_premium += 0.015
    
    # Add premium for collateral type
    if collateral_type == "None":
        risk_premium += 0.025
    elif collateral_type == "Personal Guarantee":
        risk_premium += 0.015
    
    # Interest rate calculation
    interest_rate = interest_rate_base + risk_premium + random.uniform(-0.005, 0.01)
    interest_rate = round(max(interest_rate_base, min(interest_rate, 0.15)), 4)  # Cap between base and 15%
    
    # Generate application dates
    application_date = base_date - timedelta(days=random.randint(1, 30))
    desired_funding_date = application_date + timedelta(days=random.randint(30, 90))
    
    # Generate loan ID
    loan_id = f"{company_profile['entity_id']}_loan_t{time_period}"
    
    # Create loan application dict
    loan_data = {
        "entity_id": loan_id,
        "entity_type": "LoanApplication",
        "company_id": company_profile["entity_id"],
        "company_name": company_profile["company_name"],
        "owner_id": owner_profile["entity_id"],
        "owner_name": owner_profile["owner_name"],
        "time_period": time_period,
        "loan_amount": loan_amount,
        "loan_purpose": loan_purpose,
        "term_years": term_years,
        "interest_rate": interest_rate,
        "collateral_type": collateral_type,
        "collateral_value": collateral_value,
        "ltv_ratio": 0 if collateral_value == 0 else loan_amount / collateral_value,
        "application_date": application_date.strftime("%Y-%m-%d"),
        "desired_funding_date": desired_funding_date.strftime("%Y-%m-%d"),
        "market_condition_id": market_conditions["entity_id"],
        "market_interest_rate": market_conditions["interest_rate"],
        "industry_growth": industry_growth
    }
    
    # Add temporal attributes
    loan_data.update(generate_temporal_attributes(base_date))
    
    # Add confidence attributes
    loan_data.update(generate_confidence_attributes())
    
    return loan_data

def calculate_risk_metrics(company_profile, owner_profile, financial_data, loan_application, market_conditions):
    """
    Calculate risk metrics for loan application with temporal evolution.
    
    Args:
        company_profile: Company entity data
        owner_profile: Owner entity data
        financial_data: Financial data entity
        loan_application: Loan application entity
        market_conditions: Market conditions entity
        
    Returns:
        Dictionary containing risk metrics data
    """
    # Extract key information
    years_in_business = company_profile["years_in_business"]
    debt_service_coverage = financial_data["debt_service_coverage_ratio"]
    debt_to_equity = financial_data["debt_to_equity"]
    current_ratio = financial_data["current_ratio"]
    owner_credit = owner_profile["owner_credit_score"]
    ltv = loan_application["ltv_ratio"]
    market_stability = market_conditions["market_stability"]
    industry = company_profile["industry"]
    industry_growth = market_conditions["industry_conditions"][industry]["growth_rate"]
    
    # Risk scoring (higher score = higher risk)
    risk_score = 0
    risk_factors = {}
    
    # Years in business factor (newer = riskier)
    if years_in_business < 2:
        risk_score += 30
        risk_factors["years_in_business_risk"] = "High"
    elif years_in_business < 5:
        risk_score += 15
        risk_factors["years_in_business_risk"] = "Medium"
    elif years_in_business < 10:
        risk_score += 5
        risk_factors["years_in_business_risk"] = "Low"
    else:
        risk_factors["years_in_business_risk"] = "None"
    
    # DSCR factor
    if debt_service_coverage < 1.0:
        risk_score += 30
        risk_factors["dscr_risk"] = "High"
    elif debt_service_coverage < 1.25:
        risk_score += 15
        risk_factors["dscr_risk"] = "Medium"
    elif debt_service_coverage < 1.5:
        risk_score += 5
        risk_factors["dscr_risk"] = "Low"
    else:
        risk_factors["dscr_risk"] = "None"
    
    # D/E factor
    if debt_to_equity > 3.0:
        risk_score += 25
        risk_factors["de_ratio_risk"] = "High"
    elif debt_to_equity > 2.0:
        risk_score += 15
        risk_factors["de_ratio_risk"] = "Medium"
    elif debt_to_equity > 1.0:
        risk_score += 5
        risk_factors["de_ratio_risk"] = "Low"
    else:
        risk_factors["de_ratio_risk"] = "None"
    
    # Current ratio
    if current_ratio < 1.0:
        risk_score += 20
        risk_factors["current_ratio_risk"] = "High"
    elif current_ratio < 1.2:
        risk_score += 10
        risk_factors["current_ratio_risk"] = "Medium"
    elif current_ratio < 1.5:
        risk_score += 5
        risk_factors["current_ratio_risk"] = "Low"
    else:
        risk_factors["current_ratio_risk"] = "None"
    
    # Owner credit
    if owner_credit < 620:
        risk_score += 25
        risk_factors["credit_risk"] = "High"
    elif owner_credit < 680:
        risk_score += 15
        risk_factors["credit_risk"] = "Medium"
    elif owner_credit < 720:
        risk_score += 5
        risk_factors["credit_risk"] = "Low"
    else:
        risk_factors["credit_risk"] = "None"
    
    # Loan to value
    if ltv > 0.9:
        risk_score += 20
        risk_factors["ltv_risk"] = "High"
    elif ltv > 0.8:
        risk_score += 10
        risk_factors["ltv_risk"] = "Medium"
    elif ltv > 0.7:
        risk_score += 5
        risk_factors["ltv_risk"] = "Low"
    else:
        risk_factors["ltv_risk"] = "None"
    
    # Market conditions
    if market_stability < 0.3:
        risk_score += 15
        risk_factors["market_risk"] = "High"
    elif market_stability < 0.6:
        risk_score += 7
        risk_factors["market_risk"] = "Medium"
    else:
        risk_factors["market_risk"] = "Low"
    
    # Industry growth
    if industry_growth < 0:
        risk_score += 15
        risk_factors["industry_risk"] = "High"
    elif industry_growth < 0.02:
        risk_score += 7
        risk_factors["industry_risk"] = "Medium"
    else:
        risk_factors["industry_risk"] = "Low"
    
    # Determine approval likelihood
    if risk_score < 30:
        approval_status = "Highly Likely"
        approval_probability = random.uniform(0.85, 0.95)
    elif risk_score < 60:
        approval_status = "Likely"
        approval_probability = random.uniform(0.65, 0.85)
    elif risk_score < 90:
        approval_status = "Possible"
        approval_probability = random.uniform(0.35, 0.65)
    else:
        approval_status = "Unlikely"
        approval_probability = random.uniform(0.05, 0.35)
    
    # Reflect the 5-10% overall approval rate mentioned
    adjusted_probability = approval_probability * 0.15  # Scale to get overall ~10% approval rate
    
    # Generate risk metric ID
    risk_id = f"{loan_application['entity_id']}_risk"
    
    # Create risk metrics dict
    risk_data = {
        "entity_id": risk_id,
        "entity_type": "RiskMetrics",
        "loan_id": loan_application["entity_id"],
        "company_id": company_profile["entity_id"],
        "company_name": company_profile["company_name"],
        "time_period": loan_application["time_period"],
        "risk_score": risk_score,
        "approval_status": approval_status,
        "approval_probability": adjusted_probability,
        "risk_factors": risk_factors
    }
    
    # Add temporal attributes
    risk_data.update(generate_temporal_attributes(datetime.now()))
    
    # Add confidence attributes with lower confidence (more subjective assessment)
    confidence_attrs = generate_confidence_attributes()
    confidence_attrs["confidence"] = round(confidence_attrs["confidence"] * 0.9, 3)  # Slightly lower confidence
    risk_data.update(confidence_attrs)
    
    return risk_data

def generate_complete_loan_application_set(time_periods=None):
    """
    Generate a complete loan application with all components across multiple time periods.
    
    Args:
        time_periods: Number of time periods to generate (default to Config value)
        
    Returns:
        Dictionary of entity collections
    """
    # Use config value if time_periods not provided
    if time_periods is None:
        time_periods = Config.SYNTHETIC_TIME_PERIODS
    
    # Initialize collections
    entities = {
        "companies": [],
        "owners": [],
        "financials": [],
        "market_conditions": [],
        "loan_applications": [],
        "risk_metrics": []
    }
    
    # Generate market conditions for all time periods first
    market_conditions = []
    for period in range(time_periods):
        market_condition = generate_market_conditions(period)
        market_conditions.append(market_condition)
        entities["market_conditions"].append(market_condition)
    
    # Generate companies and related entities
    logger.info(f"Generating {Config.SYNTHETIC_COMPANY_COUNT} synthetic companies with {time_periods} time periods each")
    for i in tqdm(range(Config.SYNTHETIC_COMPANY_COUNT)):
        # Create the company profile (constant across time periods)
        company_profile = generate_company_profile()
        entities["companies"].append(company_profile)
        
        # Create the owner profile (constant across time periods)
        owner_profile = generate_owner_profile(company_profile)
        entities["owners"].append(owner_profile)
        
        # For each time period, generate the time-varying components
        for period in range(time_periods):
            # Generate financial data for this period
            financial_data = generate_financial_data(company_profile, period)
            entities["financials"].append(financial_data)
            
            # Generate loan application
            loan_app = generate_loan_application(
                company_profile, 
                owner_profile, 
                financial_data, 
                market_conditions[period], 
                period
            )
            entities["loan_applications"].append(loan_app)
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(
                company_profile,
                owner_profile,
                financial_data,
                loan_app,
                market_conditions[period]
            )
            entities["risk_metrics"].append(risk_metrics)
    
    logger.info(f"Generated {len(entities['companies'])} companies with {len(entities['loan_applications'])} loan applications")
    return entities

def save_to_datastore(entities):
    """
    Save generated entities to the dataStore in appropriate formats.
    
    Args:
        entities: Dictionary of entity collections
    """
    logger.info("Saving generated entities to dataStore")
    
    # Ensure directories exist
    Config.initialize()
    
    # Save market conditions
    market_df = pd.DataFrame(entities["market_conditions"])
    market_df.to_parquet(Config.SYNTHETIC_DIR / "market_conditions.parquet", index=False)
    
    # Save companies
    companies_df = pd.DataFrame(entities["companies"])
    companies_df.to_parquet(Config.SYNTHETIC_DIR / "companies.parquet", index=False)
    
    # Save owners
    owners_df = pd.DataFrame(entities["owners"])
    owners_df.to_parquet(Config.SYNTHETIC_DIR / "owners.parquet", index=False)
    
    # Save financials
    financials_df = pd.DataFrame(entities["financials"])
    financials_df.to_parquet(Config.SYNTHETIC_DIR / "financials.parquet", index=False)
    
    # Save loan applications
    loans_df = pd.DataFrame(entities["loan_applications"])
    loans_df.to_parquet(Config.SYNTHETIC_DIR / "loan_applications.parquet", index=False)
    
    # Save risk metrics
    risks_df = pd.DataFrame(entities["risk_metrics"])
    risks_df.to_parquet(Config.SYNTHETIC_DIR / "risk_metrics.parquet", index=False)
    
    # Save combined dataset for convenience
    loans_df.to_csv(Config.SYNTHETIC_DIR / "loan_applications.csv", index=False)
    
    # Save some example entities as JSON for readability
    json_example_dir = Config.SYNTHETIC_DIR / "examples"
    json_example_dir.mkdir(exist_ok=True)
    
    with open(json_example_dir / "company_example.json", "w") as f:
        json.dump(entities["companies"][0], f, indent=2)
    
    with open(json_example_dir / "loan_example.json", "w") as f:
        json.dump(entities["loan_applications"][0], f, indent=2)
    
    with open(json_example_dir / "market_example.json", "w") as f:
        json.dump(entities["market_conditions"][0], f, indent=2)
    
    logger.info(f"Saved all entities to {Config.SYNTHETIC_DIR}")

def main():
    """Main function to generate and save synthetic loan data."""
    logger.info("Starting synthetic data generation for Dynamic-KG")
    
    # Generate entities across multiple time periods
    entities = generate_complete_loan_application_set()
    
    # Save to datastore
    save_to_datastore(entities)
    
    # Print summary statistics
    logger.info(f"Generated {len(entities['companies'])} companies")
    logger.info(f"Generated {len(entities['loan_applications'])} loan applications")
    
    # Calculate approval rate
    approvals = [x for x in entities["risk_metrics"] if x["approval_probability"] > 0.5]
    approval_rate = len(approvals) / len(entities["risk_metrics"])
    logger.info(f"Overall approval rate: {approval_rate:.2%}")
    
    logger.info("Synthetic data generation complete")

if __name__ == "__main__":
    main()