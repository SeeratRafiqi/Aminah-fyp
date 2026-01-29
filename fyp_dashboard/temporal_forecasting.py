"""
Temporal Risk Forecasting Functions
Predict how buildings will deteriorate from 2024 to 2030
"""

import pandas as pd
import numpy as np
from utils.helpers import predict_risk

def project_component_condition(current_value, trend_rate, years_ahead):
    """
    Project future component condition based on current value and degradation rate
    
    Parameters:
    - current_value: Current condition (1-5)
    - trend_rate: Rate of change per year (from WALL_RATE, WIRING_RATE, etc.)
    - years_ahead: How many years to project (e.g., 1 for 2024, 7 for 2030)
    
    Returns:
    - Projected condition (capped at 1-5)
    """
    # Project forward using linear degradation
    projected = current_value + (trend_rate * years_ahead)
    
    # Cap between 1 (best) and 5 (worst)
    return max(1, min(5, projected))

def calculate_risk_score_for_year(row, target_year):
    """
    Calculate risk score for any year (historical or future)

    Parameters:
    - row: Building data (pandas Series)
    - target_year: Target year (e.g., 2020, 2023, 2027, 2030)

    Returns:
    - Dictionary with projected/historical values and risk score
    """
    # Component weights (from your risk scoring logic)
    COMPONENT_WEIGHTS = {'WIRING': 3.5, 'PILLAR': 3.0, 'WALL': 2.0, 'PAINT': 1.0}
    CONDITION_SEVERITY = {1: 0, 2: 1, 3: 3, 4: 5, 5: 7}

    # Calculate years from 2023 baseline (negative for historical, positive for future)
    years_from_2023 = target_year - 2023

    # Get conditions for target year
    projected_components = {}
    total_score = 0

    for component in ['WALL', 'WIRING', 'PILLAR', 'PAINT']:
        # Check if we have actual historical data for this year
        if target_year <= 2023 and target_year >= 2020:
            # Use actual historical data if available
            col_name = f'{component}_{target_year}'
            if col_name in row.index and pd.notna(row[col_name]):
                condition = float(row[col_name])
            else:
                # If historical data missing, project backward from 2023
                current_condition = row[f'{component}_2023']
                degradation_rate = row[f'{component}_RATE']
                condition = project_component_condition(current_condition, degradation_rate, years_from_2023)
        else:
            # Project future condition from 2023 baseline
            current_condition = row[f'{component}_2023']
            degradation_rate = row[f'{component}_RATE']
            condition = project_component_condition(current_condition, degradation_rate, years_from_2023)

        projected_components[component] = round(condition, 2)

        # Calculate risk contribution
        severity = CONDITION_SEVERITY[min(5, max(1, int(round(condition))))]
        weight = COMPONENT_WEIGHTS[component]
        total_score += severity * weight

    # Calculate maintenance risk score
    maintenance_risk = (total_score / (7 * sum(COMPONENT_WEIGHTS.values()))) * 100

    # Calculate age risk (building age at target year)
    age_at_target_year = row['BUILDING_AGE'] + years_from_2023
    if age_at_target_year < 10: age_risk = 0
    elif age_at_target_year < 20: age_risk = 3
    elif age_at_target_year < 30: age_risk = 6
    elif age_at_target_year < 40: age_risk = 10
    elif age_at_target_year < 50: age_risk = 15
    else: age_risk = 20

    # Total risk score
    total_risk_score = min(maintenance_risk + age_risk, 100)
    # Total risk score
    total_risk_score = min(maintenance_risk + age_risk, 100)

    # Determine risk level
    if total_risk_score >= 70:
        risk_level = 5
        risk_label = 'CRITICAL'
    elif total_risk_score >= 50:
        risk_level = 4
        risk_label = 'HIGH'
    elif total_risk_score >= 30:
        risk_level = 3
        risk_label = 'MEDIUM'
    elif total_risk_score >= 15:
        risk_level = 2
        risk_label = 'LOW'
    else:
        risk_level = 1
        risk_label = 'VERY_LOW'

    # Detect rapid deterioration (any component degrading > 0.5/year)
    rapid_deterioration = any([
        abs(row[f'{comp}_RATE']) > 0.5
        for comp in ['WALL', 'WIRING', 'PILLAR', 'PAINT']
    ])

    # Identify primary issue (worst component at target year)
    worst_component = max(projected_components.items(), key=lambda x: x[1])

    return {
        'year': target_year,
        'projected_wall': projected_components['WALL'],
        'projected_wiring': projected_components['WIRING'],
        'projected_pillar': projected_components['PILLAR'],
        'projected_paint': projected_components['PAINT'],
        'building_age': age_at_target_year,
        'maintenance_risk_score': maintenance_risk,
        'age_risk_score': age_risk,
        'total_risk_score': total_risk_score,
        'risk_level': risk_level,
        'risk_label': risk_label,
        'rapid_deterioration': rapid_deterioration,
        'primary_issue': worst_component[0],
        'primary_issue_value': worst_component[1]
    }

# Backward compatibility - keep old function name as alias
def calculate_future_risk_score(row, years_ahead):
    """
    Legacy function - calculates risk for future years from 2023
    Use calculate_risk_score_for_year() for more flexibility
    """
    return calculate_risk_score_for_year(row, 2023 + years_ahead)

def forecast_building(row, start_year=2024, end_year=2030):
    """
    Forecast a single building across multiple years

    Parameters:
    - row: Building data (pandas Series)
    - start_year: First year to forecast (can be 2020-2030)
    - end_year: Last year to forecast (can be 2020-2030)

    Returns:
    - List of dictionaries, one per year
    """
    forecasts = []

    for year in range(start_year, end_year + 1):
        forecast = calculate_risk_score_for_year(row, year)
        forecast['school_name'] = row['SCHOOL_NAME']
        forecast['block_name'] = row['BLOCK_NAME'] if pd.notna(row['BLOCK_NAME']) else 'Main Block'
        forecasts.append(forecast)

    return forecasts

def forecast_all_buildings(df, year):
    """
    Forecast all buildings for a specific year (historical or future)

    Parameters:
    - df: DataFrame with building data
    - year: Target year (2020-2030)

    Returns:
    - DataFrame with forecasted/historical values for all buildings
    """
    forecasts = []

    for idx, row in df.iterrows():
        forecast = calculate_risk_score_for_year(row, year)
        forecast['school_name'] = row['SCHOOL_NAME']
        forecast['block_name'] = row['BLOCK_NAME'] if pd.notna(row['BLOCK_NAME']) else 'Main Block'
        forecast['school_type'] = row['SCHOOL_TYPE']
        forecast['current_risk_2023'] = row['TOTAL_RISK_SCORE']
        forecast['current_label_2023'] = row['RISK_LABEL']
        forecasts.append(forecast)

    return pd.DataFrame(forecasts)

def compare_two_years(df, year1, year2):
    """
    Compare forecasts between two years (can include historical years)

    Parameters:
    - df: DataFrame with building data
    - year1: First year to compare (2020-2030)
    - year2: Second year to compare (2020-2030)

    Returns:
    - DataFrame with comparison metrics
    """
    forecast_year1 = forecast_all_buildings(df, year1)
    forecast_year2 = forecast_all_buildings(df, year2)

    # Create comparison
    comparison = pd.DataFrame({
        'school_name': forecast_year1['school_name'],
        'block_name': forecast_year1['block_name'],
        f'risk_score_{year1}': forecast_year1['total_risk_score'],
        f'risk_label_{year1}': forecast_year1['risk_label'],
        f'risk_score_{year2}': forecast_year2['total_risk_score'],
        f'risk_label_{year2}': forecast_year2['risk_label'],
        'risk_change': forecast_year2['total_risk_score'] - forecast_year1['total_risk_score'],
        'deterioration_rate': (forecast_year2['total_risk_score'] - forecast_year1['total_risk_score']) / (year2 - year1) if year2 != year1 else 0
    })

    return comparison