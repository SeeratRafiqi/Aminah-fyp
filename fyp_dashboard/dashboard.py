"""
School Canteen Maintenance Risk Dashboard
Redesigned with Stock Peers Template Style

Single-file dashboard with:
- Light professional theme
- Top navigation (no sidebar)
- Left panel filters
- Bordered containers
- Clean typography (no emojis in production UI)
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
DATA_PATH = os.path.join("data","canteen_labeled_enhanced.csv")
df=pd.read_csv(DATA_PATH)
# Import existing utilities
from utils.helpers import (
    load_models, 
    load_data, 
    predict_risk, 
    get_risk_color, 
    get_risk_icon
)

from temporal_forecasting import (
    calculate_future_risk_score,
    forecast_all_buildings,
    compare_two_years
)

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="School Canteen Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

# ============================================
# CUSTOM STYLING (Stock Peers Inspired)
# ============================================

st.markdown("""
<style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Navigation buttons */
    .stButton button {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        border-color: #615fff;
        background-color: #f8fafc;
    }
    
    /* All containers get borders (stock peers style) */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        background-color: white;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Headers - clean and professional */
    h1, h2, h3, h4 {
        font-weight: 600;
        color: #1e293b;
    }
    
    /* Reduce spacing between elements */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    
    /* Remove emoji from metric labels */
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Overview'

# ============================================
# NAVIGATION FUNCTIONS
# ============================================

def navigate_to(page_name):
    """Navigate to a different page"""
    st.session_state.current_page = page_name
    st.rerun()

# ============================================
# TOP NAVIGATION BAR
# ============================================

st.markdown("# School Canteen Maintenance Risk Dashboard")
st.markdown("""
The dashboard is designed as a decision-support tool to help stakeholders :green[identify
high-risk canteen blocks], :green[prioritise inspections], and :green[plan maintenance interventions]. 
Risk levels are generated using a **Random Forest** model trained on 
**historical building condition data**, **degradation trends**, and **age-related risk factors**.
""")
st.markdown("")

# Create navigation buttons
nav_cols = st.columns([2, 2, 2, 2, 2, 2])

pages = [
    ("Overview", "Overview"),
    ("Risk Prediction", "Risk Prediction"),
    ("Risk Analysis", "Risk Analysis"),
    ("Model Performance", "Model Performance"),
    ("Recommendations", "Recommendations")
]

for idx, (display_name, page_name) in enumerate(pages):
    with nav_cols[idx]:
        button_type = "primary" if st.session_state.current_page == page_name else "secondary"
        if st.button(display_name, use_container_width=True, type=button_type):
            navigate_to(page_name)

st.markdown("---")
st.markdown("")

# ============================================
# HELPER FUNCTIONS FOR COST ESTIMATION
# ============================================

# Component costs (from your existing code)
COMPONENT_COSTS = {
    'WIRING': {1: 0, 2: 5000, 3: 15000, 4: 30000, 5: 50000},
    'PILLAR': {1: 0, 2: 8000, 3: 20000, 4: 40000, 5: 70000},
    'WALL': {1: 0, 2: 3000, 3: 10000, 4: 25000, 5: 45000},
    'PAINT': {1: 0, 2: 2000, 3: 5000, 4: 10000, 5: 15000}
}

def estimate_maintenance_cost(row):
    """Estimate maintenance cost for a building (from your existing code)"""
    total_cost = 0
    component_costs = {}
    
    for component in ['WIRING', 'PILLAR', 'WALL', 'PAINT']:
        condition = int(row[f'{component}_2023'])
        cost = COMPONENT_COSTS[component][condition]
        component_costs[component] = cost
        total_cost += cost
    
    # Apply multipliers
    if row['RAPID_DETERIORATION']:
        total_cost *= 1.3
    if row['BUILDING_AGE'] > 50:
        total_cost *= 1.2
    
    return total_cost, component_costs

# ============================================
# LOAD DATA ONCE (cached)
# ============================================

@st.cache_data
def get_dashboard_data():
    """Load data for dashboard"""
    return load_data()

@st.cache_resource
def get_models():
    """Load models for dashboard"""
    return load_models()

# Load data and models
df = get_dashboard_data()
models = get_models()

# Check if data loaded successfully
if df is None:
    st.error("Failed to load dataset. Please check data/canteen_labeled_enhanced.csv")
    st.stop()

# ============================================================================
# PAGE ROUTING - All your 5 pages converted to single file
# ============================================================================

# ============================================================================
# PAGE 1: OVERVIEW (converted from pages/1_🏠_Home.py)
# ============================================================================

if st.session_state.current_page == "Overview":
    
    # Two-column layout: Left panel for filters, Main content for dashboard
    left_panel, main_content = st.columns([1, 3])
    
    # ===== LEFT PANEL: SYSTEM STATUS & FILTERS =====
    with left_panel:
        with st.container(border=True):
            st.markdown("#### System Status")
            # st.markdown("")
            
            # Dataset metrics
            total_buildings = len(df)
            unique_schools = df['SCHOOL_NAME'].nunique()
            
            st.metric("Total Buildings", total_buildings)
            st.metric("Total Schools", unique_schools)

            # st.markdown("")
            st.markdown("---")
            # st.markdown("")
            
            st.markdown("#### Quick Filters")
            # st.markdown("")
            
            # Risk level filter
            risk_filter = st.multiselect(
                "Risk Level",
                options=['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                default=['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            )
            
            # School type filter
            school_types = sorted(df['SCHOOL_TYPE'].unique().tolist())
            type_filter = st.multiselect(
                "School Type",
                options=school_types,
                default=school_types
            )
            
            # Age range
            age_range = st.slider(
                "Building Age (years)",
                min_value=int(df['BUILDING_AGE'].min()),
                max_value=int(df['BUILDING_AGE'].max()),
                value=(int(df['BUILDING_AGE'].min()), int(df['BUILDING_AGE'].max()))
            )
            
            st.markdown("")
            filter_applied = st.button("Apply Filters", use_container_width=True, type="primary")
        
        # Model status
        with st.container(border=True):
            st.markdown("#### Models")
            # st.markdown("")
            if models is not None:
                st.success("RF & ANN Ready")
                st.caption("98% accuracy")
            else:
                st.error("Models not loaded")
    
    # ===== MAIN CONTENT AREA =====
    with main_content:
        st.markdown("## Overview Dashboard")
        st.markdown("Comprehensive view of school canteen building maintenance status across:")

        # st.markdown("### Location Filters")
        col1, col2 = st.columns(2)
        with col1:
            state = st.selectbox(
                "State",
                options=["Terengganu"],
                index=0
            )
        with col2:
            district = st.selectbox(
                "District",
                options=["Kuala Terengganu"],
                index=0
            )
        st.caption(
            "**Note:** Location filters are fixed in this prototype as the current study scope "
            "focuses only on schools in Kuala Terengganu."
        )

        # st.markdown("")
        
        # Apply filters
        filtered_df = df[
            (df['RISK_LABEL'].isin(risk_filter)) &
            (df['SCHOOL_TYPE'].isin(type_filter)) &
            (df['BUILDING_AGE'] >= age_range[0]) &
            (df['BUILDING_AGE'] <= age_range[1])
        ]
        
        # Dataset info banner
        with st.container(border=True):
            st.markdown(f"""
            **Dataset Information:** This dashboard analyzes {len(df)} school canteen buildings 
            (generated from 68 real schools using Gaussian Copula synthetic data generation). 
            Currently showing :green[**{len(filtered_df)}**] buildings after filters.
            """)
        
        st.markdown("")
        
        # Top metrics row
        metric_cols = st.columns(5)
        
        with metric_cols[0]:
            with st.container(border=True):
                avg_risk = filtered_df['TOTAL_RISK_SCORE'].mean()
                st.metric("Avg Risk Score", f"{avg_risk:.1f}")
        
        with metric_cols[1]:
            with st.container(border=True):
                critical_count = len(filtered_df[filtered_df['RISK_LABEL'] == 'CRITICAL'])
                st.metric("Critical", critical_count)
        
        with metric_cols[2]:
            with st.container(border=True):
                high_count = len(filtered_df[filtered_df['RISK_LABEL'] == 'HIGH'])
                st.metric("High Risk", high_count)
        
        with metric_cols[3]:
            with st.container(border=True):
                avg_age = filtered_df['BUILDING_AGE'].mean()
                st.metric("Avg Age", f"{avg_age:.0f} yrs")
        
        with metric_cols[4]:
            with st.container(border=True):
                rapid_det = filtered_df['RAPID_DETERIORATION'].sum()
                st.metric("Rapid Deterioration", rapid_det)
        
        st.markdown("")
        
        # Risk distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("#### Risk Level Distribution")
                risk_counts = filtered_df['RISK_LABEL'].value_counts()
                
                fig_risk = go.Figure(data=[
                    go.Pie(
                        labels=risk_counts.index,
                        values=risk_counts.values,
                        marker=dict(colors=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#c0392b']),
                        hole=0.4
                    )
                ])
                fig_risk.update_layout(height=350, showlegend=True, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            with st.container(border=True):
                st.markdown("#### Component Health Overview")
                component_cols = ['WALL_2023', 'WIRING_2023', 'PILLAR_2023', 'PAINT_2023']
                component_avg = filtered_df[component_cols].mean()
                component_avg.index = ['Wall', 'Wiring', 'Pillar', 'Paint']
                
                fig_comp = go.Figure(data=[
                    go.Bar(
                        x=component_avg.index,
                        y=component_avg.values,
                        marker_color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'],
                        text=[f"{v:.2f}" for v in component_avg.values],
                        textposition='auto'
                    )
                ])
                fig_comp.update_layout(
                    height=350,
                    yaxis_title="Avg Condition (1-5)",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=40)
                )
                st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("")
        
        # Priority buildings table
        with st.container(border=True):
            st.markdown("#### High Priority Buildings")
            st.markdown("Buildings requiring immediate attention")
            # st.markdown("")
            
            priority_df = filtered_df[
                filtered_df['RISK_LABEL'].isin(['CRITICAL', 'HIGH'])
            ].sort_values('TOTAL_RISK_SCORE', ascending=False).head(10)
            
            if len(priority_df) > 0:
                display_cols = ['SCHOOL_NAME', 'BUILDING_AGE', 'RISK_LABEL', 
                               'TOTAL_RISK_SCORE', 'PRIMARY_ISSUE']
                st.dataframe(
                    priority_df[display_cols],
                    hide_index=True,
                    use_container_width=True,
                    height=300
                )
            else:
                st.success("No high-priority buildings in current filter")
        
        st.markdown("")
        
        # Analysis charts
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("#### Age vs Risk Score")
                fig_scatter = px.scatter(
                    filtered_df,
                    x='BUILDING_AGE',
                    y='TOTAL_RISK_SCORE',
                    color='RISK_LABEL',
                    color_discrete_map={
                        'VERY_LOW': '#2ecc71',
                        'LOW': '#3498db',
                        'MEDIUM': '#f39c12',
                        'HIGH': '#e74c3c',
                        'CRITICAL': '#c0392b'
                    },
                    height=350
                )
                fig_scatter.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            with st.container(border=True):
                st.markdown("#### Risk Score Distribution")
                fig_hist = go.Figure(data=[
                    go.Histogram(
                        x=filtered_df['TOTAL_RISK_SCORE'],
                        nbinsx=20,
                        marker_color='#615fff'
                    )
                ])
                fig_hist.update_layout(
                    height=350,
                    xaxis_title="Risk Score",
                    yaxis_title="Count",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=40)
                )
                st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================================
# PAGE 2: RISK PREDICTION WITH TEMPORAL FORECASTING
# Replace the existing "elif st.session_state.current_page == 'Risk Prediction':" section
# ============================================================================

elif st.session_state.current_page == "Risk Prediction":

    left_panel, main_content = st.columns([1, 3])

    # ===== LEFT PANEL: PREDICTION SETTINGS =====
    with left_panel:
        with st.container(border=True):
            st.markdown("#### Prediction Mode")
            # st.markdown("")

            prediction_mode = st.radio(
                "Select Mode",
                options=["Manual Entry", "Temporal Analysis", "School-Level Forecast"],
                help="Choose prediction approach"
            )

        # Mode-specific settings
        if prediction_mode == "Temporal Analysis":
            with st.container(border=True):
                st.markdown("#### Analysis Settings")
                # st.markdown("")

                analysis_type = st.radio(
                    "View Type",
                    ["Single Year", "Year Comparison"]
                )

                if analysis_type == "Single Year":
                    target_year = st.selectbox(
                        "Target Year",
                        options=list(range(2024, 2031)),
                        index=3  # Default to 2027
                    )
                else:  # Year Comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        year1 = st.selectbox("Year 1", list(range(2020, 2031)), index=0)  # Now starts from 2020
                    with col2:
                        year2 = st.selectbox("Year 2", list(range(2020, 2031)), index=10)  # Default to 2030

        elif prediction_mode == "School-Level Forecast":
            with st.container(border=True):
                st.markdown("#### School Selection")
                # st.markdown("")

                schools = sorted(df['SCHOOL_NAME'].unique())
                selected_school = st.selectbox("Select School", schools)

                # st.markdown("")
                st.markdown("---")
                # st.markdown("")

                st.markdown("#### Forecast Year")
                # st.markdown("")

                forecast_year = st.selectbox(
                    "Year",
                    options=list(range(2024, 2031)),
                    index=3
                )

        # Model info
        with st.container(border=True):
            st.markdown("#### Model Info")
            # st.markdown("")
            st.success("Random Forest")
            st.caption("98% accuracy")
            if models is None:
                st.error("Models not loaded")

    # ===== MAIN CONTENT AREA =====
    with main_content:
        st.markdown("## Risk Prediction")

        # ========================================
        # MODE 1: MANUAL ENTRY (UNCHANGED)
        # ========================================
        if prediction_mode == "Manual Entry":
            st.markdown("Predict risk for custom building parameters")
            # st.markdown("")

            with st.container(border=True):
                st.markdown("#### Building Conditions")
                # st.markdown("")

                with st.form("prediction_form"):
                    st.markdown("**Current Conditions (2023)**")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        wall_2023 = st.selectbox("Wall", [1, 2, 3, 4, 5], index=2)
                    with col2:
                        wiring_2023 = st.selectbox("Wiring", [1, 2, 3, 4, 5], index=2)
                    with col3:
                        pillar_2023 = st.selectbox("Pillar", [1, 2, 3, 4, 5], index=2)
                    with col4:
                        paint_2023 = st.selectbox("Paint", [1, 2, 3, 4, 5], index=2)

                    st.markdown("")
                    st.markdown("**Historical Averages**")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        wall_avg = st.number_input("Wall Avg", 1.0, 5.0, float(wall_2023), 0.1)
                    with col2:
                        wiring_avg = st.number_input("Wiring Avg", 1.0, 5.0, float(wiring_2023), 0.1)
                    with col3:
                        pillar_avg = st.number_input("Pillar Avg", 1.0, 5.0, float(pillar_2023), 0.1)
                    with col4:
                        paint_avg = st.number_input("Paint Avg", 1.0, 5.0, float(paint_2023), 0.1)

                    st.markdown("")
                    st.markdown("**Building Characteristics**")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        building_age = st.number_input("Age (years)", 0, 120, 30)
                    with col2:
                        enrol = st.number_input("Enrollment", 0, 5000, 500)
                    with col3:
                        teachers = st.number_input("Teachers", 0, 300, 50)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        wall_trend = st.slider("Wall Trend", -4, 4, 0)
                    with col2:
                        wiring_trend = st.slider("Wiring Trend", -4, 4, 0)
                    with col3:
                        pillar_trend = st.slider("Pillar Trend", -4, 4, 0)
                    with col4:
                        paint_trend = st.slider("Paint Trend", -4, 4, 0)

                    rapid_det = st.checkbox("Rapid Deterioration")

                    st.markdown("")
                    submitted = st.form_submit_button("Predict Risk", use_container_width=True, type="primary")

                if submitted:
                    # Calculate risk score (from your existing logic)
                    CONDITION_SEVERITY = {1: 0, 2: 1, 3: 3, 4: 5, 5: 7}
                    WEIGHTS = {'WIRING': 3.5, 'PILLAR': 3.0, 'WALL': 2.0, 'PAINT': 1.0}

                    total_score = 0
                    for comp, weight in WEIGHTS.items():
                        if comp == 'WALL':
                            curr, avg, trend = wall_2023, wall_avg, wall_trend
                        elif comp == 'WIRING':
                            curr, avg, trend = wiring_2023, wiring_avg, wiring_trend
                        elif comp == 'PILLAR':
                            curr, avg, trend = pillar_2023, pillar_avg, pillar_trend
                        else:
                            curr, avg, trend = paint_2023, paint_avg, paint_trend

                        curr_sev = CONDITION_SEVERITY[curr]
                        avg_sev = CONDITION_SEVERITY[int(round(avg))]
                        score = (curr_sev * 0.5 + avg_sev * 0.3 + max(trend, 0) * 0.2) * weight
                        total_score += score

                    maintenance_risk = (total_score / (7 * sum(WEIGHTS.values()))) * 100

                    if building_age < 10:
                        age_risk = 0
                    elif building_age < 20:
                        age_risk = 3
                    elif building_age < 30:
                        age_risk = 6
                    elif building_age < 40:
                        age_risk = 10
                    elif building_age < 50:
                        age_risk = 15
                    else:
                        age_risk = 20

                    total_risk_score = min(maintenance_risk + age_risk, 100)

                    input_data = {
                        'WALL_2023': wall_2023, 'WIRING_2023': wiring_2023,
                        'PILLAR_2023': pillar_2023, 'PAINT_2023': paint_2023,
                        'WALL_AVG': wall_avg, 'WIRING_AVG': wiring_avg,
                        'PILLAR_AVG': pillar_avg, 'PAINT_AVG': paint_avg,
                        'WALL_TREND': wall_trend, 'WIRING_TREND': wiring_trend,
                        'PILLAR_TREND': pillar_trend, 'PAINT_TREND': paint_trend,
                        'BUILDING_AGE': building_age, 'ENROL_JUN23': enrol,
                        'TEACHER_JUN23': teachers, 'TOTAL_RISK_SCORE': total_risk_score,
                        'RAPID_DETERIORATION': int(rapid_det)
                    }

                    result = predict_risk(input_data, models, use_ann=False)

                    st.markdown("")
                    with st.container(border=True):
                        st.markdown("#### Prediction Result")
                        # st.markdown("")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Risk Score", f"{result['risk_score']:.1f}/100")
                        with col2:
                            st.metric("Risk Level", result['risk_label'])
                        with col3:
                            st.metric("Confidence", f"{result['confidence']:.1f}%")

                        st.markdown("")
                        st.markdown("#### Recommendations")
                        if result['risk_label'] in ['HIGH', 'CRITICAL']:
                            st.error("**URGENT ACTION REQUIRED**")
                            st.markdown("- Schedule immediate inspection\n- Prioritize repairs")
                        elif result['risk_label'] == 'MEDIUM':
                            st.warning("**MAINTENANCE NEEDED**")
                            st.markdown("- Schedule within 3 months")
                        else:
                            st.success("**GOOD CONDITION**")
                            st.markdown("- Continue monitoring")

        # ========================================
        # MODE 2: TEMPORAL ANALYSIS (NEW!)
        # ========================================
        elif prediction_mode == "Temporal Analysis":
            st.markdown("Analyze risk trends across all buildings over time (2024-2030)")
            # st.markdown("")

            if analysis_type == "Single Year":
                # Import forecasting functions
                from temporal_forecasting import forecast_all_buildings

                # Generate forecast
                with st.spinner(f"Forecasting for {target_year}..."):
                    forecast_df = forecast_all_buildings(df, target_year)

                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    with st.container(border=True):
                        avg_risk = forecast_df['total_risk_score'].mean()
                        current_avg = df['TOTAL_RISK_SCORE'].mean()
                        delta = avg_risk - current_avg
                        st.metric(f"Avg Risk {target_year}", f"{avg_risk:.1f}",
                                  delta=f"{delta:+.1f} from 2023")

                with col2:
                    with st.container(border=True):
                        critical_count = len(forecast_df[forecast_df['risk_label'] == 'CRITICAL'])
                        current_critical = len(df[df['RISK_LABEL'] == 'CRITICAL'])
                        st.metric("Critical Buildings", critical_count,
                                  delta=f"{critical_count - current_critical:+d}")

                with col3:
                    with st.container(border=True):
                        high_count = len(forecast_df[forecast_df['risk_label'] == 'HIGH'])
                        current_high = len(df[df['RISK_LABEL'] == 'HIGH'])
                        st.metric("High Risk", high_count,
                                  delta=f"{high_count - current_high:+d}")

                with col4:
                    with st.container(border=True):
                        rapid_count = forecast_df['rapid_deterioration'].sum()
                        st.metric("Rapid Deterioration", rapid_count)

                st.markdown("")

                # Risk distribution comparison
                col1, col2 = st.columns(2)

                with col1:
                    with st.container(border=True):
                        st.markdown(f"#### Risk Distribution {target_year}")

                        risk_counts = forecast_df['risk_label'].value_counts()
                        fig = go.Figure(data=[
                            go.Pie(
                                labels=risk_counts.index,
                                values=risk_counts.values,
                                marker=dict(colors=['#c0392b', '#e74c3c', '#f39c12', '#3498db', '#2ecc71']),
                                hole=0.4
                            )
                        ])
                        fig.update_layout(height=350, showlegend=True, margin=dict(l=20, r=20, t=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    with st.container(border=True):
                        st.markdown("#### Risk Score Distribution")

                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=forecast_df['total_risk_score'],
                            name=str(target_year),
                            nbinsx=20,
                            marker_color='#615fff',
                            opacity=0.7
                        ))
                        fig.add_trace(go.Histogram(
                            x=df['TOTAL_RISK_SCORE'],
                            name='2023 (Current)',
                            nbinsx=20,
                            marker_color='#95a5a6',
                            opacity=0.5
                        ))
                        fig.update_layout(
                            barmode='overlay',
                            height=350,
                            xaxis_title="Risk Score",
                            yaxis_title="Count",
                            margin=dict(l=20, r=20, t=20, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                st.markdown("")

                # Top deteriorating buildings
                with st.container(border=True):
                    st.markdown(f"#### Top 20 Highest Risk Buildings in {target_year}")

                    top_risk = forecast_df.sort_values('total_risk_score', ascending=False).head(20)

                    display_df = pd.DataFrame({
                        'School': top_risk['school_name'],
                        'Block': top_risk['block_name'],
                        f'Risk Score {target_year}': top_risk['total_risk_score'].round(1),
                        f'Risk Level {target_year}': top_risk['risk_label'],
                        'Current Risk 2023': top_risk['current_risk_2023'].round(1),
                        'Change': (top_risk['total_risk_score'] - top_risk['current_risk_2023']).round(1),
                        'Primary Issue': top_risk['primary_issue']
                    })

                    st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)

            else:  # Year Comparison
                from temporal_forecasting import compare_two_years

                if year1 >= year2:
                    st.error("Year 2 must be after Year 1")
                else:
                    # Add info about historical vs future
                    comparison_type = ""
                    if year1 <= 2023 and year2 <= 2023:
                        comparison_type = "📊 **Historical Comparison** (both years have actual data)"
                    elif year1 <= 2023 and year2 > 2023:
                        comparison_type = "📊 **Baseline vs Forecast** (comparing historical baseline with future projection)"
                    else:
                        comparison_type = "📊 **Future Projection Comparison**"

                    st.info(comparison_type)
                    # st.markdown("")

                    with st.spinner(f"Comparing {year1} vs {year2}..."):
                        comparison_df = compare_two_years(df, year1, year2)

                    # Metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        with st.container(border=True):
                            avg_change = comparison_df['risk_change'].mean()
                            st.metric("Avg Risk Change", f"{avg_change:+.1f}",
                                      delta=f"{avg_change / (year2 - year1):.2f} per year")

                    with col2:
                        with st.container(border=True):
                            worsening = len(comparison_df[comparison_df['risk_change'] > 10])
                            st.metric("Significantly Worsening", worsening,
                                      delta="Change > 10 points")

                    with col3:
                        with st.container(border=True):
                            upgrading = len(comparison_df[comparison_df[f'risk_label_{year2}'] > comparison_df[
                                f'risk_label_{year1}']])
                            st.metric("Risk Level Upgrades", upgrading,
                                      delta="Moving to higher risk")

                    st.markdown("")

                    # Scatter comparison
                    with st.container(border=True):
                        st.markdown(f"#### Risk Score: {year1} vs {year2}")

                        fig = go.Figure()

                        # Add scatter points
                        fig.add_trace(go.Scatter(
                            x=comparison_df[f'risk_score_{year1}'],
                            y=comparison_df[f'risk_score_{year2}'],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=comparison_df['risk_change'],
                                colorscale='RdYlGn_r',
                                showscale=True,
                                colorbar=dict(title="Risk Change")
                            ),
                            text=comparison_df['school_name'],
                            hovertemplate='<b>%{text}</b><br>' +
                                          f'{year1}: %{{x:.1f}}<br>' +
                                          f'{year2}: %{{y:.1f}}<br>' +
                                          'Change: %{marker.color:.1f}<extra></extra>'
                        ))

                        # Add diagonal line (no change)
                        fig.add_trace(go.Scatter(
                            x=[0, 100],
                            y=[0, 100],
                            mode='lines',
                            line=dict(color='gray', dash='dash'),
                            name='No Change',
                            showlegend=True
                        ))

                        fig.update_layout(
                            xaxis_title=f"Risk Score {year1}",
                            yaxis_title=f"Risk Score {year2}",
                            height=500,
                            hovermode='closest'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("")

                    # Top deteriorating
                    with st.container(border=True):
                        st.markdown("#### Most Rapidly Deteriorating Buildings")

                        top_deteriorating = comparison_df.sort_values('risk_change', ascending=False).head(15)

                        display_df = pd.DataFrame({
                            'School': top_deteriorating['school_name'],
                            'Block': top_deteriorating['block_name'],
                            f'{year1} Score': top_deteriorating[f'risk_score_{year1}'].round(1),
                            f'{year1} Level': top_deteriorating[f'risk_label_{year1}'],
                            f'{year2} Score': top_deteriorating[f'risk_score_{year2}'].round(1),
                            f'{year2} Level': top_deteriorating[f'risk_label_{year2}'],
                            'Change': top_deteriorating['risk_change'].round(1),
                            'Rate/Year': top_deteriorating['deterioration_rate'].round(2)
                        })

                        st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)

        # ========================================
        # MODE 3: SCHOOL-LEVEL FORECAST (NEW!)
        # ========================================
        else:  # School-Level Forecast
            st.markdown(f"Detailed forecast for {selected_school}")
            # st.markdown("")

            from temporal_forecasting import forecast_building

            # Get all blocks for this school
            school_df = df[df['SCHOOL_NAME'] == selected_school]

            # Overview of current state
            with st.container(border=True):
                st.markdown("#### Current Status (2023)")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Blocks", len(school_df))
                with col2:
                    avg_risk = school_df['TOTAL_RISK_SCORE'].mean()
                    st.metric("Avg Risk Score", f"{avg_risk:.1f}")
                with col3:
                    high_risk = len(school_df[school_df['RISK_LABEL'].isin(['HIGH', 'CRITICAL'])])
                    st.metric("High Risk Blocks", high_risk)
                with col4:
                    avg_age = school_df['BUILDING_AGE'].mean()
                    st.metric("Avg Age", f"{avg_age:.0f} yrs")

            st.markdown("")

            # Forecast each block
            for idx, row in school_df.iterrows():
                block_name = row['BLOCK_NAME'] if pd.notna(row['BLOCK_NAME']) else 'Main Block'

                with st.container(border=True):
                    st.markdown(f"### {block_name}")

                    # Current vs Forecast
                    forecast = calculate_future_risk_score(row, forecast_year - 2023)

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("#### Forecast Results")

                        metric_cols = st.columns(4)

                        with metric_cols[0]:
                            st.metric(
                                f"Risk Score {forecast_year}",
                                f"{forecast['total_risk_score']:.1f}",
                                delta=f"{forecast['total_risk_score'] - row['TOTAL_RISK_SCORE']:+.1f} from 2023"
                            )

                        with metric_cols[1]:
                            st.metric(
                                f"Risk Level {forecast_year}",
                                forecast['risk_label']
                            )

                        with metric_cols[2]:
                            st.metric(
                                "Building Age",
                                f"{forecast['building_age']:.0f} yrs",
                                delta=f"+{forecast_year - 2023} yrs"
                            )

                        with metric_cols[3]:
                            st.metric(
                                "Primary Issue",
                                forecast['primary_issue'],
                                delta=f"{forecast['primary_issue_value']:.1f}/5"
                            )

                    with col2:
                        st.markdown("#### Component Forecast")

                        components_df = pd.DataFrame({
                            'Component': ['Wall', 'Wiring', 'Pillar', 'Paint'],
                            '2023': [row['WALL_2023'], row['WIRING_2023'],
                                     row['PILLAR_2023'], row['PAINT_2023']],
                            str(forecast_year): [
                                forecast['projected_wall'],
                                forecast['projected_wiring'],
                                forecast['projected_pillar'],
                                forecast['projected_paint']
                            ]
                        })

                        st.dataframe(
                            components_df.style.format({
                                '2023': '{:.0f}',
                                str(forecast_year): '{:.2f}'
                            }),
                            hide_index=True,
                            use_container_width=True
                        )

                    # st.markdown("")

                    # Recommendations based on forecast
                    st.markdown("#### Recommended Actions")

                    if forecast['risk_label'] in ['HIGH', 'CRITICAL']:
                        st.error(f"**URGENT: Will require immediate attention by {forecast_year}**")
                        st.markdown(f"""
                        - **Primary concern:** {forecast['primary_issue']} (projected: {forecast['primary_issue_value']:.1f}/5)
                        - **Action:** Schedule preventive maintenance now to avoid critical failure
                        - **Budget:** Allocate funds for major repairs before {forecast_year}
                        """)
                    elif forecast['risk_label'] == 'MEDIUM':
                        st.warning(f"**MONITOR: Will need maintenance by {forecast_year}**")
                        st.markdown(f"""
                        - **Primary concern:** {forecast['primary_issue']}
                        - **Action:** Plan maintenance within next {forecast_year - 2024} years
                        - **Budget:** Include in medium-term planning
                        """)
                    else:
                        st.success(f"**GOOD: Expected to remain in good condition through {forecast_year}**")
                        st.markdown("""
                        - **Action:** Continue regular monitoring and preventive maintenance
                        """)

# ============================================================================
# PAGE 3: RISK ANALYSIS (converted from pages/3_📊_Risk_Analysis.py)
# ============================================================================

elif st.session_state.current_page == "Risk Analysis":
    
    left_panel, main_content = st.columns([1, 3])
    
    # ===== LEFT PANEL: ANALYSIS FILTERS =====
    with left_panel:
        with st.container(border=True):
            st.markdown("#### Analysis Filters")
            # st.markdown("")
            
            # Risk levels
            risk_levels = st.multiselect(
                "Risk Levels",
                ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                default=['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            )
            
            # Age range
            age_min, age_max = st.slider(
                "Building Age",
                int(df['BUILDING_AGE'].min()),
                int(df['BUILDING_AGE'].max()),
                (int(df['BUILDING_AGE'].min()), int(df['BUILDING_AGE'].max()))
            )
            
            # School types
            school_types = st.multiselect(
                "School Types",
                df['SCHOOL_TYPE'].unique(),
                default=df['SCHOOL_TYPE'].unique()
            )
            
            # Rapid deterioration
            show_rapid = st.checkbox("Only Rapid Deterioration")
        
        with st.container(border=True):
            st.markdown("#### Display Options")
            st.markdown("")
            
            analysis_type = st.radio(
                "Analysis Type",
                ["Overview", "Component Analysis", "Trend Analysis"]
            )
    
    # ===== MAIN CONTENT AREA =====
    with main_content:
        st.markdown("## Risk Analysis")
        st.markdown("Comprehensive analysis and insights")
        st.markdown("")
        
        # Apply filters
        filtered_df = df[
            (df['RISK_LABEL'].isin(risk_levels)) &
            (df['BUILDING_AGE'] >= age_min) &
            (df['BUILDING_AGE'] <= age_max) &
            (df['SCHOOL_TYPE'].isin(school_types))
        ]
        
        if show_rapid:
            filtered_df = filtered_df[filtered_df['RAPID_DETERIORATION'] == True]
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with st.container(border=True):
                avg_risk = filtered_df['TOTAL_RISK_SCORE'].mean()
                st.metric("Avg Risk Score", f"{avg_risk:.1f}")
        
        with col2:
            with st.container(border=True):
                high_pct = len(filtered_df[filtered_df['RISK_LABEL'].isin(['HIGH', 'CRITICAL'])]) / len(filtered_df) * 100
                st.metric("High Risk %", f"{high_pct:.1f}%")
        
        with col3:
            with st.container(border=True):
                avg_age = filtered_df['BUILDING_AGE'].mean()
                st.metric("Avg Age", f"{avg_age:.0f} yrs")
        
        with col4:
            with st.container(border=True):
                rapid_count = filtered_df['RAPID_DETERIORATION'].sum()
                st.metric("Rapid Deterioration", rapid_count)
        
        st.markdown("")
        
        # Charts based on analysis type
        if analysis_type == "Overview":
            col1, col2 = st.columns(2)
            
            with col1:
                with st.container(border=True):
                    st.markdown("#### Risk Distribution")
                    risk_counts = filtered_df['RISK_LABEL'].value_counts()
                    fig = go.Figure(data=[go.Pie(labels=risk_counts.index, values=risk_counts.values, hole=0.4)])
                    fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                with st.container(border=True):
                    st.markdown("#### Age vs Risk")
                    fig = px.scatter(filtered_df, x='BUILDING_AGE', y='TOTAL_RISK_SCORE', 
                                    color='RISK_LABEL', height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Component Analysis":
            with st.container(border=True):
                st.markdown("#### Component Conditions")
                
                components = ['WALL_2023', 'WIRING_2023', 'PILLAR_2023', 'PAINT_2023']
                component_names = ['Wall', 'Wiring', 'Pillar', 'Paint']
                
                cols = st.columns(4)
                for idx, (col_name, display_name) in enumerate(zip(components, component_names)):
                    with cols[idx]:
                        with st.container(border=True):
                            avg_cond = filtered_df[col_name].mean()
                            poor_count = len(filtered_df[filtered_df[col_name] >= 4])
                            
                            st.markdown(f"**{display_name}**")
                            st.metric("Avg Condition", f"{avg_cond:.2f}/5")
                            st.caption(f"{poor_count} need attention")
        
        else:  # Trend Analysis
            with st.container(border=True):
                st.markdown("#### Degradation Trends")
                
                trend_cols = ['WALL_TREND', 'WIRING_TREND', 'PILLAR_TREND', 'PAINT_TREND']
                trend_names = ['Wall', 'Wiring', 'Pillar', 'Paint']
                
                improving = []
                stable = []
                deteriorating = []
                
                for col in trend_cols:
                    improving.append(len(filtered_df[filtered_df[col] < 0]))
                    stable.append(len(filtered_df[filtered_df[col] == 0]))
                    deteriorating.append(len(filtered_df[filtered_df[col] > 0]))
                
                fig = go.Figure(data=[
                    go.Bar(name='Improving', x=trend_names, y=improving, marker_color='#2ecc71'),
                    go.Bar(name='Stable', x=trend_names, y=stable, marker_color='#3498db'),
                    go.Bar(name='Deteriorating', x=trend_names, y=deteriorating, marker_color='#e74c3c')
                ])
                fig.update_layout(barmode='stack', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("")
        
        # Data table
        with st.container(border=True):
            st.markdown("#### Building Data")
            display_cols = ['SCHOOL_NAME', 'BUILDING_AGE', 'RISK_LABEL', 'TOTAL_RISK_SCORE', 'PRIMARY_ISSUE']
            st.dataframe(filtered_df[display_cols].head(20), hide_index=True, use_container_width=True, height=400)

# ============================================================================
# PAGE 4: MODEL PERFORMANCE (converted from pages/4_🤖_Model_Performance.py)
# ============================================================================

elif st.session_state.current_page == "Model Performance":
    
    st.markdown("## Model Performance")
    st.markdown("Comparing Random Forest and Neural Network models")
    st.markdown("")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("#### Best Model")
            st.metric("Random Forest", "98%")
            st.caption("Recommended for production")
    
    with col2:
        with st.container(border=True):
            st.markdown("#### Improvement")
            st.metric("Baseline → Enhanced", "+4%")
            st.caption("Feature engineering impact")
    
    with col3:
        with st.container(border=True):
            st.markdown("#### Features Used")
            st.metric("Enhanced Model", "17")
            st.caption("vs 7 in baseline")
    
    st.markdown("")
    
    # Accuracy comparison
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.container(border=True):
            st.markdown("#### Model Accuracy Comparison")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Random Forest',
                x=['Baseline', 'Enhanced'],
                y=[0.94, 0.98],
                marker_color='steelblue',
                text=['94%', '98%'],
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name='Neural Network',
                x=['Baseline', 'Enhanced'],
                y=[0.92, 0.90],
                marker_color='coral',
                text=['92%', '90%'],
                textposition='auto'
            ))
            fig.update_layout(barmode='group', height=400, yaxis=dict(range=[0.85, 1.0]))
            fig.add_hline(y=0.95, line_dash="dash", line_color="green", annotation_text="Target: 95%")
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        with st.container(border=True):
            st.markdown("#### Key Findings")
            st.markdown("""
            **RF Enhanced: 98%**
            - Best performance
            - Stable with features
            - Production ready
            
            **ANN: 90%**
            - Overfitting on small data
            - Less interpretable
            - Not recommended
            """)
    
    st.markdown("")
    
    # Feature importance
    with st.container(border=True):
        st.markdown("#### Top Feature Importance (Random Forest)")
        
        features = ['TOTAL_RISK_SCORE', 'BUILDING_AGE', 'WIRING_2023', 'WIRING_AVG', 
                   'PILLAR_AVG', 'PILLAR_TREND', 'TEACHER_JUN23', 'PILLAR_2023']
        importance = [0.407, 0.117, 0.096, 0.081, 0.061, 0.054, 0.045, 0.034]
        
        fig = go.Figure(data=[
            go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker=dict(color=importance, colorscale='Viridis'),
                text=[f"{v:.3f}" for v in importance],
                textposition='auto'
            )
        ])
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: RECOMMENDATIONS (converted from pages/5_🔧_Maintenance_Recommendations.py)
# ============================================================================

elif st.session_state.current_page == "Recommendations":
    
    left_panel, main_content = st.columns([1, 3])
    
    # ===== LEFT PANEL: BUDGET & FILTERS =====
    with left_panel:
        with st.container(border=True):
            st.markdown("#### Priority Filters")
            st.markdown("")
            
            priority_filter = st.multiselect(
                "Risk Levels",
                ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
                default=['CRITICAL', 'HIGH', 'MEDIUM']
            )
        
        with st.container(border=True):
            st.markdown("#### Budget Planning")
            st.markdown("")
            
            budget_available = st.number_input(
                "Available Budget (RM)",
                min_value=0,
                max_value=10000000,
                value=500000,
                step=50000
            )
    
    # ===== MAIN CONTENT AREA =====
    with main_content:
        st.markdown("## Maintenance Recommendations")
        st.markdown("AI-powered maintenance planning and cost estimation")
        st.markdown("")
        
        # Calculate costs
        filtered_df = df[df['RISK_LABEL'].isin(priority_filter)].copy()
        filtered_df['ESTIMATED_COST'] = filtered_df.apply(lambda row: estimate_maintenance_cost(row)[0], axis=1)
        
        # Priority overview
        col1, col2, col3, col4 = st.columns(4)
        
        counts = {
            'CRITICAL': len(df[df['RISK_LABEL'] == 'CRITICAL']),
            'HIGH': len(df[df['RISK_LABEL'] == 'HIGH']),
            'MEDIUM': len(df[df['RISK_LABEL'] == 'MEDIUM']),
            'LOW': len(df[df['RISK_LABEL'].isin(['LOW', 'VERY_LOW'])])
        }
        
        colors = {'CRITICAL': '#c0392b', 'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#2ecc71'}
        
        for idx, (level, count) in enumerate(counts.items()):
            with [col1, col2, col3, col4][idx]:
                with st.container(border=True):
                    st.markdown(f"**{level}**")
                    st.metric("Count", count, delta=None)
        
        st.markdown("")
        
        # Cost summary
        total_cost = filtered_df['ESTIMATED_COST'].sum()
        priority_cost = filtered_df[filtered_df['RISK_LABEL'].isin(['CRITICAL', 'HIGH'])]['ESTIMATED_COST'].sum()
        coverage = (budget_available / total_cost * 100) if total_cost > 0 else 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container(border=True):
                st.markdown("**Total Cost**")
                st.metric("Estimated", f"RM {total_cost:,.0f}")
                st.caption(f"{len(filtered_df)} buildings")
        
        with col2:
            with st.container(border=True):
                st.markdown("**Priority Cost**")
                st.metric("Critical & High", f"RM {priority_cost:,.0f}")
                st.caption("Urgent repairs")
        
        with col3:
            with st.container(border=True):
                st.markdown("**Budget Coverage**")
                st.metric("Available", f"{min(coverage, 100):.0f}%")
                st.caption(f"RM {budget_available:,.0f}")
        
        st.markdown("")
        
        # Priority list
        with st.container(border=True):
            st.markdown("#### Priority Maintenance List")
            
            priority_df = filtered_df.sort_values('TOTAL_RISK_SCORE', ascending=False)
            
            display_cols = ['SCHOOL_NAME', 'RISK_LABEL', 'TOTAL_RISK_SCORE', 
                           'PRIMARY_ISSUE', 'ESTIMATED_COST']
            
            st.dataframe(
                priority_df[display_cols].head(20),
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            # Download button
            csv = priority_df[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Priority List (CSV)",
                csv,
                "maintenance_priority.csv",
                "text/csv",
                use_container_width=True
            )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 1rem;">
    <p style="margin: 0;">School Canteen Maintenance Risk Dashboard</p>
    <p style="margin: 0; font-size: 0.875rem;">Built with Streamlit • Random Forest 98% Accuracy</p>
</div>
""", unsafe_allow_html=True)

