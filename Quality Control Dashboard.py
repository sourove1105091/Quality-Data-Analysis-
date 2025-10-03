import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import math
import io
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import requests
from io import BytesIO
import base64
import datetime

# Try to import optional dependencies
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from pycaret.regression import setup, compare_models, create_model, predict_model, finalize_model, save_model, load_model, pull
    from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models
    from pycaret.time_series import setup as ts_setup, compare_models as ts_compare_models
    HAS_PYCARET = True
except ImportError:
    HAS_PYCARET = False

# Set page configuration
st.set_page_config(
    page_title="Advanced Quality Control Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2e86ab;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .success { color: #28a745; font-weight: bold; }
    .warning { color: #ffc107; font-weight: bold; }
    .danger { color: #dc3545; font-weight: bold; }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #28a745;
    }
    .profile-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .profile-image {
        border-radius: 50%;
        border: 4px solid white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .certification-badge {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .user-stats {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin: 20px 0;
    }
    .logo-img {
        height: 60px;
        opacity: 0.8;
        transition: opacity 0.3s;
    }
    .logo-img:hover {
        opacity: 1;
    }
    .user-registration {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .about-me {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to load profile image from Google Drive
def load_profile_image():
    try:
        # Your Google Drive picture link
        image_url = "https://drive.google.com/uc?id=13sguggcj6l8CjD8Glw4Udtw_oBgIeci8"
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        try:
            # Create a simple profile image with initials as fallback
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (150, 150), color='#667eea')
            d = ImageDraw.Draw(img)
            
            # Try to use a font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 60)
            except:
                font = ImageFont.load_default()
            
            d.text((50, 45), "SM", fill='white', font=font)
            return img
        except:
            # Return a blank image if loading fails
            return Image.new('RGB', (150, 150), color='#667eea')

# Load user data from Google Sheets
def load_user_data():
    """Load user information from Google Sheets"""
    try:
        sheet_id = "1EGbRgkMRJY7SOLGxgZ5j9H1XC6Sxv5AhXaynYP6dNss"
        csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'
        user_df = pd.read_csv(csv_url)
        return user_df
    except Exception as e:
        st.error(f"Error loading user data from Google Sheets: {e}")
        return None

# User registration system
class UserRegistration:
    def __init__(self):
        if 'users' not in st.session_state:
            # Try to load from Google Sheets first
            user_df = load_user_data()
            if user_df is not None:
                st.session_state.users = user_df.to_dict('records')
            else:
                st.session_state.users = []
        
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
    
    def register_user(self, name, contact, email, organization, position):
        user_data = {
            'name': name,
            'contact': contact,
            'email': email,
            'organization': organization,
            'position': position,
            'registration_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'usage_count': 0,
            'last_login': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Check if user already exists
        existing_user = next((u for u in st.session_state.users if u['email'] == email), None)
        if existing_user:
            existing_user['last_login'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            existing_user['usage_count'] += 1
            st.session_state.current_user = existing_user
        else:
            st.session_state.users.append(user_data)
            st.session_state.current_user = user_data
        
        return st.session_state.current_user
    
    def get_user_stats(self):
        if not st.session_state.users:
            return None
        
        stats = {
            'total_users': len(st.session_state.users),
            'total_usage': sum(user.get('usage_count', 0) for user in st.session_state.users),
            'avg_usage_per_user': sum(user.get('usage_count', 0) for user in st.session_state.users) / len(st.session_state.users),
            'organizations': list(set(user.get('organization', '') for user in st.session_state.users)),
            'positions': list(set(user.get('position', '') for user in st.session_state.users))
        }
        return stats

# Initialize user registration
user_reg = UserRegistration()

# Enhanced manufacturing data generation with multiple frequencies
def generate_manufacturing_data():
    np.random.seed(42)
    n = 1000  # Increased for better forecasting
    
    # Time series data with multiple frequency options
    dates_daily = pd.date_range('2023-01-01', periods=n, freq='D')
    dates_hourly = pd.date_range('2023-01-01', periods=n, freq='H')
    dates_weekly = pd.date_range('2023-01-01', periods=n, freq='W')
    
    # Use daily frequency as default
    dates = dates_daily
    
    # Create realistic manufacturing patterns with multiple trends
    base_length = 10.0
    trend = np.linspace(0, 0.5, n)  # Gradual upward trend
    seasonal_monthly = 0.1 * np.sin(2 * np.pi * np.arange(n) / 30)  # Monthly seasonality
    seasonal_weekly = 0.05 * np.sin(2 * np.pi * np.arange(n) / 7)   # Weekly seasonality
    noise = np.random.normal(0, 0.05, n)
    
    length_values = base_length + trend + seasonal_monthly + seasonal_weekly + noise
    
    data = {
        'date': dates,
        'part_id': range(1, n+1),
        'length': length_values,
        'diameter': np.random.normal(5.0, 0.08, n),
        'weight': np.random.normal(100.0, 2.5, n),
        'hardness': np.random.normal(45.0, 3.0, n),
        'temperature': np.random.normal(75.0, 5.0, n),
        'pressure': np.random.normal(100.0, 10.0, n),
        'defect': np.random.choice([0, 1], n, p=[0.92, 0.08]),
        'operator': np.random.choice(['Sourove', 'Nasif', 'Sadip', 'Rakib', 'Chandan'], n),
        'machine': np.random.choice(['CNC-1', 'CNC-2', 'LATHE-1', 'MILL-1'], n),
        'shift': np.random.choice(['Morning', 'Afternoon', 'Night'], n),
        'material_batch': np.random.choice(['A123', 'B456', 'C789', 'D012', 'E345'], n),
        'production_rate': np.random.normal(50, 5, n)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation for regression analysis
    df['quality_score'] = (df['length'] * 0.3 + df['hardness'] * 0.2 + 
                          df['temperature'] * 0.1 + np.random.normal(0, 0.5, n))
    
    return df

# Quality metrics calculation functions
def calculate_cp(upper_spec, lower_spec, std_dev):
    """Calculate Process Capability Index (Cp)"""
    if std_dev == 0:
        return float('inf')
    return (upper_spec - lower_spec) / (6 * std_dev)

def calculate_cpk(upper_spec, lower_spec, mean, std_dev):
    """Calculate Process Capability Index (Cpk)"""
    if std_dev == 0:
        return float('inf')
    cpu = (upper_spec - mean) / (3 * std_dev) if std_dev > 0 else float('inf')
    cpl = (mean - lower_spec) / (3 * std_dev) if std_dev > 0 else float('inf')
    return min(cpu, cpl)

def calculate_pp(upper_spec, lower_spec, std_dev):
    """Calculate Process Performance Index (Pp)"""
    return calculate_cp(upper_spec, lower_spec, std_dev)

def calculate_ppk(upper_spec, lower_spec, mean, std_dev):
    """Calculate Process Performance Index (Ppk)"""
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_cmk(upper_spec, lower_spec, mean, std_dev):
    """Calculate Machine Capability Index (Cmk)"""
    return calculate_cpk(upper_spec, lower_spec, mean, std_dev)

def calculate_dpmo(defect_count, total_units):
    """Calculate Defects Per Million Opportunities (DPMO)"""
    if total_units == 0:
        return 0
    return (defect_count / total_units) * 1000000

def calculate_sigma_level(dpmo):
    """Calculate Sigma Level from DPMO"""
    if dpmo <= 0:
        return float('inf')
    if not HAS_SCIPY:
        # Simple approximation without scipy
        if dpmo <= 3.4: return 6.0
        elif dpmo <= 233: return 5.0
        elif dpmo <= 6200: return 4.0
        elif dpmo <= 66800: return 3.0
        elif dpmo <= 308000: return 2.0
        else: return 1.0
    return stats.norm.ppf(1 - dpmo/1000000) + 1.5

# Enhanced Sampling recommendation functions with detailed information
def recommend_sampling_method(data_type, data_nature, application):
    """Recommend sampling method based on data characteristics"""
    recommendations = []
    
    if data_type == "Variable":
        recommendations.append("📏 **Variables Sampling**: Use measurement data for precise analysis")
        recommendations.append("✅ **Recommended Methods**: ")
        recommendations.append("   • **SPC Control Charts**: X-bar R charts (subgroups 4-5), X-bar S charts (subgroups >10)")
        recommendations.append("   • **Acceptance Sampling by Variables**: ANSI/ASQ Z1.9, MIL-STD-414")
        recommendations.append("   • **Process Capability Analysis**: Minimum 100 individual measurements")
        recommendations.append("   • **Measurement System Analysis**: Gage R&R studies")
    elif data_type == "Attribute":
        recommendations.append("🔢 **Attributes Sampling**: Use count data (pass/fail) for defect analysis")
        recommendations.append("✅ **Recommended Methods**: ")
        recommendations.append("   • **Acceptance Sampling by Attributes**: ANSI/ASQ Z1.4, MIL-STD-105E")
        recommendations.append("   • **Control Charts**: p-charts (proportion defective), np-charts (number defective)")
        recommendations.append("   • **Specialized Charts**: c-charts (defects per unit), u-charts (defects per unit variable sample size)")
        recommendations.append("   • **Lot Tolerance Percent Defective (LTPD)**: For high-reliability applications")
    
    if data_nature == "Continuous":
        recommendations.append("⏰ **Continuous Data**: Consider time-based sampling at regular intervals")
        recommendations.append("   • **Frequency**: Every hour, shift, or batch")
        recommendations.append("   • **Rational Subgrouping**: Group data by time, machine, or operator")
    elif data_nature == "Discrete":
        recommendations.append("📦 **Discrete Data**: Consider lot-based sampling or batch sampling")
        recommendations.append("   • **Lot Size**: Sample size based on ANSI/ASQ Z1.4 tables")
        recommendations.append("   • **AQL Levels**: Choose appropriate Acceptable Quality Levels")
    
    if "Normal" in data_nature:
        recommendations.append("📊 **Normal Distribution**: Parametric statistical methods can be used")
        recommendations.append("   • **Confidence Intervals**: Use t-distribution for means")
        recommendations.append("   • **Hypothesis Testing**: t-tests, ANOVA for group comparisons")
        recommendations.append("   • **Process Capability**: Cp, Cpk calculations are valid")
    elif "Non-normal" in data_nature:
        recommendations.append("📈 **Non-normal Distribution**: Use non-parametric methods or transform data")
        recommendations.append("   • **Transformations**: Box-Cox, Johnson transformations")
        recommendations.append("   • **Non-parametric Tests**: Mann-Whitney, Kruskal-Wallis")
        recommendations.append("   • **Distribution Fitting**: Weibull, Lognormal, Gamma distributions")
    
    if application == "Process Control":
        recommendations.append("🎯 **Process Control**: Use SPC control charts with regular sampling intervals")
        recommendations.append("📋 **Sample Size Guidelines**:")
        recommendations.append("   • **Initial Study**: 20-25 subgroups of 4-5 samples each")
        recommendations.append("   • **Ongoing Monitoring**: Reduced frequency once process is stable")
        recommendations.append("   • **Subgroup Rationale**: Capture within-subgroup and between-subgroup variation")
    elif application == "Lot Acceptance":
        recommendations.append("📋 **Lot Acceptance**: Use ANSI/ASQ Z1.4 (MIL-STD-105E) for attributes")
        recommendations.append("📏 **Variables Acceptance**: Use ANSI/ASQ Z1.9 (MIL-STD-414)")
        recommendations.append("🔍 **Sampling Plans**:")
        recommendations.append("   • **Single Sampling**: One sample decision")
        recommendations.append("   • **Double Sampling**: Second sample if first is inconclusive")
        recommendations.append("   • **Multiple Sampling**: Multiple stages for high-volume inspection")
    elif application == "Capability Analysis":
        recommendations.append("📐 **Capability Analysis**: Ensure random sampling, minimum 100 individual measurements")
        recommendations.append("⏰ **Timing**: Collect data over different time periods for reliable analysis")
        recommendations.append("📊 **Data Requirements**:")
        recommendations.append("   • **Stable Process**: Only analyze when process is in statistical control")
        recommendations.append("   • **Representative Data**: Include all normal process variation")
        recommendations.append("   • **Measurement Capability**: Ensure measurement system is adequate")
    elif application == "Defect Analysis":
        recommendations.append("🔍 **Defect Analysis**: Use stratified sampling by defect type/category")
        recommendations.append("📊 **Sample Strategy**: Focus on high-defect areas for detailed analysis")
        recommendations.append("🎯 **Approaches**:")
        recommendations.append("   • **Stratified Sampling**: Sample proportional to defect rates")
        recommendations.append("   • **Cluster Sampling**: Sample entire clusters when defects are clustered")
        recommendations.append("   • **Sequential Sampling**: Continue sampling until decision can be made")
    
    return recommendations

# Google Sheets integration
def load_from_google_sheets(url):
    """Load data from Google Sheets"""
    try:
        if 'docs.google.com/spreadsheets' in url:
            # Convert Google Sheets URL to CSV export
            if '/d/' in url:
                sheet_id = url.split('/d/')[1].split('/')[0]
            else:
                sheet_id = url.split('spreadsheets/d/')[1].split('/')[0]
            
            csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'
            return pd.read_csv(csv_url)
        return None
    except Exception as e:
        st.error(f"Error loading Google Sheets: {e}")
        return None

# Usage tracking
class UsageTracker:
    def __init__(self):
        if 'usage_count' not in st.session_state:
            st.session_state.usage_count = 0
        if 'user_sessions' not in st.session_state:
            st.session_state.user_sessions = set()
        if 'feature_usage' not in st.session_state:
            st.session_state.feature_usage = {}
    
    def track_usage(self, feature_name):
        st.session_state.usage_count += 1
        st.session_state.user_sessions.add(id(st))
        if feature_name in st.session_state.feature_usage:
            st.session_state.feature_usage[feature_name] += 1
        else:
            st.session_state.feature_usage[feature_name] = 1
    
    def get_stats(self):
        return {
            'total_uses': st.session_state.usage_count,
            'unique_sessions': len(st.session_state.user_sessions),
            'feature_usage': st.session_state.feature_usage
        }

# Initialize usage tracker
tracker = UsageTracker()

# Enhanced PyCaret Regression Function with prediction
def run_pycaret_regression(data, target, features, train_size=0.8):
    """Run PyCaret regression with proper setup and return predictions"""
    try:
        # Prepare data
        model_data = data[features + [target]].dropna()
        
        if len(model_data) < 10:
            return None, "Not enough data for model training. Need at least 10 complete records."
        
        # Setup PyCaret environment
        setup_result = setup(
            data=model_data,
            target=target,
            session_id=123,
            train_size=train_size,
            normalize=True,
            silent=True,
            verbose=False,
            fold=5
        )
        
        # Compare models
        best_model = compare_models()
        
        # Get comparison results
        comparison_results = pull()
        
        # Create future predictions
        future_predictions = None
        if len(model_data) > 100:
            # Generate future data for prediction
            future_dates = pd.date_range(start=model_data.index[-1] if hasattr(model_data.index, 'dtype') else pd.Timestamp.now(), 
                                       periods=100, freq='D')
            future_data = pd.DataFrame(index=range(len(model_data), len(model_data) + 100))
            
            # Create synthetic future features based on historical patterns
            for feature in features:
                if feature in model_data.columns:
                    historical_mean = model_data[feature].mean()
                    historical_std = model_data[feature].std()
                    future_data[feature] = np.random.normal(historical_mean, historical_std, 100)
            
            # Make predictions
            future_predictions = predict_model(best_model, data=future_data)
        
        return {
            'best_model': best_model,
            'model_name': type(best_model).__name__,
            'comparison_results': comparison_results,
            'setup_result': setup_result,
            'future_predictions': future_predictions
        }, None
        
    except Exception as e:
        return None, f"Error in model training: {e}"

# Time Series Analysis Function
def run_time_series_analysis(data, date_col, target_col, forecast_periods=100):
    """Run time series analysis and forecasting"""
    try:
        # Prepare time series data
        ts_data = data[[date_col, target_col]].copy()
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        ts_data = ts_data.set_index(date_col).sort_index()
        ts_data = ts_data.dropna()
        
        if len(ts_data) < 30:
            return None, "Need at least 30 data points for time series analysis"
        
        # Simple time series decomposition using statsmodels if available
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Resample to regular frequency if needed
            ts_resampled = ts_data[target_col].resample('D').mean().ffill()
            
            # Seasonal decomposition
            decomposition = seasonal_decompose(ts_resampled, period=30, model='additive')
            
            # Create future forecasts using simple methods
            last_value = ts_resampled.iloc[-1]
            trend = ts_resampled.diff().mean() if len(ts_resampled) > 1 else 0
            
            # Generate future predictions
            future_dates = pd.date_range(start=ts_resampled.index[-1], periods=forecast_periods + 1, freq='D')[1:]
            future_predictions = []
            
            current_value = last_value
            for i in range(forecast_periods):
                # Simple trend + seasonal pattern
                seasonal_component = 0.1 * np.sin(2 * np.pi * (len(ts_resampled) + i) / 30)
                current_value = current_value + trend + seasonal_component + np.random.normal(0, ts_resampled.std() * 0.1)
                future_predictions.append(current_value)
            
            future_df = pd.DataFrame({
                'date': future_dates,
                'prediction': future_predictions,
                'lower_bound': [p * 0.9 for p in future_predictions],
                'upper_bound': [p * 1.1 for p in future_predictions]
            })
            
            return {
                'decomposition': decomposition,
                'future_predictions': future_df,
                'historical_data': ts_resampled
            }, None
            
        except ImportError:
            # Fallback method without statsmodels
            future_dates = pd.date_range(start=ts_data.index[-1], periods=forecast_periods + 1, freq='D')[1:]
            future_predictions = [ts_data[target_col].iloc[-1]] * forecast_periods
            
            future_df = pd.DataFrame({
                'date': future_dates,
                'prediction': future_predictions,
                'lower_bound': [p * 0.9 for p in future_predictions],
                'upper_bound': [p * 1.1 for p in future_predictions]
            })
            
            return {
                'future_predictions': future_df,
                'historical_data': ts_data[target_col]
            }, "Statsmodels not available, using simple forecasting"
            
    except Exception as e:
        return None, f"Error in time series analysis: {e}"

# Seaborn-based visualization functions
def create_distribution_seaborn(data, column_name):
    """Create distribution plot using Seaborn"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[column_name].dropna(), kde=True, ax=ax, color='skyblue')
    ax.axvline(data[column_name].mean(), color='red', linestyle='--', label=f'Mean: {data[column_name].mean():.2f}')
    ax.set_title(f'Distribution of {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Frequency')
    ax.legend()
    return fig

def create_correlation_heatmap_seaborn(corr_data):
    """Create correlation heatmap using Seaborn"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, ax=ax, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Heatmap')
    return fig

def create_control_charts_seaborn(subgroup_means, subgroup_ranges, overall_mean, 
                               xbar_ucl, xbar_lcl, mean_range, r_ucl, r_lcl, quality_char):
    """Create control charts using Seaborn"""
    # X-bar chart
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(subgroup_means, 'bo-', linewidth=1, markersize=4, label='Subgroup Mean')
    ax1.axhline(overall_mean, color='green', linestyle='-', label=f'CL: {overall_mean:.3f}')
    ax1.axhline(xbar_ucl, color='red', linestyle='--', label=f'UCL: {xbar_ucl:.3f}')
    ax1.axhline(xbar_lcl, color='red', linestyle='--', label=f'LCL: {xbar_lcl:.3f}')
    ax1.set_title(f'X-bar Control Chart for {quality_char}')
    ax1.set_xlabel('Subgroup Number')
    ax1.set_ylabel('Subgroup Mean')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # R chart
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(subgroup_ranges, 'ro-', linewidth=1, markersize=4, label='Subgroup Range')
    ax2.axhline(mean_range, color='green', linestyle='-', label=f'CL: {mean_range:.3f}')
    ax2.axhline(r_ucl, color='red', linestyle='--', label=f'UCL: {r_ucl:.3f}')
    if r_lcl > 0:
        ax2.axhline(r_lcl, color='red', linestyle='--', label=f'LCL: {r_lcl:.3f}')
    ax2.set_title(f'R Control Chart for {quality_char}')
    ax2.set_xlabel('Subgroup Number')
    ax2.set_ylabel('Subgroup Range')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    return fig1, fig2

def create_pareto_chart_seaborn(defect_counts, total_defects, selected_cat):
    """Create Pareto chart using Seaborn"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar chart for frequencies
    bars = ax1.bar(range(len(defect_counts)), defect_counts.values, color='skyblue', alpha=0.7)
    ax1.set_xlabel(selected_cat)
    ax1.set_ylabel('Frequency', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Line chart for cumulative percentage
    cumulative_pct = (defect_counts.cumsum() / total_defects) * 100
    ax2 = ax1.twinx()
    ax2.plot(range(len(defect_counts)), cumulative_pct, 'ro-', linewidth=2, markersize=6)
    ax2.set_ylabel('Cumulative Percentage (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 110)
    
    # Set x-axis labels
    ax1.set_xticks(range(len(defect_counts)))
    ax1.set_xticklabels([str(x) for x in defect_counts.index], rotation=45)
    
    plt.title(f'Pareto Chart - {selected_cat}')
    fig.tight_layout()
    
    return fig

def create_normality_plot_seaborn(data, column_name):
    """Create comprehensive normality assessment plot using Seaborn"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram with normal curve
    hist_data = data[column_name].dropna()
    
    sns.histplot(hist_data, kde=True, ax=ax1, color='skyblue')
    ax1.set_title(f'Distribution - {column_name}')
    ax1.set_xlabel(column_name)
    ax1.set_ylabel('Density')
    
    # Q-Q Plot
    if HAS_SCIPY:
        stats.probplot(hist_data, dist="norm", plot=ax2)
        ax2.set_title(f'Q-Q Plot - {column_name}')
    else:
        ax2.text(0.5, 0.5, 'Q-Q Plot requires scipy', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes)
        ax2.set_title(f'Q-Q Plot - {column_name} (scipy not available)')
    
    plt.tight_layout()
    return fig

# Import matplotlib for Seaborn plots
import matplotlib.pyplot as plt

# Enhanced Main Application
def main():
    # User Registration Section - Show this first
    if st.session_state.current_user is None:
        st.markdown("""
        <div class="user-registration">
            <h2>🔐 User Registration</h2>
            <p>Please register to use the Advanced Quality Control Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email Address", placeholder="Enter your email")
            contact = st.text_input("Contact Number", placeholder="Enter your contact number")
        
        with col2:
            organization = st.text_input("Organization", placeholder="Enter your organization name")
            position = st.text_input("Position", placeholder="Enter your position")
        
        if st.button("Register & Continue", type="primary"):
            if name and email and contact and organization and position:
                user = user_reg.register_user(name, contact, email, organization, position)
                st.success(f"Welcome {user['name']}! Registration successful.")
                st.experimental_rerun()
            else:
                st.error("Please fill in all fields to continue.")
        
        st.stop()
    
    # Initialize session state for data
    if 'df' not in st.session_state:
        st.session_state.df = generate_manufacturing_data()
    
    # Profile Header with Logos
    st.markdown("""
    <div class="logo-container">
        <img src="https://cdn-icons-png.flaticon.com/512/2173/2173475.png" class="logo-img" alt="Quality Icon">
        <img src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png" class="logo-img" alt="Analytics Icon">
        <img src="https://cdn-icons-png.flaticon.com/512/2942/2942789.png" class="logo-img" alt="Manufacturing Icon">
        <img src="https://cdn-icons-png.flaticon.com/512/2933/2933245.png" class="logo-img" alt="AI Icon">
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        profile_image = load_profile_image()
        st.image(profile_image, width=150, caption="Md. Sourove Akther Momin")
    
    with col2:
        current_user = st.session_state.current_user
        st.markdown(f"""
        <div class="profile-header">
            <h1>🏭 Advanced Quality Control Dashboard</h1>
            <h3>About Me - Md. Sourove Akther Momin</h3>
            <p><strong>MSc. in Applied Statistics and Data Science | BSc. in Mechanical Engineering</strong></p>
            <div>
                <span class="certification-badge">Certified Metal Cutting Professional (CMP)</span>
                <span class="certification-badge">World Class Manufacturing Practices Manager (WCMPM)</span>
                <span class="certification-badge">Six Sigma Black Belt</span>
            </div>
            <p><strong>Expertise:</strong> Process Engineering, Quality Management, Production Optimization, 
            Time Series Forecasting, Machine Learning, Big Data Analytics, Deep Learning, Artificial Intelligence</p>
            <p><strong>Professional Affiliations:</strong> Member of Technical Committee and Reviewer for 
            2nd IEOM 2025 World Congress on Industrial Engineering and Operations Management</p>
        </div>
        """, unsafe_allow_html=True)
    
    # User Statistics
    user_stats = user_reg.get_user_stats()
    if user_stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", user_stats['total_users'])
        with col2:
            st.metric("Total Usage", user_stats['total_usage'])
        with col3:
            st.metric("Avg Usage/User", f"{user_stats['avg_usage_per_user']:.1f}")
        with col4:
            st.metric("Organizations", len(user_stats['organizations']))
    
    # Sidebar for navigation
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Dashboard Section",
        ["📊 Data Overview", "📈 Quality Metrics", "🔍 Statistical Analysis", 
         "🤖 ML & Forecasting", "📋 Sampling Advisor", "⚙️ Settings"]
    )
    
    # Data Upload Section
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Data Management")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your manufacturing data", 
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel files with your manufacturing data"
    )
    
    google_sheets_url = st.sidebar.text_input(
        "Or enter Google Sheets URL:",
        placeholder="https://docs.google.com/spreadsheets/d/..."
    )
    
    # Load data from uploaded file or Google Sheets
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.sidebar.success(f"Data loaded successfully! {len(st.session_state.df)} records")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    
    if google_sheets_url:
        gs_data = load_from_google_sheets(google_sheets_url)
        if gs_data is not None:
            st.session_state.df = gs_data
            st.sidebar.success(f"Google Sheets data loaded! {len(st.session_state.df)} records")
    
    # Data Overview Section
    if app_mode == "📊 Data Overview":
        tracker.track_usage("Data Overview")
        st.markdown('<div class="section-header">📊 Data Overview & Quality Metrics</div>', unsafe_allow_html=True)
        
        df = st.session_state.df
        
        # Basic Information
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data Preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column Information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Basic Statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Data Quality Assessment
        st.subheader("📋 Data Quality Assessment")
        
        quality_metrics = []
        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                completeness = (1 - df[col].isnull().sum() / len(df)) * 100
                cv = (col_data.std() / col_data.mean()) * 100 if col_data.mean() != 0 else float('inf')
                
                quality_metrics.append({
                    'Feature': col,
                    'Completeness (%)': f"{completeness:.1f}",
                    'Mean': f"{col_data.mean():.3f}",
                    'Std Dev': f"{col_data.std():.3f}",
                    'CV (%)': f"{cv:.1f}" if cv != float('inf') else "Inf",
                    'Outliers (%)': f"{(np.abs(stats.zscore(col_data)) > 3).sum() / len(col_data) * 100:.1f}" if HAS_SCIPY else "N/A"
                })
        
        if quality_metrics:
            quality_df = pd.DataFrame(quality_metrics)
            st.dataframe(quality_df, use_container_width=True)
        
        # Distribution Visualization
        st.subheader("📈 Distribution Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select feature for distribution analysis:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram with Seaborn
                fig = create_distribution_seaborn(df, selected_col)
                st.pyplot(fig)
            
            with col2:
                # Box plot with Seaborn
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(y=df[selected_col], ax=ax, color='lightblue')
                ax.set_title(f'Box Plot - {selected_col}')
                st.pyplot(fig)
    
    # Quality Metrics Section
    elif app_mode == "📈 Quality Metrics":
        tracker.track_usage("Quality Metrics")
        st.markdown('<div class="section-header">📈 Quality Control Metrics & Analysis</div>', unsafe_allow_html=True)
        
        df = st.session_state.df
        
        # Process Capability Analysis
        st.subheader("📊 Process Capability Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                quality_char = st.selectbox("Select Quality Characteristic:", numeric_cols)
            
            with col2:
                lower_spec = st.number_input("Lower Specification Limit (LSL):", 
                                           value=float(df[quality_char].mean() - 3*df[quality_char].std()))
            
            with col3:
                upper_spec = st.number_input("Upper Specification Limit (USL):", 
                                           value=float(df[quality_char].mean() + 3*df[quality_char].std()))
            
            if lower_spec < upper_spec:
                # Calculate capability indices
                data_series = df[quality_char].dropna()
                mean_val = data_series.mean()
                std_val = data_series.std()
                
                cp = calculate_cp(upper_spec, lower_spec, std_val)
                cpk = calculate_cpk(upper_spec, lower_spec, mean_val, std_val)
                pp = calculate_pp(upper_spec, lower_spec, std_val)
                ppk = calculate_ppk(upper_spec, lower_spec, mean_val, std_val)
                cmk = calculate_cmk(upper_spec, lower_spec, mean_val, std_val)
                
                # Display capability metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Cp", f"{cp:.3f}")
                with col2:
                    st.metric("Cpk", f"{cpk:.3f}")
                with col3:
                    st.metric("Pp", f"{pp:.3f}")
                with col4:
                    st.metric("Ppk", f"{ppk:.3f}")
                with col5:
                    st.metric("Cmk", f"{cmk:.3f}")
                
                # Capability interpretation
                st.subheader("📋 Process Capability Interpretation")
                
                capability_text = ""
                if cpk >= 1.67:
                    capability_text = "🎉 **Excellent** - Process is highly capable"
                elif cpk >= 1.33:
                    capability_text = "✅ **Good** - Process is capable"
                elif cpk >= 1.0:
                    capability_text = "⚠️ **Marginal** - Process requires close monitoring"
                else:
                    capability_text = "❌ **Poor** - Process is not capable"
                
                st.markdown(f"**Process Capability Status:** {capability_text}")
                
                # Display process distribution with specification limits
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Histogram
                sns.histplot(data_series, kde=True, ax=ax, color='skyblue', alpha=0.7)
                
                # Specification limits
                ax.axvline(lower_spec, color='red', linestyle='--', linewidth=2, label=f'LSL: {lower_spec:.3f}')
                ax.axvline(upper_spec, color='red', linestyle='--', linewidth=2, label=f'USL: {upper_spec:.3f}')
                ax.axvline(mean_val, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
                
                ax.set_title(f'Process Distribution with Specification Limits - {quality_char}')
                ax.set_xlabel(quality_char)
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
        
        # Six Sigma Metrics
        st.subheader("🎯 Six Sigma Metrics")
        
        if 'defect' in df.columns:
            defect_count = df['defect'].sum()
            total_units = len(df)
            dpmo = calculate_dpmo(defect_count, total_units)
            sigma_level = calculate_sigma_level(dpmo)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Defect Count", int(defect_count))
            with col2:
                st.metric("Total Units", total_units)
            with col3:
                st.metric("DPMO", f"{dpmo:,.0f}")
            with col4:
                st.metric("Sigma Level", f"{sigma_level:.2f}")
        
        # Control Charts
        st.subheader("📊 Control Charts")
        
        if numeric_cols:
            control_char = st.selectbox("Select characteristic for control chart:", numeric_cols)
            subgroup_size = st.slider("Subgroup Size:", min_value=2, max_value=10, value=5)
            
            # Generate subgroup data
            data_series = df[control_char].dropna()
            n_subgroups = len(data_series) // subgroup_size
            subgroup_means = []
            subgroup_ranges = []
            
            for i in range(n_subgroups):
                subgroup = data_series[i*subgroup_size:(i+1)*subgroup_size]
                subgroup_means.append(subgroup.mean())
                subgroup_ranges.append(subgroup.max() - subgroup.min())
            
            if subgroup_means:
                # Calculate control limits
                overall_mean = np.mean(subgroup_means)
                mean_range = np.mean(subgroup_ranges)
                
                # Constants for control charts
                A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 
                      6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
                D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
                D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 
                      6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
                
                xbar_ucl = overall_mean + A2[subgroup_size] * mean_range
                xbar_lcl = overall_mean - A2[subgroup_size] * mean_range
                r_ucl = D4[subgroup_size] * mean_range
                r_lcl = D3[subgroup_size] * mean_range
                
                # Create control charts
                fig1, fig2 = create_control_charts_seaborn(
                    subgroup_means, subgroup_ranges, overall_mean,
                    xbar_ucl, xbar_lcl, mean_range, r_ucl, r_lcl, control_char
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig1)
                with col2:
                    st.pyplot(fig2)
                
                # Control chart interpretation
                st.subheader("📋 Control Chart Interpretation")
                
                # Check for out-of-control points
                out_of_control_xbar = sum(1 for x in subgroup_means if x > xbar_ucl or x < xbar_lcl)
                out_of_control_r = sum(1 for r in subgroup_ranges if r > r_ucl or r < r_lcl)
                
                col1, col2 = st.columns(2)
                with col1:
                    if out_of_control_xbar == 0:
                        st.success("✅ X-bar Chart: Process appears to be in statistical control")
                    else:
                        st.warning(f"⚠️ X-bar Chart: {out_of_control_xbar} points outside control limits")
                
                with col2:
                    if out_of_control_r == 0:
                        st.success("✅ R Chart: Process variation appears stable")
                    else:
                        st.warning(f"⚠️ R Chart: {out_of_control_r} points outside control limits")
    
    # Statistical Analysis Section
    elif app_mode == "🔍 Statistical Analysis":
        tracker.track_usage("Statistical Analysis")
        st.markdown('<div class="section-header">🔍 Statistical Analysis & Hypothesis Testing</div>', unsafe_allow_html=True)
        
        df = st.session_state.df
        
        # Correlation Analysis
        st.subheader("📊 Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            fig = create_correlation_heatmap_seaborn(corr_matrix)
            st.pyplot(fig)
            
            # Detailed correlation table
            st.subheader("Detailed Correlation Table")
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1), 
                        use_container_width=True)
        
        # Normality Testing
        st.subheader("📈 Normality Assessment")
        
        if numeric_cols:
            normality_col = st.selectbox("Select feature for normality test:", numeric_cols)
            
            if st.button("Run Normality Analysis"):
                data_series = df[normality_col].dropna()
                
                if len(data_series) >= 3:
                    # Create normality plots
                    fig = create_normality_plot_seaborn(df, normality_col)
                    st.pyplot(fig)
                    
                    # Statistical tests if scipy is available
                    if HAS_SCIPY:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Shapiro-Wilk test
                            shapiro_stat, shapiro_p = stats.shapiro(data_series)
                            st.metric("Shapiro-Wilk Test", 
                                    f"p-value: {shapiro_p:.4f}",
                                    delta="Normal" if shapiro_p > 0.05 else "Non-normal",
                                    delta_color="normal" if shapiro_p > 0.05 else "inverse")
                        
                        with col2:
                            # Anderson-Darling test
                            anderson_result = stats.anderson(data_series, dist='norm')
                            st.metric("Anderson-Darling Test", 
                                    f"Statistic: {anderson_result.statistic:.4f}",
                                    delta="Normal" if anderson_result.statistic < anderson_result.critical_values[2] else "Non-normal",
                                    delta_color="normal" if anderson_result.statistic < anderson_result.critical_values[2] else "inverse")
                    
                    # Skewness and Kurtosis
                    skewness = data_series.skew()
                    kurtosis = data_series.kurtosis()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Skewness", f"{skewness:.3f}")
                    with col2:
                        st.metric("Kurtosis", f"{kurtosis:.3f}")
                    
                    # Interpretation
                    st.subheader("📋 Normality Interpretation")
                    
                    if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
                        st.success("✅ Data appears approximately normal")
                    elif abs(skewness) < 1 and abs(kurtosis) < 1:
                        st.warning("⚠️ Data shows moderate deviation from normality")
                    else:
                        st.error("❌ Data shows significant deviation from normality")
        
        # Pareto Analysis
        st.subheader("📊 Pareto Analysis")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            selected_cat = st.selectbox("Select categorical feature for Pareto analysis:", categorical_cols)
            
            if st.button("Generate Pareto Chart"):
                defect_counts = df[selected_cat].value_counts()
                total_defects = defect_counts.sum()
                
                fig = create_pareto_chart_seaborn(defect_counts, total_defects, selected_cat)
                st.pyplot(fig)
                
                # Display top categories
                st.subheader("Top Categories")
                top_categories = defect_counts.head(10)
                st.dataframe(top_categories, use_container_width=True)
    
    # Machine Learning & Forecasting Section
    elif app_mode == "🤖 ML & Forecasting":
        tracker.track_usage("ML & Forecasting")
        st.markdown('<div class="section-header">🤖 Machine Learning & Time Series Forecasting</div>', unsafe_allow_html=True)
        
        df = st.session_state.df
        
        if not HAS_PYCARET:
            st.warning("""
            ⚠️ **PyCaret is not available**  
            Machine learning features require PyCaret installation.  
            Install with: `pip install pycaret`
            """)
        
        # Regression Analysis
        st.subheader("📈 Regression Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2 and HAS_PYCARET:
            col1, col2 = st.columns(2)
            
            with col1:
                target_var = st.selectbox("Select Target Variable:", numeric_cols)
            
            with col2:
                feature_vars = st.multiselect("Select Feature Variables:", 
                                            [col for col in numeric_cols if col != target_var],
                                            default=[col for col in numeric_cols if col != target_var][:3])
            
            if target_var and feature_vars and st.button("Run Regression Analysis"):
                with st.spinner("Training machine learning models..."):
                    result, error = run_pycaret_regression(df, target_var, feature_vars)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"✅ Best Model: {result['model_name']}")
                        
                        # Display model comparison
                        st.subheader("Model Comparison Results")
                        st.dataframe(result['comparison_results'], use_container_width=True)
        
        # Time Series Forecasting
        st.subheader("⏰ Time Series Forecasting")
        
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if not date_cols:
            # Check for string columns that might be dates
            string_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in string_cols:
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    continue
        
        if date_cols and numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                date_var = st.selectbox("Select Date Column:", date_cols)
            
            with col2:
                ts_target = st.selectbox("Select Target Variable for Forecasting:", numeric_cols)
            
            forecast_periods = st.slider("Forecast Periods:", min_value=7, max_value=365, value=100)
            
            if st.button("Run Time Series Analysis"):
                with st.spinner("Analyzing time series patterns..."):
                    result, error = run_time_series_analysis(df, date_var, ts_target, forecast_periods)
                    
                    if error:
                        st.warning(error)
                    
                    if result:
                        # Display historical data and forecasts
                        st.subheader("Historical Data & Forecast")
                        
                        # Create time series plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot historical data
                        if hasattr(result['historical_data'], 'index'):
                            ax.plot(result['historical_data'].index, result['historical_data'].values, 
                                   label='Historical', color='blue', linewidth=2)
                        
                        # Plot forecasts
                        future_df = result['future_predictions']
                        ax.plot(future_df['date'], future_df['prediction'], 
                               label='Forecast', color='red', linewidth=2)
                        
                        # Plot confidence intervals
                        ax.fill_between(future_df['date'], future_df['lower_bound'], 
                                      future_df['upper_bound'], alpha=0.2, color='red',
                                      label='Confidence Interval')
                        
                        ax.set_title(f'Time Series Forecast - {ts_target}')
                        ax.set_xlabel('Date')
                        ax.set_ylabel(ts_target)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Display forecast data
                        st.subheader("Forecast Data")
                        st.dataframe(future_df, use_container_width=True)
    
    # Sampling Advisor Section
    elif app_mode == "📋 Sampling Advisor":
        tracker.track_usage("Sampling Advisor")
        st.markdown('<div class="section-header">📋 Intelligent Sampling Advisor</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 About the Sampling Advisor</h4>
            <p>This intelligent advisor recommends optimal sampling strategies based on your data characteristics, 
            process requirements, and quality objectives. Get expert guidance on sample sizes, methods, and 
            statistical approaches for your specific situation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sampling Configuration
        st.subheader("🔧 Sampling Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_type = st.selectbox(
                "Data Type:",
                ["Variable", "Attribute"],
                help="Variable: Continuous measurement data | Attribute: Discrete count data"
            )
            
            data_nature = st.selectbox(
                "Data Nature:",
                ["Continuous", "Discrete", "Normal Distribution", "Non-normal Distribution"],
                help="Characteristics of your data distribution"
            )
        
        with col2:
            application = st.selectbox(
                "Primary Application:",
                ["Process Control", "Lot Acceptance", "Capability Analysis", "Defect Analysis"],
                help="How will the sampling results be used?"
            )
            
            confidence_level = st.slider(
                "Confidence Level (%):",
                min_value=90,
                max_value=99,
                value=95,
                help="Statistical confidence level for sampling"
            )
        
        # Get recommendations
        if st.button("Get Sampling Recommendations", type="primary"):
            recommendations = recommend_sampling_method(data_type, data_nature, application)
            
            st.subheader("📋 Recommended Sampling Strategy")
            
            for rec in recommendations:
                st.markdown(rec)
            
            # Additional statistical guidance
            st.subheader("📊 Statistical Considerations")
            
            if data_type == "Variable":
                st.markdown("""
                **Sample Size Guidelines for Variables Data:**
                - **Initial Capability Study**: 100+ individual measurements
                - **Control Charts**: 20-25 subgroups of 4-5 samples
                - **Hypothesis Testing**: Depends on effect size and power
                - **Measurement Studies**: 10 parts, 3 operators, 2-3 repetitions
                """)
            else:
                st.markdown("""
                **Sample Size Guidelines for Attributes Data:**
                - **Acceptance Sampling**: Use ANSI/ASQ Z1.4 tables based on AQL
                - **Defect Rate Estimation**: Sample size depends on expected defect rate
                - **Reliability Testing**: Larger samples for low failure rates
                - **Attribute Control Charts**: Minimum 25 subgroups
                """)
            
            # Sample size calculator
            st.subheader("🧮 Sample Size Calculator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                population_size = st.number_input("Population Size (if known):", 
                                                min_value=1, value=1000)
                margin_error = st.slider("Margin of Error (%):", 
                                       min_value=1, max_value=10, value=5)
            
            with col2:
                expected_proportion = st.slider("Expected Proportion (%):", 
                                              min_value=1, max_value=50, value=50)
                display_calc = st.checkbox("Show sample size calculation")
            
            # Sample size calculation
            z_score = {90: 1.645, 95: 1.96, 99: 2.576}.get(confidence_level, 1.96)
            p = expected_proportion / 100
            e = margin_error / 100
            
            n_infinite = (z_score**2 * p * (1-p)) / (e**2)
            n_finite = n_infinite / (1 + (n_infinite - 1) / population_size)
            
            st.metric("Recommended Sample Size", f"{math.ceil(n_finite):,}")
            
            if display_calc:
                st.info(f"""
                **Calculation Details:**
                - Z-score for {confidence_level}% confidence: {z_score}
                - Expected proportion: {expected_proportion}%
                - Margin of error: ±{margin_error}%
                - Finite population correction applied
                """)
    
    # Settings Section
    elif app_mode == "⚙️ Settings":
        tracker.track_usage("Settings")
        st.markdown('<div class="section-header">⚙️ System Settings & Information</div>', unsafe_allow_html=True)
        
        # System Information
        st.subheader("🖥️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Python Version:** 3.11
            **Streamlit Version:** {st.__version__}
            **Pandas Version:** {pd.__version__}
            **NumPy Version:** {np.__version__}
            **Seaborn Version:** {sns.__version__}
            """)
        
        with col2:
            st.info(f"""
            **PyCaret Available:** {HAS_PYCARET}
            **SciPy Available:** {HAS_SCIPY}
            **Current User:** {st.session_state.current_user['name']}
            **Organization:** {st.session_state.current_user['organization']}
            """)
        
        # Usage Statistics
        st.subheader("📊 Usage Statistics")
        
        usage_stats = tracker.get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Uses", usage_stats['total_uses'])
        with col2:
            st.metric("Unique Sessions", usage_stats['unique_sessions'])
        with col3:
            st.metric("Current Session", "Active")
        
        # Feature Usage
        st.subheader("🎯 Feature Usage")
        
        if usage_stats['feature_usage']:
            feature_usage_df = pd.DataFrame({
                'Feature': list(usage_stats['feature_usage'].keys()),
                'Usage Count': list(usage_stats['feature_usage'].values())
            }).sort_values('Usage Count', ascending=False)
            
            st.dataframe(feature_usage_df, use_container_width=True)
        
        # Data Management
        st.subheader("🗃️ Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate New Sample Data"):
                st.session_state.df = generate_manufacturing_data()
                st.success("New sample data generated!")
        
        with col2:
            if st.button("Clear All Data"):
                st.session_state.df = generate_manufacturing_data()
                st.success("Data cleared and reset to sample data!")
        
        # Export Data
        st.subheader("📤 Data Export")
        
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="quality_control_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create Excel file in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Quality_Data')
            excel_data = output.getvalue()
            
            st.download_button(
                label="Download Data as Excel",
                data=excel_data,
                file_name="quality_control_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # About Me Section (always visible at bottom)
    st.markdown("---")
    st.markdown("""
    <div class="about-me">
        <h3>👨‍💻 About the Developer - Md. Sourove Akther Momin</h3>
        <p><strong>Dual Expertise:</strong> Mechanical Engineering + Applied Statistics & Data Science</p>
        <p><strong>Mission:</strong> Bridging the gap between traditional manufacturing and modern data science 
        to create intelligent quality control solutions that drive operational excellence.</p>
        <p><strong>Specializations:</strong> Process Optimization, Predictive Maintenance, Quality 4.0, 
        Industrial AI Applications, Statistical Process Control, Machine Learning in Manufacturing</p>
        <p><strong>Connect:</strong> Available for consulting projects, research collaborations, and 
        advanced manufacturing analytics solutions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()