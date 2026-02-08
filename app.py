import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configure Streamlit application settings and layout parameters
st.set_page_config(
    page_title="Customer Churn Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling for dashboard theme and component aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        color: #a8b2d1;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2746 0%, #2a3356 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .metric-label {
        color: #8892b0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin: 0.3rem 0;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    
    .metric-delta.positive {
        color: #50fa7b;
    }
    
    .metric-delta.negative {
        color: #ff5555;
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    
    /* Streamlit Components Override */
    .stMetric {
        background: linear-gradient(135deg, #1e2746 0%, #2a3356 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .stMetric label {
        color: #8892b0 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stDataFrame"] {
        background: #1e2746;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 39, 70, 0.6);
        border-radius: 12px 12px 0 0;
        padding: 12px 24px;
        color: #8892b0;
        font-weight: 600;
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1e2746 0%, #2a3356 100%);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        color: #a8b2d1;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #50fa7b 0%, #2ecc71 100%);
        color: #0a0e27;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(80, 250, 123, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(80, 250, 123, 0.5);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #a8b2d1;
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stSlider > div > div > div {
        background: rgba(30, 39, 70, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        color: #a8b2d1;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background: rgba(30, 39, 70, 0.8);
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    
    /* Custom badge styles */
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    
    .risk-high {
        background: rgba(255, 85, 85, 0.2);
        color: #ff5555;
        border: 1px solid #ff5555;
    }
    
    .risk-medium {
        background: rgba(255, 184, 108, 0.2);
        color: #ffb86c;
        border: 1px solid #ffb86c;
    }
    
    .risk-low {
        background: rgba(80, 250, 123, 0.2);
        color: #50fa7b;
        border: 1px solid #50fa7b;
    }
    
    /* Insights box */
    .insight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #a8b2d1;
    }
    
    .insight-box h3 {
        color: #667eea;
        margin-top: 0;
    }
    
    .insight-item {
        margin: 0.8rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .insight-item:before {
        content: "▸";
        position: absolute;
        left: 0;
        color: #667eea;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Render primary dashboard header with branding and context
st.markdown("""
<div class='main-header'>
    <h1> Customer Churn Analytics Platform</h1>
    <p>Enterprise-Grade Predictive Analytics • Risk Segmentation • Revenue Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Load customer churn dataset with caching for performance optimization
@st.cache_data
def load_data():
    return pd.read_csv("Churn_Modelling.csv")

try:
    df = load_data().copy()
except FileNotFoundError:
    st.error("""
    **Dataset Not Found**
    
    Please ensure `Churn_Modelling.csv` is in the same directory as this script.
    """)
    st.stop()

# Verify dataset contains required columns for analysis pipeline
required_cols = [
    "CustomerId","Geography","Tenure","Balance",
    "NumOfProducts","IsActiveMember","Exited"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"""
    **Dataset Schema Validation Failed**
    
    Missing columns: `{', '.join(missing)}`
    
    Please upload a valid churn modeling dataset.
    """)
    st.stop()

# Standardize column names for consistency across analysis modules
df = df.rename(columns={
    "Exited":"Churn",
    "Geography":"Country",
    "NumOfProducts":"Products",
    "HasCrCard":"CreditCard",
    "IsActiveMember":"ActiveMember"
})

# Generate derived customer attributes for enhanced risk segmentation
def tenure_group(x):
    if x <= 2: return "New (0–2 yrs)"
    elif x <= 5: return "Mid (3–5 yrs)"
    return "Loyal (6+ yrs)"

df["TenureGroup"] = df["Tenure"].apply(tenure_group)

p75, p50 = df["Balance"].quantile([0.75, 0.50])

def value_segment(x):
    if x >= p75: return "High Value"
    elif x >= p50: return "Mid Value"
    return "Low Value"

df["CustomerValue"] = df["Balance"].apply(value_segment)
df["Engagement"] = df["ActiveMember"].map({1:"Engaged", 0:"Low Engagement"})

# Build predictive model using gradient boosting for churn classification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from xgboost import XGBClassifier

ml_df = df.copy()

label_cols = ["Country","Gender","CustomerValue","TenureGroup","Engagement"]
encoders = {}

for col in label_cols:
    le = LabelEncoder()
    ml_df[col] = le.fit_transform(ml_df[col])
    encoders[col] = le

features = [
    "CreditScore","Age","Tenure","Balance","EstimatedSalary",
    "Products","ActiveMember","Country","Gender",
    "CustomerValue","Engagement"
]

X = ml_df[features]
y = ml_df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Configure and train XGBoost classifier with optimal hyperparameters
model = XGBClassifier(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

model_auc = roc_auc_score(y_test, y_prob)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

# Extract feature importance scores to identify primary churn drivers
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Configure business parameters and display model diagnostics in sidebar
with st.sidebar:
    st.markdown("### Business Configuration")
    
    monthly_yield = st.slider(
        "Monthly Revenue Yield (%)",
        min_value=0.5, 
        max_value=5.0, 
        value=2.0,
        step=0.1,
        help="Estimated monthly revenue as % of account balance"
    ) / 100
    
    retention_cost = st.slider(
        "Retention Cost per Customer (₹)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Average cost to retain a customer"
    )
    
    st.markdown("---")
    st.markdown("### ML Model Performance")
    st.metric("ROC-AUC Score", f"{model_auc:.3f}")
    st.metric("Precision", f"{precision:.3f}")
    st.metric("Recall", f"{recall:.3f}")
    st.metric("F1-Score", f"{f1:.3f}")

# Calculate customer lifetime value metrics based on balance and tenure
df["EstimatedMonthlyValue"] = df["Balance"] * monthly_yield
df["EstimatedCLV"] = df["EstimatedMonthlyValue"] * (df["Tenure"] + 1)

# Display executive-level KPIs for portfolio health assessment
total_customers = len(df)
churned = df[df["Churn"]==1].shape[0]
active = total_customers - churned
overall_churn_rate = round((churned/total_customers)*100, 2)

st.markdown("<div class='section-header'> Executive Dashboard</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}",
        help="Total number of customers in portfolio"
    )

with col2:
    st.metric(
        "Active Customers",
        f"{active:,}",
        delta=f"{round((active/total_customers)*100, 1)}%",
        help="Customers currently retained"
    )

with col3:
    st.metric(
        "Churned Customers",
        f"{churned:,}",
        delta=f"-{overall_churn_rate}%",
        delta_color="inverse",
        help="Customers lost to churn"
    )

with col4:
    st.metric(
        "Churn Rate",
        f"{overall_churn_rate}%",
        delta=f"{round(overall_churn_rate - 20, 1)}% vs Industry",
        delta_color="inverse",
        help="Overall customer churn percentage"
    )

# Quantify financial impact of customer attrition on revenue streams
st.markdown("<div class='section-header'>Financial Impact Analysis</div>", unsafe_allow_html=True)

total_clv = df["EstimatedCLV"].sum()
lost_clv = df[df["Churn"]==1]["EstimatedCLV"].sum()
retained_clv = df[df["Churn"]==0]["EstimatedCLV"].sum()
high_value_at_risk = df[(df["CustomerValue"]=="High Value") & (df["Churn"]==1)]["EstimatedCLV"].sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Portfolio Value",
        f"₹{total_clv/1e6:.2f}M",
        help="Total customer lifetime value"
    )

with col2:
    st.metric(
        "Revenue Lost to Churn",
        f"₹{lost_clv/1e6:.2f}M",
        delta=f"-{(lost_clv/total_clv)*100:.1f}%",
        delta_color="inverse",
        help="CLV lost due to customer churn"
    )

with col3:
    st.metric(
        "High-Value Churn Loss",
        f"₹{high_value_at_risk/1e6:.2f}M",
        delta=f"-{(high_value_at_risk/lost_clv)*100:.1f}% of total loss",
        delta_color="inverse",
        help="Revenue lost from high-value customer segment"
    )

with col4:
    potential_savings = lost_clv * 0.3  # Assume 30% reduction
    roi = (potential_savings - (churned * retention_cost)) / (churned * retention_cost)
    st.metric(
        "Potential Savings (30%)",
        f"₹{potential_savings/1e6:.2f}M",
        delta=f"ROI: {roi:.1f}x",
        help="Estimated savings with 30% churn reduction"
    )

# Provide interactive controls for segment-level data exploration
st.markdown("<div class='section-header'>Advanced Filters & Segmentation</div>", unsafe_allow_html=True)

with st.expander("Apply Filters", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        country_filter = st.multiselect(
            "Country",
            options=df["Country"].unique(),
            default=df["Country"].unique(),
            help="Filter by geographic region"
        )
    
    with col2:
        value_filter = st.selectbox(
            "Customer Value",
            options=["All", "High Value", "Mid Value", "Low Value"],
            help="Filter by customer value segment"
        )
    
    with col3:
        engagement_filter = st.selectbox(
            "Engagement",
            options=["All", "Engaged", "Low Engagement"],
            help="Filter by customer engagement level"
        )
    
    with col4:
        tenure_filter = st.selectbox(
            "Tenure",
            options=["All", "New (0–2 yrs)", "Mid (3–5 yrs)", "Loyal (6+ yrs)"],
            help="Filter by customer tenure group"
        )

# Process user selections and filter dataset to target segment
filtered_df = df[df["Country"].isin(country_filter)].copy()

if value_filter != "All":
    filtered_df = filtered_df[filtered_df["CustomerValue"] == value_filter]

if engagement_filter != "All":
    filtered_df = filtered_df[filtered_df["Engagement"] == engagement_filter]

if tenure_filter != "All":
    filtered_df = filtered_df[filtered_df["TenureGroup"] == tenure_filter]

segment_churn = round(filtered_df["Churn"].mean() * 100, 2) if len(filtered_df) > 0 else 0
delta = round(segment_churn - overall_churn_rate, 2)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.metric(
        "Segment Churn Rate",
        f"{segment_churn}%",
        delta=f"{delta}% vs Overall",
        delta_color="inverse",
        help="Churn rate for selected segment"
    )

with col2:
    st.metric(
        "Segment Size",
        f"{len(filtered_df):,}",
        help="Number of customers in segment"
    )

with col3:
    segment_revenue = filtered_df["EstimatedCLV"].sum()
    st.metric(
        "Segment Value",
        f"₹{segment_revenue/1e6:.2f}M",
        help="Total CLV of segment"
    )

# Classify customers into risk tiers based on behavioral patterns
st.markdown("<div class='section-header'>🚦 Customer Risk Intelligence</div>", unsafe_allow_html=True)

def risk_group(row):
    if row["CustomerValue"]=="High Value" and row["Engagement"]=="Low Engagement":
        return "Critical: High Value Low Engagement"
    elif row["TenureGroup"]=="New (0–2 yrs)" and row["Engagement"]=="Low Engagement":
        return "High: New & Disengaged"
    elif row["CustomerValue"]=="High Value" and row["TenureGroup"]=="New (0–2 yrs)":
        return "Medium: High Value Onboarding"
    elif row["Engagement"]=="Low Engagement":
        return "Medium: Disengaged"
    else:
        return "Low: Stable Customer"

filtered_df["RiskGroup"] = filtered_df.apply(risk_group, axis=1)

priority_map = {
    "Critical: High Value Low Engagement": 1,
    "High: New & Disengaged": 2,
    "Medium: High Value Onboarding": 3,
    "Medium: Disengaged": 4,
    "Low: Stable Customer": 5
}

filtered_df["RiskPriority"] = filtered_df["RiskGroup"].map(priority_map)
filtered_df["ChurnStatus"] = filtered_df["Churn"].map({1:"Churned", 0:"Active"})

# Aggregate risk metrics across customer segments for comparison
risk_dist = filtered_df.groupby("RiskGroup").agg({
    "CustomerId": "count",
    "EstimatedCLV": "sum",
    "Churn": "mean"
}).reset_index()
risk_dist.columns = ["Risk Group", "Customer Count", "Total CLV", "Churn Rate"]
risk_dist["Churn Rate"] = (risk_dist["Churn Rate"] * 100).round(2)
risk_dist = risk_dist.sort_values("Total CLV", ascending=False)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("##### Risk Distribution by Segment")
    st.dataframe(
        risk_dist.style.format({
            "Customer Count": "{:,.0f}",
            "Total CLV": "₹{:,.0f}",
            "Churn Rate": "{:.2f}%"
        }),
        use_container_width=True,
        height=250
    )

with col2:
    st.markdown("##### Risk Priority Breakdown")
    fig_risk = px.pie(
        risk_dist,
        values="Customer Count",
        names="Risk Group",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Plasma_r
    )
    fig_risk.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=11
    )
    fig_risk.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a8b2d1', size=12),
        height=250,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False
    )
    st.plotly_chart(fig_risk, use_container_width=True)

# Isolate customers requiring immediate retention intervention
st.markdown("##### High-Priority Customers Requiring Immediate Action")

high_risk_df = filtered_df[
    filtered_df["RiskPriority"].isin([1, 2])
].sort_values("RiskPriority")

if len(high_risk_df) > 0:
    display_cols = [
        "CustomerId", "Country", "Age", "Tenure", "TenureGroup",
        "Balance", "CustomerValue", "Engagement",
        "RiskGroup", "ChurnStatus", "EstimatedCLV"
    ]
    
    st.dataframe(
        high_risk_df[display_cols].head(20),
        use_container_width=True,
        height=300
    )
    
    st.download_button(
        "⬇ Download High-Risk Customer Report",
        high_risk_df[display_cols].to_csv(index=False),
        "high_risk_customers.csv",
        "text/csv"
    )
else:
    st.info("No high-risk customers in the selected segment")

# Render comprehensive visual analytics across multiple dimensions
st.markdown("<div class='section-header'>Churn Analytics & Insights</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Geographic Analysis",
    "Customer Demographics",
    "Behavioral Patterns",
    "ML Insights"
])

# Define cohesive color palette for visual consistency
colors_gradient = ['#667eea', '#764ba2', '#f093fb', '#4facfe']

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Analyze geographic patterns in customer retention rates
        churn_country = filtered_df.groupby("Country").agg({
            "Churn": "mean",
            "CustomerId": "count",
            "EstimatedCLV": "sum"
        }).reset_index()
        churn_country.columns = ["Country", "Churn Rate", "Customer Count", "Total CLV"]
        churn_country["Churn Rate"] = churn_country["Churn Rate"] * 100
        
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=churn_country["Country"],
            y=churn_country["Churn Rate"],
            text=churn_country["Churn Rate"].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            marker=dict(
                color=churn_country["Churn Rate"],
                colorscale='Plasma',
                showscale=False,
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
        ))
        
        fig1.update_layout(
            title="Churn Rate by Country",
            xaxis_title="Country",
            yaxis_title="Churn Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a8b2d1'),
            height=400,
            yaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)'),
            margin=dict(t=60, b=60, l=60, r=40)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Compare revenue concentration across geographic markets
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=churn_country["Country"],
            y=churn_country["Total CLV"] / 1e6,
            text=(churn_country["Total CLV"] / 1e6).round(2),
            texttemplate='₹%{text}M',
            textposition='outside',
            marker=dict(
                color=colors_gradient[1],
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            hovertemplate='<b>%{x}</b><br>Total CLV: ₹%{y:.2f}M<extra></extra>'
        ))
        
        fig2.update_layout(
            title="Customer Lifetime Value by Country",
            xaxis_title="Country",
            yaxis_title="Total CLV (₹ Million)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a8b2d1'),
            height=400,
            yaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)'),
            margin=dict(t=60, b=60, l=60, r=40)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Visualize churn intensity across country-value segment combinations
    st.markdown("##### Geographic Performance Matrix")
    country_matrix = filtered_df.groupby(["Country", "CustomerValue"]).agg({
        "Churn": "mean",
        "CustomerId": "count"
    }).reset_index()
    country_matrix.columns = ["Country", "Value Segment", "Churn Rate", "Count"]
    
    pivot_churn = country_matrix.pivot(index="Value Segment", columns="Country", values="Churn Rate")
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot_churn.values * 100,
        x=pivot_churn.columns,
        y=pivot_churn.index,
        colorscale='RdYlGn_r',
        text=np.round(pivot_churn.values * 100, 1),
        texttemplate='%{text}%',
        textfont={"size": 14},
        colorbar=dict(title="Churn %", ticksuffix="%")
    ))
    
    fig_heat.update_layout(
        title="Churn Rate Heatmap: Country × Customer Value",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a8b2d1'),
        height=350,
        margin=dict(t=60, b=60, l=60, r=40)
    )
    
    st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Examine churn vulnerability across customer age cohorts
        age_bins = [0, 30, 40, 50, 60, 100]
        age_labels = ['<30', '30-40', '40-50', '50-60', '60+']
        filtered_df['AgeGroup'] = pd.cut(filtered_df['Age'], bins=age_bins, labels=age_labels)
        
        age_churn = filtered_df.groupby('AgeGroup')["Churn"].mean().reset_index()
        age_churn["Churn"] = age_churn["Churn"] * 100
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=age_churn['AgeGroup'],
            y=age_churn['Churn'],
            mode='lines+markers',
            line=dict(color=colors_gradient[0], width=3),
            marker=dict(size=12, color=colors_gradient[1], line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)',
            hovertemplate='<b>Age: %{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
        ))
        
        fig3.update_layout(
            title="Churn Rate by Age Group",
            xaxis_title="Age Group",
            yaxis_title="Churn Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a8b2d1'),
            height=400,
            yaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)'),
            xaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)'),
            margin=dict(t=60, b=60, l=60, r=40)
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Compare retention performance between gender segments
        gender_data = filtered_df.groupby("Gender").agg({
            "Churn": "mean",
            "CustomerId": "count"
        }).reset_index()
        gender_data.columns = ["Gender", "Churn Rate", "Count"]
        gender_data["Churn Rate"] = gender_data["Churn Rate"] * 100
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Bar(
            name='Churn Rate',
            x=gender_data['Gender'],
            y=gender_data['Churn Rate'],
            text=gender_data['Churn Rate'].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            marker=dict(color=colors_gradient[2]),
            yaxis='y',
            offsetgroup=1
        ))
        
        fig4.add_trace(go.Bar(
            name='Customer Count',
            x=gender_data['Gender'],
            y=gender_data['Count'],
            text=gender_data['Count'],
            texttemplate='%{text}',
            textposition='outside',
            marker=dict(color=colors_gradient[3]),
            yaxis='y2',
            offsetgroup=2
        ))
        
        fig4.update_layout(
            title="Churn Analysis by Gender",
            xaxis=dict(title='Gender'),
            yaxis=dict(title='Churn Rate (%)', side='left', gridcolor='rgba(102, 126, 234, 0.1)'),
            yaxis2=dict(title='Customer Count', side='right', overlaying='y', showgrid=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a8b2d1'),
            height=400,
            barmode='group',
            legend=dict(x=0.7, y=1.1, orientation='h'),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    # Track customer lifecycle patterns and tenure-based attrition trends
    st.markdown("##### Customer Lifecycle: Churn Trend by Tenure")
    
    tenure_trend = filtered_df.groupby("Tenure").agg({
        "Churn": "mean",
        "CustomerId": "count"
    }).reset_index()
    tenure_trend.columns = ["Tenure", "Churn Rate", "Customer Count"]
    tenure_trend["Churn Rate"] = tenure_trend["Churn Rate"] * 100
    
    fig5 = go.Figure()
    
    fig5.add_trace(go.Scatter(
        x=tenure_trend["Tenure"],
        y=tenure_trend["Churn Rate"],
        mode='lines+markers',
        name='Churn Rate',
        line=dict(color='#ff5555', width=3),
        marker=dict(size=8, color='#ff5555', line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor='rgba(255, 85, 85, 0.1)',
        yaxis='y',
        hovertemplate='<b>Tenure: %{x} years</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
    ))
    
    fig5.add_trace(go.Bar(
        x=tenure_trend["Tenure"],
        y=tenure_trend["Customer Count"],
        name='Customer Count',
        marker=dict(color='rgba(102, 126, 234, 0.3)', line=dict(color='#667eea', width=1)),
        yaxis='y2',
        hovertemplate='<b>Tenure: %{x} years</b><br>Customers: %{y}<extra></extra>'
    ))
    
    fig5.update_layout(
        title="Churn Rate & Customer Distribution Across Tenure",
        xaxis=dict(title='Tenure (Years)', gridcolor='rgba(102, 126, 234, 0.1)'),
        yaxis=dict(title='Churn Rate (%)', side='left', gridcolor='rgba(102, 126, 234, 0.1)'),
        yaxis2=dict(title='Customer Count', side='right', overlaying='y', showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a8b2d1'),
        height=450,
        legend=dict(x=0.7, y=1.08, orientation='h'),
        margin=dict(t=80, b=60, l=60, r=60),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig5, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Assess churn risk across customer value tiers
        value_churn = filtered_df.groupby("CustomerValue").agg({
            "Churn": "mean",
            "EstimatedCLV": "sum",
            "CustomerId": "count"
        }).reset_index()
        value_churn.columns = ["Customer Value", "Churn Rate", "Total CLV", "Count"]
        value_churn["Churn Rate"] = value_churn["Churn Rate"] * 100
        
        value_order = ["Low Value", "Mid Value", "High Value"]
        value_churn["Customer Value"] = pd.Categorical(
            value_churn["Customer Value"],
            categories=value_order,
            ordered=True
        )
        value_churn = value_churn.sort_values("Customer Value")
        
        fig6 = go.Figure()
        
        fig6.add_trace(go.Bar(
            x=value_churn["Customer Value"],
            y=value_churn["Churn Rate"],
            text=value_churn["Churn Rate"].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            marker=dict(
                color=value_churn["Churn Rate"],
                colorscale='Reds',
                showscale=False,
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            name='Churn Rate',
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
        ))
        
        fig6.update_layout(
            title="Churn Rate by Customer Value Segment",
            xaxis_title="Customer Value",
            yaxis_title="Churn Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a8b2d1'),
            height=400,
            yaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)'),
            margin=dict(t=60, b=60, l=60, r=40)
        )
        
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        # Measure impact of customer engagement on retention outcomes
        engagement_churn = filtered_df.groupby("Engagement").agg({
            "Churn": "mean",
            "CustomerId": "count"
        }).reset_index()
        engagement_churn.columns = ["Engagement", "Churn Rate", "Count"]
        engagement_churn["Churn Rate"] = engagement_churn["Churn Rate"] * 100
        
        fig7 = go.Figure()
        
        fig7.add_trace(go.Bar(
            x=engagement_churn["Engagement"],
            y=engagement_churn["Churn Rate"],
            text=engagement_churn["Churn Rate"].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            marker=dict(
                color=['#50fa7b', '#ff5555'],
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
        ))
        
        fig7.update_layout(
            title="Churn Rate by Engagement Level",
            xaxis_title="Engagement Status",
            yaxis_title="Churn Rate (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a8b2d1'),
            height=400,
            yaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)'),
            margin=dict(t=60, b=60, l=60, r=40)
        )
        
        st.plotly_chart(fig7, use_container_width=True)
    
    # Evaluate relationship between product portfolio and customer loyalty
    st.markdown("##### Product Holding Patterns")
    
    products_churn = filtered_df.groupby("Products").agg({
        "Churn": "mean",
        "CustomerId": "count",
        "EstimatedCLV": "mean"
    }).reset_index()
    products_churn.columns = ["Products", "Churn Rate", "Customer Count", "Avg CLV"]
    products_churn["Churn Rate"] = products_churn["Churn Rate"] * 100
    
    fig8 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Churn Rate by Product Count', 'Average CLV by Product Count'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig8.add_trace(
        go.Bar(
            x=products_churn["Products"],
            y=products_churn["Churn Rate"],
            text=products_churn["Churn Rate"].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            marker=dict(color=colors_gradient[0]),
            name='Churn Rate',
            hovertemplate='<b>Products: %{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig8.add_trace(
        go.Bar(
            x=products_churn["Products"],
            y=products_churn["Avg CLV"],
            text=(products_churn["Avg CLV"]/1000).round(1),
            texttemplate='₹%{text}K',
            textposition='outside',
            marker=dict(color=colors_gradient[1]),
            name='Avg CLV',
            hovertemplate='<b>Products: %{x}</b><br>Avg CLV: ₹%{y:,.0f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig8.update_xaxes(title_text="Number of Products", row=1, col=1)
    fig8.update_xaxes(title_text="Number of Products", row=1, col=2)
    fig8.update_yaxes(title_text="Churn Rate (%)", row=1, col=1)
    fig8.update_yaxes(title_text="Average CLV (₹)", row=1, col=2)
    
    fig8.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a8b2d1'),
        height=400,
        showlegend=False,
        margin=dict(t=80, b=60, l=60, r=40)
    )
    
    fig8.update_yaxes(gridcolor='rgba(102, 126, 234, 0.1)')
    
    st.plotly_chart(fig8, use_container_width=True)

with tab4:
    st.markdown("##### Machine Learning Model Insights")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Feature Importance Analysis**")
        
        top_features = feature_importance.head(10)
        
        fig9 = go.Figure()
        
        fig9.add_trace(go.Bar(
            y=top_features['feature'],
            x=top_features['importance'],
            orientation='h',
            text=top_features['importance'].round(3),
            texttemplate='%{text}',
            textposition='outside',
            marker=dict(
                color=top_features['importance'],
                colorscale='Viridis',
                showscale=False,
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        ))
        
        fig9.update_layout(
            title="Top 10 Features Driving Churn Predictions",
            xaxis_title="Feature Importance",
            yaxis_title="",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a8b2d1'),
            height=450,
            xaxis=dict(gridcolor='rgba(102, 126, 234, 0.1)'),
            margin=dict(t=60, b=60, l=120, r=60)
        )
        
        st.plotly_chart(fig9, use_container_width=True)
    
    with col2:
        st.markdown("**Model Performance Metrics**")
        
        # Visualize model classification performance on test data
        cm = confusion_matrix(y_test, y_pred)
        
        fig10 = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Active', 'Predicted Churn'],
            y=['Actual Active', 'Actual Churn'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            showscale=False
        ))
        
        fig10.update_layout(
            title="Confusion Matrix",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a8b2d1'),
            height=350,
            margin=dict(t=60, b=60, l=60, r=40)
        )
        
        st.plotly_chart(fig10, use_container_width=True)
        
        # Present consolidated model evaluation statistics
        st.markdown("**Performance Summary**")
        st.markdown(f"""
        - **Accuracy**: {((cm[0,0] + cm[1,1]) / cm.sum()):.3f}
        - **Precision**: {precision:.3f}
        - **Recall**: {recall:.3f}
        - **F1-Score**: {f1:.3f}
        - **ROC-AUC**: {model_auc:.3f}
        """)
    
    # Translate model outputs into actionable business intelligence
    st.markdown("---")
    st.markdown("##### Key Model Insights")
    
    st.markdown(f"""
    <div class='insight-box'>
        <h3> Automated Insights</h3>
        <div class='insight-item'>
            <strong>Top Churn Driver:</strong> {feature_importance.iloc[0]['feature']} 
            (Importance: {feature_importance.iloc[0]['importance']:.3f})
        </div>
        <div class='insight-item'>
            <strong>Model Confidence:</strong> The model achieves {model_auc:.1%} ROC-AUC, 
            indicating {'excellent' if model_auc > 0.85 else 'good' if model_auc > 0.75 else 'moderate'} 
            predictive accuracy
        </div>
        <div class='insight-item'>
            <strong>Precision-Recall Trade-off:</strong> With {precision:.1%} precision and {recall:.1%} recall, 
            the model {'balances accuracy with coverage' if abs(precision - recall) < 0.1 else 'prioritizes ' + ('accuracy' if precision > recall else 'coverage')}
        </div>
        <div class='insight-item'>
            <strong>Business Impact:</strong> Targeting the top {int(len(df) * 0.2)} at-risk customers 
            could prevent approximately ₹{(lost_clv * 0.2)/1e6:.2f}M in revenue loss
        </div>
    </div>
    """, unsafe_allow_html=True)

# Enable financial modeling for retention initiative planning
st.markdown("<div class='section-header'> Retention Strategy ROI Calculator</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("##### Scenario Planning")
    
    reduction_target = st.slider(
        " Target Churn Reduction (%)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Expected reduction in churn rate from retention initiatives"
    )
    
    intervention_rate = st.slider(
        " % of At-Risk Customers to Target",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="Percentage of high-risk customers to include in retention program"
    )

with col2:
    st.markdown("##### Cost Parameters")
    st.metric("Per-Customer Cost", f"₹{retention_cost:,}")
    st.metric("Target Customer Count", f"{int(churned * (intervention_rate/100)):,}")

# Compute ROI metrics based on retention strategy parameters
current_loss = lost_clv
targeted_customers = int(churned * (intervention_rate / 100))
total_intervention_cost = targeted_customers * retention_cost
prevented_churn = current_loss * (reduction_target / 100) * (intervention_rate / 100)
net_benefit = prevented_churn - total_intervention_cost
roi = (net_benefit / total_intervention_cost) * 100 if total_intervention_cost > 0 else 0

st.markdown("##### Financial Impact Projection")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        " Current Annual Loss",
        f"₹{current_loss/1e6:.2f}M",
        help="Current revenue loss due to churn"
    )

with col2:
    st.metric(
        " Potential Savings",
        f"₹{prevented_churn/1e6:.2f}M",
        delta=f"{(prevented_churn/current_loss)*100:.1f}% of loss",
        help="Revenue preserved through retention"
    )

with col3:
    st.metric(
        " Program Investment",
        f"₹{total_intervention_cost/1e6:.2f}M",
        help="Total cost of retention program"
    )

with col4:
    st.metric(
        " Net Benefit (ROI)",
        f"₹{net_benefit/1e6:.2f}M",
        delta=f"{roi:.0f}% ROI",
        delta_color="normal",
        help="Net financial benefit of retention program"
    )

# Compare financial outcomes across retention strategy scenarios
st.markdown("##### Financial Scenario Comparison")

scenarios = pd.DataFrame({
    'Scenario': ['Current State', 'Conservative (10%)', 'Moderate (20%)', 'Aggressive (30%)'],
    'Churn Reduction': [0, 10, 20, 30],
    'Revenue Saved': [
        0,
        current_loss * 0.10 * (intervention_rate/100),
        current_loss * 0.20 * (intervention_rate/100),
        current_loss * 0.30 * (intervention_rate/100)
    ],
    'Program Cost': [0, total_intervention_cost, total_intervention_cost, total_intervention_cost]
})

scenarios['Net Benefit'] = scenarios['Revenue Saved'] - scenarios['Program Cost']

fig_roi = go.Figure()

fig_roi.add_trace(go.Bar(
    name='Revenue Saved',
    x=scenarios['Scenario'],
    y=scenarios['Revenue Saved'] / 1e6,
    marker=dict(color='#50fa7b'),
    text=(scenarios['Revenue Saved'] / 1e6).round(2),
    texttemplate='₹%{text}M',
    textposition='outside'
))

fig_roi.add_trace(go.Bar(
    name='Program Cost',
    x=scenarios['Scenario'],
    y=scenarios['Program Cost'] / 1e6,
    marker=dict(color='#ff5555'),
    text=(scenarios['Program Cost'] / 1e6).round(2),
    texttemplate='₹%{text}M',
    textposition='outside'
))

fig_roi.add_trace(go.Scatter(
    name='Net Benefit',
    x=scenarios['Scenario'],
    y=scenarios['Net Benefit'] / 1e6,
    mode='lines+markers',
    line=dict(color='#667eea', width=3),
    marker=dict(size=12, color='#764ba2', line=dict(color='white', width=2)),
    yaxis='y2',
    text=(scenarios['Net Benefit'] / 1e6).round(2),
    texttemplate='₹%{text}M',
    textposition='top center'
))

fig_roi.update_layout(
    title="ROI Analysis Across Retention Scenarios",
    xaxis_title="Retention Strategy",
    yaxis=dict(title='Revenue & Cost (₹ Million)', side='left', gridcolor='rgba(102, 126, 234, 0.1)'),
    yaxis2=dict(title='Net Benefit (₹ Million)', side='right', overlaying='y', showgrid=False),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#a8b2d1'),
    height=500,
    barmode='group',
    legend=dict(x=0.6, y=1.12, orientation='h'),
    margin=dict(t=100, b=60, l=60, r=60),
    hovermode='x unified'
)

st.plotly_chart(fig_roi, use_container_width=True)

# Surface data-driven recommendations for retention strategy execution
st.markdown("<div class='section-header'>🎯 Strategic Action Plan</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='insight-box'>
        <h3> Critical Priority Actions</h3>
        <div class='insight-item'>
            <strong>High-Value Disengaged Customers:</strong> Launch immediate personalized retention campaign with dedicated relationship managers
        </div>
        <div class='insight-item'>
            <strong>Early Tenure Attrition:</strong> Implement enhanced onboarding program with regular check-ins during first 2 years
        </div>
        <div class='insight-item'>
            <strong>Product Underutilization:</strong> Customers with single products show elevated churn - cross-sell opportunities
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='insight-box'>
        <h3> Medium-Term Initiatives</h3>
        <div class='insight-item'>
            <strong>Geographic Focus:</strong> Deploy region-specific retention strategies based on local churn patterns
        </div>
        <div class='insight-item'>
            <strong>Engagement Programs:</strong> Develop loyalty initiatives to convert low-engagement customers to active status
        </div>
        <div class='insight-item'>
            <strong>Predictive Monitoring:</strong> Implement real-time ML scoring for proactive churn intervention
        </div>
    </div>
    """, unsafe_allow_html=True)

# Provide downloadable reports for offline analysis and stakeholder distribution
st.markdown("<div class='section-header'> Export & Reporting</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        " Download Full Analysis Report",
        filtered_df.to_csv(index=False),
        "churn_analysis_complete.csv",
        "text/csv",
        use_container_width=True
    )

with col2:
    st.download_button(
        " Download High-Risk Segment",
        high_risk_df.to_csv(index=False) if len(high_risk_df) > 0 else "No high-risk customers",
        "high_risk_segment.csv",
        "text/csv",
        use_container_width=True
    )

with col3:
    st.download_button(
        " Download Feature Importance",
        feature_importance.to_csv(index=False),
        "feature_importance.csv",
        "text/csv",
        use_container_width=True
    )

# Display platform attribution and technical stack information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8892b0; padding: 2rem 0;'>
    <p style='margin: 0; font-size: 0.9rem;'>
        Enterprise Customer Analytics Platform • Built with Advanced ML & Predictive Intelligence
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #4a5568;'>
        Powered by XGBoost • Streamlit • Plotly
    </p>
</div>
""", unsafe_allow_html=True)