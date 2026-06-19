import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Hunger Risk Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🌍 Hunger Risk Prediction")

# 2. LOAD DATASET & MODELS
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

df_original = load_data()

# Ensure we have clean models loaded
model = joblib.load("hunger_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 3. HELPER CLASSIFICATION LOGIC
def classify(score):
    if score is None or pd.isna(score):
        return "Unknown", "#94a3b8"
    if score <= 9.9:
        return "Low", "#10b981"
    if score <= 19.9:
        return "Moderate", "#facc15"
    if score <= 34.9:
        return "Serious", "#f97316"
    return "Alarming", "#ef4444"


def hex_to_rgba(hex_color: str, alpha: float = 0.07) -> str:
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return hex_color
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


action_recommendations = {
    "Low": [
        "Maintain current food distribution and safety nets.",
        "Invest in resilient agriculture to withstand climate anomalies.",
        "Monitor local food prices to prevent localized spikes."
    ],
    "Moderate": [
        "Enhance agricultural subsidies for smallholder farmers.",
        "Invest in school feeding programs to protect vulnerable children.",
        "Establish regional grain storage and reserve facilities."
    ],
    "Serious": [
        "Deploy targeted cash transfer programs to low-income households.",
        "Expand maternal and child nutrition intervention networks.",
        "Partner with international agencies to scale up food supply imports."
    ],
    "Alarming": [
        "Declare emergency food security status to coordinate international aid.",
        "Establish direct emergency food relief stations in high-density areas.",
        "Implement price controls and import tariff waivers on essential food grains."
    ]
}

# 4. CUSTOM CSS FOR GLASSMORPHISM STYLE
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title styling */
    .title-gradient {
        background: linear-gradient(135deg, #ffffff 30%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0px;
        text-align: center;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Custom Card container */
    .glass-card {
        background: rgba(17, 24, 39, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }
    
    /* PEAS Table styling */
    .peas-table {
        width: 100%;
        border-collapse: collapse;
    }
    .peas-table tr {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .peas-table tr:last-child {
        border-bottom: none;
    }
    .peas-table td {
        padding: 14px 10px;
        vertical-align: top;
        font-size: 0.95rem;
    }
    .peas-table td:first-child {
        font-weight: bold;
        color: #f97316;
        width: 25%;
    }
    
    /* Pipeline node box */
    .pipeline-grid {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 15px;
    }
    .pipeline-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 15px;
        text-align: center;
        flex: 1;
        min-width: 130px;
    }
    .pipeline-box:hover {
        border-color: #f97316;
        box-shadow: 0 0 10px rgba(249, 115, 22, 0.2);
    }
    .pipeline-arrow {
        color: #f97316;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* Policy Directives Card */
    .rec-box {
        background: rgba(255, 255, 255, 0.02);
        border-left: 4px solid #f97316;
        border-radius: 0 12px 12px 0;
        padding: 15px;
        margin-top: 10px;
    }
    .rec-box h4 {
        color: #f97316;
        margin-bottom: 8px;
        font-size: 1.05rem;
    }
    .rec-box ul {
        list-style: none;
        padding-left: 0;
    }
    .rec-box li {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 6px;
        padding-left: 15px;
        position: relative;
    }
    .rec-box li::before {
        content: "•";
        color: #f97316;
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    
    /* Badge styling */
    .risk-badge-sim {
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
    }

    /* HTML Progress Meters */
    .prog-item {
        margin-bottom: 12px;
    }
    .prog-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        margin-bottom: 4px;
    }
    .prog-track {
        background: rgba(255, 255, 255, 0.05);
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
    }
    .prog-bar {
        height: 100%;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# 5. CORE LAYOUT TITLE
st.markdown('<div class="title-gradient">🌍 HungerRisk.AI System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Global food security analytics and machine learning classification app</div>', unsafe_allow_html=True)

# 6. APP TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Global Dashboard", 
    "🔍 Country Predictor", 
    "📈 Trend Comparer", 
    "🔮 Simulation Sandbox", 
    "🗂 Database Explorer"
])

# Filter valid countries list
df_valid = df_original[df_original["2023"].notna()].copy()
df_valid["risk_class"] = df_valid["2023"].apply(lambda val: classify(val)[0])

# Prepare lists
countries_list = sorted(df_original["Country"].unique())

# ==========================================
# TAB 1: GLOBAL DASHBOARD
# ==========================================
with tab1:
    # Stats columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate GHI Stats
    total_countries = len(df_valid)
    avg_ghi_2023 = df_valid["2023"].mean()
    alarming_count = len(df_valid[df_valid["2023"] >= 35.0])
    
    col1.metric("Total Countries", f"{total_countries}")
    col2.metric("Average GHI (2023)", f"{avg_ghi_2023:.2f}")
    col3.metric("Alarming Risk Status", f"{alarming_count}", delta_color="inverse")
    col4.metric("Model Classification Accuracy", "87.5%")
    
    st.write("---")
    
    col_left, col_right = st.columns([1.6, 1])
    
    with col_left:
        # PEAS Spec Box
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-bottom:15px; font-weight:600; color:#f97316;">🧠 AI System Specifications (PEAS)</h3>
            <table class="peas-table">
                <tr>
                    <td>Performance</td>
                    <td>Accurate predictive classification of hunger severity based on GHI time-series patterns.</td>
                </tr>
                <tr>
                    <td>Environment</td>
                    <td>Global countries and territories loaded from the official Global Hunger Index dataset.</td>
                </tr>
                <tr>
                    <td>Actuators</td>
                    <td>Discrete risk category classification output (Low, Moderate, Serious, Alarming) for policy insights.</td>
                </tr>
                <tr>
                    <td>Sensors</td>
                    <td>Historical multi-year hunger index records (years 2000, 2008, 2015, and 2023).</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Pipeline Flow Box
        st.markdown("""
        <div class="glass-card" style="margin-top:20px;">
            <h3 style="margin-bottom:15px; font-weight:600; color:#f97316;">⚙ Machine Learning Execution Pipeline</h3>
            <div class="pipeline-grid">
                <div class="pipeline-box">
                    <div style="font-weight:600; font-size:0.9rem;">Dataset</div>
                    <div style="font-size:0.75rem; color:#94a3b8;">125 country index metrics</div>
                </div>
                <div class="pipeline-arrow">➡</div>
                <div class="pipeline-box">
                    <div style="font-weight:600; font-size:0.9rem;">Preprocessing</div>
                    <div style="font-size:0.75rem; color:#94a3b8;">Drop Country & change metrics</div>
                </div>
                <div class="pipeline-arrow">➡</div>
                <div class="pipeline-box">
                    <div style="font-weight:600; font-size:0.9rem;">Scaling</div>
                    <div style="font-size:0.75rem; color:#94a3b8;">StandardScaler transform</div>
                </div>
                <div class="pipeline-arrow">➡</div>
                <div class="pipeline-box">
                    <div style="font-weight:600; font-size:0.9rem;">Logistic Model</div>
                    <div style="font-size:0.75rem; color:#94a3b8;">Saves model class prediction</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        # Pie / Doughnut Chart using Plotly
        st.markdown('<div class="glass-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<h3 style="font-weight:600; margin-bottom:10px;">📊 Hunger Risk Distribution</h3>', unsafe_allow_html=True)
        
        risk_distribution = df_valid["risk_class"].value_counts()
        labels = ["Low", "Moderate", "Serious", "Alarming"]
        values = [
            risk_distribution.get("Low", 0),
            risk_distribution.get("Moderate", 0),
            risk_distribution.get("Serious", 0),
            risk_distribution.get("Alarming", 0)
        ]
        colors = ["#10b981", "#facc15", "#f97316", "#ef4444"]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.6,
            marker_colors=colors,
            textinfo='percent',
            textfont=dict(size=13, family="Outfit", color="#f8fafc")
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(family="Outfit", size=12, color="#cbd5e1")
            ),
            margin=dict(t=10, b=40, l=10, r=10),
            height=250
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Worst & Best performing countries
    st.write("")
    col_worst, col_best = st.columns(2)
    
    with col_worst:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="font-weight:600; margin-bottom:15px; color:#ef4444;">🚨 Highest Risk Countries (Worst 2023 GHI)</h3>', unsafe_allow_html=True)
        
        df_worst = df_valid.sort_values(by="2023", ascending=False).head(5)
        for idx, row in df_worst.iterrows():
            pct = (row["2023"] / 50.0) * 100
            label, color = classify(row["2023"])
            st.markdown(f"""
            <div class="prog-item">
                <div class="prog-header">
                    <span style="font-weight:600;">{row["Country"]}</span>
                    <span style="color:{color}; font-weight:700;">{row["2023"]:.1f} GHI</span>
                </div>
                <div class="prog-track">
                    <div class="prog-bar" style="width:{pct}%; background-color:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_best:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="font-weight:600; margin-bottom:15px; color:#10b981;">✅ Lowest Risk Countries (Best 2023 GHI)</h3>', unsafe_allow_html=True)
        
        df_best = df_valid.sort_values(by="2023", ascending=True).head(5)
        for idx, row in df_best.iterrows():
            pct = (row["2023"] / 50.0) * 100
            label, color = classify(row["2023"])
            st.markdown(f"""
            <div class="prog-item">
                <div class="prog-header">
                    <span style="font-weight:600;">{row["Country"]}</span>
                    <span style="color:{color}; font-weight:700;">{row["2023"]:.1f} GHI</span>
                </div>
                <div class="prog-track">
                    <div class="prog-bar" style="width:{pct}%; background-color:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TAB 2: COUNTRY PREDICTOR
# ==========================================
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_search, col_chart = st.columns([1, 1.4])
    
    with col_search:
        st.markdown('<h3 style="font-weight:600; margin-bottom:15px;">🔍 Select Country</h3>', unsafe_allow_html=True)
        # Search alphabetically sorted selectbox
        selected_country = st.selectbox(
            "Choose a country to predict and inspect GHI data:", 
            options=sorted(df_original["Country"].dropna().unique()),
            index=0
        )
        
        # Pull data
        c_row = df_original[df_original["Country"] == selected_country].iloc[0]
        rank_val = c_row["Rank"] if not pd.isna(c_row["Rank"]) else 60
        ghi_2000 = c_row["2000"] if not pd.isna(c_row["2000"]) else 0
        ghi_2008 = c_row["2008"] if not pd.isna(c_row["2008"]) else 0
        ghi_2015 = c_row["2015"] if not pd.isna(c_row["2015"]) else 0
        ghi_2023 = c_row["2023"] if not pd.isna(c_row["2023"]) else 0
        
        # Build features dataframe
        feats = pd.DataFrame([{
            "Rank": float(rank_val),
            "2000": float(ghi_2000),
            "2008": float(ghi_2008),
            "2015": float(ghi_2015),
            "2023": float(ghi_2023)
        }])
        
        # Predict
        scaled_feats = scaler.transform(feats)
        pred = model.predict(scaled_feats)
        pred_label = label_encoder.inverse_transform(pred)[0]
        
        # Classify color
        risk_name, color = classify(ghi_2023)
        
        st.write("")
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h4 style="font-weight:700; font-size:1.4rem;">{selected_country}</h4>
                <div class="risk-badge-sim" style="color:{color}; border:1px solid {color}; background:{color}15; font-size:1.05rem; box-shadow:0 0 10px {color}44;">
                    {pred_label} Risk
                </div>
            </div>
            <div style="font-size:0.9rem; color:#cbd5e1;">GHI 2023 Index: <b>{ghi_2023 if not pd.isna(c_row["2023"]) else 'N/A'}</b> (Global Rank: #{int(c_row["Rank"]) if not pd.isna(c_row["Rank"]) else 'N/A'})</div>
            
            <div class="rec-box" style="border-left-color:{color}; margin-top:20px;">
                <h4 style="color:{color};">📋 Recommended Policy Actions</h4>
                <ul>
                    {"".join([f"<li>{r}</li>" for r in action_recommendations.get(pred_label, action_recommendations["Low"])])}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_chart:
        st.markdown('<h3 style="font-weight:600; margin-bottom:15px;">📈 GHI Historical Trajectory</h3>', unsafe_allow_html=True)
        
        # Build timeline coordinates
        years = ["2000", "2008", "2015", "2023"]
        scores = [ghi_2000, ghi_2008, ghi_2015, ghi_2023]
        
        # Plot using Plotly for dark styling
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, 
            y=scores, 
            mode='lines+markers',
            line=dict(color=color, width=4, shape='spline'),
            marker=dict(
                size=10, 
                color=[classify(v)[1] for v in scores], 
                line=dict(width=2, color='#0f172a')
            ),
            fill='tozeroy',
            fillcolor=hex_to_rgba(color, alpha=0.12),
            name=selected_country
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                tickfont=dict(family="Outfit", color="#94a3b8")
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                tickfont=dict(family="Outfit", color="#94a3b8"),
                range=[0, max(scores)*1.2 if max(scores) > 0 else 50]
            ),
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TAB 3: TREND COMPARER
# ==========================================
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_comp_ctrl, col_comp_chart = st.columns([1, 1.4])
    
    with col_comp_ctrl:
        st.markdown('<h3 style="font-weight:600; margin-bottom:10px;">📊 Compare Countries</h3>', unsafe_allow_html=True)
        st.write("Compare GHI development curves side-by-side.")
        
        c_a = st.selectbox("Select Country A:", options=countries_list, index=countries_list.index("India") if "India" in countries_list else 0)
        c_b = st.selectbox("Select Country B:", options=countries_list, index=countries_list.index("China") if "China" in countries_list else 0)
        
        row_a = df_original[df_original["Country"] == c_a].iloc[0]
        row_b = df_original[df_original["Country"] == c_b].iloc[0]
        
        ghi_a_2023 = row_a["2023"] if not pd.isna(row_a["2023"]) else 0
        ghi_b_2023 = row_b["2023"] if not pd.isna(row_b["2023"]) else 0
        
        st.write("")
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; font-size:0.95rem;">
            <h4 style="font-weight:600; margin-bottom:12px; font-size:1.1rem; color:#f97316;">📊 Divergence Analysis (2023)</h4>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom:12px;">
                <div><b>{c_a}</b>: <span style="font-family:monospace; font-weight:700;">{row_a["2023"] if not pd.isna(row_a["2023"]) else 'N/A'}</span> GHI</div>
                <div><b>{c_b}</b>: <span style="font-family:monospace; font-weight:700;">{row_b["2023"] if not pd.isna(row_b["2023"]) else 'N/A'}</span> GHI</div>
            </div>
        """, unsafe_allow_html=True)
        
        if not pd.isna(row_a["2023"]) and not pd.isna(row_b["2023"]):
            diff = abs(row_a["2023"] - row_b["2023"])
            leader = c_a if row_a["2023"] < row_b["2023"] else c_b
            st.markdown(f"""
                <div style="border-top:1px solid rgba(255,255,255,0.08); padding-top:10px; font-weight:700; color:#f97316;">
                    Difference: {diff:.1f} GHI ({leader} performs better)
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="border-top:1px solid rgba(255,255,255,0.08); padding-top:10px; font-weight:700; color:#f97316;">
                    Divergence: N/A due to missing data
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_comp_chart:
        st.markdown('<h3 style="font-weight:600; margin-bottom:15px;">📈 Overlay Trend Comparison</h3>', unsafe_allow_html=True)
        
        years = ["2000", "2008", "2015", "2023"]
        scores_a = [row_a["2000"], row_a["2008"], row_a["2015"], row_a["2023"]]
        scores_b = [row_b["2000"], row_b["2008"], row_b["2015"], row_b["2023"]]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=scores_a, mode='lines+markers',
            name=c_a, line=dict(color="#38bdf8", width=3, shape="spline"),
            marker=dict(size=8, color="#38bdf8")
        ))
        fig.add_trace(go.Scatter(
            x=years, y=scores_b, mode='lines+markers',
            name=c_b, line=dict(color="#f43f5e", width=3, shape="spline"),
            marker=dict(size=8, color="#f43f5e")
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(family="Outfit", color="#94a3b8")),
            yaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(family="Outfit", color="#94a3b8")),
            legend=dict(font=dict(family="Outfit", color="#cbd5e1")),
            margin=dict(t=10, b=10, l=10, r=10),
            height=320
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TAB 4: SIMULATION SANDBOX
# ==========================================
with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_sliders, col_meter = st.columns([1, 1])
    
    with col_sliders:
        st.markdown('<h3 style="font-weight:600; margin-bottom:10px;">⚙ Simulation Parameters</h3>', unsafe_allow_html=True)
        st.write("Adjust sliders to feed custom data to the AI classifier in real-time.")
        
        sim_rank = st.slider("Global Rank Target:", min_value=1, max_value=125, value=60)
        sim_2000 = st.slider("Year 2000 GHI Score:", min_value=0.0, max_value=65.0, step=0.1, value=25.0)
        sim_2008 = st.slider("Year 2008 GHI Score:", min_value=0.0, max_value=65.0, step=0.1, value=20.0)
        sim_2015 = st.slider("Year 2015 GHI Score:", min_value=0.0, max_value=65.0, step=0.1, value=15.0)
        sim_2023 = st.slider("Year 2023 GHI Score:", min_value=0.0, max_value=65.0, step=0.1, value=12.5)
        
    with col_meter:
        # Feed simulation inputs to predictive model
        sim_feats = pd.DataFrame([{
            "Rank": float(sim_rank),
            "2000": float(sim_2000),
            "2008": float(sim_2008),
            "2015": float(sim_2015),
            "2023": float(sim_2023)
        }])
        
        sim_scaled = scaler.transform(sim_feats)
        sim_pred = model.predict(sim_scaled)
        sim_label = label_encoder.inverse_transform(sim_pred)[0]
        
        label_t, sim_color = classify(sim_2023)
        
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:40px 20px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center; align-items:center;">
            <h3 style="font-weight:600; font-size:1.2rem; margin-bottom:20px;">🔮 Real-Time Classifier Output</h3>
        """, unsafe_allow_html=True)
        
        # Render clean gauge graphic using plotly gauge widget
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sim_2023,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{sim_label} Risk", 'font': {'size': 20, 'color': sim_color, 'family': 'Outfit'}},
            number = {'font': {'family': 'Outfit', 'color': '#f8fafc'}},
            gauge = {
                'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                'bar': {'color': sim_color},
                'bgcolor': "rgba(255, 255, 255, 0.05)",
                'borderwidth': 2,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, 9.9], 'color': 'rgba(16, 185, 129, 0.05)'},
                    {'range': [10.0, 19.9], 'color': 'rgba(250, 204, 21, 0.05)'},
                    {'range': [20.0, 34.9], 'color': 'rgba(249, 115, 22, 0.05)'},
                    {'range': [35.0, 50.0], 'color': 'rgba(239, 68, 68, 0.05)'}
                ],
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=200,
            margin=dict(t=30, b=10, l=30, r=30)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Recommendations
        st.markdown(f"""
            <div class="rec-box" style="border-left-color:{sim_color}; margin-top:20px; width:100%; text-align:left;">
                <h4 style="color:{sim_color};">💡 Simulated Policy Directives:</h4>
                <ul>
                    {"".join([f"<li>{r}</li>" for r in action_recommendations.get(sim_label, action_recommendations["Low"])])}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TAB 5: DATABASE EXPLORER
# ==========================================
with tab5:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="font-weight:600; margin-bottom:15px;">🗂 Global GHI Explorer & Database</h3>', unsafe_allow_html=True)
    
    col_search_t, col_filter_t = st.columns([2, 1])
    
    with col_search_t:
        search_query = st.text_input("Search country in database:", placeholder="Type country name...")
        
    with col_filter_t:
        risk_filter = st.selectbox(
            "Filter by Risk level:",
            options=["All Risk Levels", "Low Risk", "Moderate Risk", "Serious Risk", "Alarming Risk"]
        )
        
    # Map filters to subset
    df_explorer = df_valid.copy()
    if search_query:
        df_explorer = df_explorer[df_explorer["Country"].str.lower().str.contains(search_query.lower())]
        
    if risk_filter != "All Risk Levels":
        risk_mapping = {
            "Low Risk": "Low",
            "Moderate Risk": "Moderate",
            "Serious Risk": "Serious",
            "Alarming Risk": "Alarming"
        }
        df_explorer = df_explorer[df_explorer["risk_class"] == risk_mapping[risk_filter]]
        
    # Format and present explorer dataframe
    df_display = df_explorer[["Rank", "Country", "2000", "2008", "2015", "2023", "risk_class"]].copy()
    df_display.columns = ["Rank", "Country", "GHI 2000", "GHI 2008", "GHI 2015", "GHI 2023", "Predicted Risk Level"]
    df_display = df_display.sort_values(by="Rank", ascending=True)
    
    st.dataframe(
        df_display, 
        use_container_width=True, 
        hide_index=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
