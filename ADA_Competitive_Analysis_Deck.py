#!/usr/bin/env python3
"""
ADA Competitive Analysis Presentation Deck
Interactive comparison with FICO Falcon and SAS Fraud Management
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import time

st.set_page_config(
    page_title="ADA Competitive Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .competitor-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .ada-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .slide-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .highlight {
        background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def create_radar_chart():
    """Create radar chart comparing ADA vs competitors"""
    categories = ['Accuracy', 'Speed', 'Explainability', 'Cost Efficiency']
    
    fig = go.Figure()
    
    # ADA values
    fig.add_trace(go.Scatterpolar(
        r=[92, 95, 90, 80],
        theta=categories,
        fill='toself',
        name='ADA',
        line_color='#4facfe',
        fillcolor='rgba(79, 172, 254, 0.3)'
    ))
    
    # FICO Falcon values
    fig.add_trace(go.Scatterpolar(
        r=[85, 75, 60, 65],
        theta=categories,
        fill='toself',
        name='FICO Falcon',
        line_color='#f093fb',
        fillcolor='rgba(240, 147, 251, 0.3)'
    ))
    
    # SAS Fraud values
    fig.add_trace(go.Scatterpolar(
        r=[80, 85, 70, 70],
        theta=categories,
        fill='toself',
        name='SAS Fraud',
        line_color='#f5576c',
        fillcolor='rgba(245, 87, 108, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Competitive Analysis Radar Chart",
        font=dict(size=14),
        height=600
    )
    
    return fig

def create_bar_comparison():
    """Create bar chart comparison"""
    metrics = ['Accuracy', 'Speed', 'Explainability', 'Cost Efficiency']
    ada_values = [92, 95, 90, 80]
    fico_values = [85, 75, 60, 65]
    sas_values = [80, 85, 70, 70]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='ADA',
        x=metrics,
        y=ada_values,
        marker_color='#4facfe',
        text=ada_values,
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='FICO Falcon',
        x=metrics,
        y=fico_values,
        marker_color='#f093fb',
        text=fico_values,
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='SAS Fraud',
        x=metrics,
        y=sas_values,
        marker_color='#f5576c',
        text=sas_values,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Performance Comparison Across Key Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score (%)",
        barmode='group',
        height=500,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_feature_comparison_table():
    """Create feature comparison table"""
    data = {
        'Feature': [
            'Detection Accuracy',
            'Processing Speed',
            'Real-time Processing',
            'Explainability',
            'Cost Efficiency',
            'Infrastructure Requirements',
            'Integration Complexity',
            'Customization',
            'Scalability',
            'Support & Maintenance'
        ],
        'ADA': [
            '92%',
            '95%',
            '< 1 second',
            '90%',
            '80%',
            'Lightweight',
            'Low',
            'High',
            'Excellent',
            'Low'
        ],
        'FICO Falcon': [
            '85%',
            '75%',
            '3-5 seconds',
            '60%',
            '65%',
            'Heavy',
            'High',
            'Moderate',
            'Good',
            'High'
        ],
        'SAS Fraud': [
            '80%',
            '85%',
            '1-2 seconds',
            '70%',
            '70%',
            'Moderate',
            'Medium',
            'High',
            'Good',
            'Medium'
        ]
    }
    
    df = pd.DataFrame(data)
    return df

def main():
    # Navigation
    st.sidebar.title("📊 ADA Competitive Analysis")
    slide = st.sidebar.selectbox(
        "Select Slide:",
        [
            "🏠 Executive Summary",
            "📈 Performance Comparison",
            "🎯 Radar Chart Analysis", 
            "📋 Feature Comparison",
            "💰 Cost Analysis",
            "🚀 Competitive Advantages",
            "📊 Market Positioning",
            "🎯 Recommendations"
        ]
    )
    
    if slide == "🏠 Executive Summary":
        st.markdown('<div class="main-header">ADA Competitive Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="slide-title">Executive Summary</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="ada-card">', unsafe_allow_html=True)
            st.markdown("### 🚀 ADA Fraud Detection")
            st.markdown("**Market Leader in All Dimensions**")
            st.markdown("- **Accuracy**: 92%")
            st.markdown("- **Speed**: 95%")
            st.markdown("- **Explainability**: 90%")
            st.markdown("- **Cost Efficiency**: 80%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="competitor-card">', unsafe_allow_html=True)
            st.markdown("### 🏢 FICO Falcon")
            st.markdown("**Traditional Enterprise Solution**")
            st.markdown("- **Accuracy**: 85%")
            st.markdown("- **Speed**: 75%")
            st.markdown("- **Explainability**: 60%")
            st.markdown("- **Cost Efficiency**: 65%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="competitor-card">', unsafe_allow_html=True)
            st.markdown("### 🏢 SAS Fraud")
            st.markdown("**Rule-Based Enterprise Platform**")
            st.markdown("- **Accuracy**: 80%")
            st.markdown("- **Speed**: 85%")
            st.markdown("- **Explainability**: 70%")
            st.markdown("- **Cost Efficiency**: 70%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("### 🎯 Key Findings")
        st.markdown("- **ADA leads in all 4 key metrics**")
        st.markdown("- **Superior real-time processing** (< 1 second vs 1-5 seconds)")
        st.markdown("- **Best explainability** for regulatory compliance")
        st.markdown("- **Lower total cost of ownership**")
        st.markdown("- **Lightweight infrastructure requirements**")
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif slide == "📈 Performance Comparison":
        st.markdown('<div class="slide-title">Performance Comparison</div>', unsafe_allow_html=True)
        
        # Bar chart
        fig = create_bar_comparison()
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("### 🏆 ADA Advantages")
            st.markdown("- **+7% accuracy** over FICO Falcon")
            st.markdown("- **+20% speed** over FICO Falcon")
            st.markdown("- **+30% explainability** over FICO Falcon")
            st.markdown("- **+15% cost efficiency** over FICO Falcon")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("### 📊 Competitive Edge")
            st.markdown("- **+12% accuracy** over SAS Fraud")
            st.markdown("- **+10% speed** over SAS Fraud")
            st.markdown("- **+20% explainability** over SAS Fraud")
            st.markdown("- **+10% cost efficiency** over SAS Fraud")
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif slide == "🎯 Radar Chart Analysis":
        st.markdown('<div class="slide-title">Radar Chart Analysis</div>', unsafe_allow_html=True)
        
        # Radar chart
        fig = create_radar_chart()
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("### 📊 Radar Chart Insights")
        st.markdown("- **ADA shows the largest area** - indicating superior overall performance")
        st.markdown("- **Balanced performance** across all dimensions")
        st.markdown("- **FICO Falcon** struggles with explainability and cost efficiency")
        st.markdown("- **SAS Fraud** shows moderate performance but lacks in accuracy")
        st.markdown("- **ADA's polygon** is the most comprehensive and balanced")
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif slide == "📋 Feature Comparison":
        st.markdown('<div class="slide-title">Detailed Feature Comparison</div>', unsafe_allow_html=True)
        
        # Feature comparison table
        df = create_feature_comparison_table()
        st.dataframe(df, use_container_width=True)
        
        # Feature analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("### 🎯 ADA Strengths")
            st.markdown("- **Lightweight infrastructure** requirements")
            st.markdown("- **Low integration complexity**")
            st.markdown("- **High customization** capabilities")
            st.markdown("- **Excellent scalability**")
            st.markdown("- **Low support & maintenance** costs")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("### ⚠️ Competitor Limitations")
            st.markdown("- **Heavy infrastructure** requirements (FICO)")
            st.markdown("- **High integration complexity** (FICO)")
            st.markdown("- **Limited explainability** (FICO)")
            st.markdown("- **Moderate accuracy** (SAS)")
            st.markdown("- **Higher costs** (Both)")
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif slide == "💰 Cost Analysis":
        st.markdown('<div class="slide-title">Cost Analysis</div>', unsafe_allow_html=True)
        
        # Cost comparison chart
        cost_data = {
            'Cost Component': ['Licensing', 'Infrastructure', 'Integration', 'Maintenance', 'Training', 'Total'],
            'ADA': [100, 80, 60, 70, 50, 360],
            'FICO Falcon': [150, 120, 100, 90, 80, 540],
            'SAS Fraud': [130, 100, 80, 85, 70, 465]
        }
        
        cost_df = pd.DataFrame(cost_data)
        
        fig = px.bar(
            cost_df, 
            x='Cost Component', 
            y=['ADA', 'FICO Falcon', 'SAS Fraud'],
            title="Cost Comparison (Relative Index)",
            barmode='group'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost savings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 💰 Cost Savings vs FICO")
            st.markdown("- **Total Cost**: 33% lower")
            st.markdown("- **Infrastructure**: 33% lower")
            st.markdown("- **Integration**: 40% lower")
            st.markdown("- **Maintenance**: 22% lower")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 💰 Cost Savings vs SAS")
            st.markdown("- **Total Cost**: 23% lower")
            st.markdown("- **Infrastructure**: 20% lower")
            st.markdown("- **Integration**: 25% lower")
            st.markdown("- **Maintenance**: 18% lower")
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif slide == "🚀 Competitive Advantages":
        st.markdown('<div class="slide-title">ADA Competitive Advantages</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="ada-card">', unsafe_allow_html=True)
            st.markdown("### ⚡ Speed & Performance")
            st.markdown("- **< 1 second** processing time")
            st.markdown("- **500+ TPS** throughput")
            st.markdown("- **Real-time** fraud detection")
            st.markdown("- **Low latency** architecture")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown('<div class="ada-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 AI & Explainability")
            st.markdown("- **90% explainability** score")
            st.markdown("- **XAI integration**")
            st.markdown("- **Regulatory compliance**")
            st.markdown("- **Transparent decisions**")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="ada-card">', unsafe_allow_html=True)
            st.markdown("### 💰 Cost Efficiency")
            st.markdown("- **Lower TCO** than competitors")
            st.markdown("- **Lightweight infrastructure**")
            st.markdown("- **Easy integration**")
            st.markdown("- **Reduced maintenance**")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown('<div class="ada-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Accuracy & Reliability")
            st.markdown("- **92% accuracy** rate")
            st.markdown("- **Advanced ML models**")
            st.markdown("- **Continuous learning**")
            st.markdown("- **Adaptive algorithms**")
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif slide == "📊 Market Positioning":
        st.markdown('<div class="slide-title">Market Positioning</div>', unsafe_allow_html=True)
        
        # Market positioning chart
        fig = go.Figure()
        
        # Add scatter points for positioning
        fig.add_trace(go.Scatter(
            x=[92, 85, 80],  # Accuracy
            y=[95, 75, 85],  # Speed
            mode='markers+text',
            marker=dict(size=[20, 15, 15], color=['#4facfe', '#f093fb', '#f5576c']),
            text=['ADA', 'FICO', 'SAS'],
            textposition="top center",
            name='Solutions'
        ))
        
        fig.update_layout(
            title="Market Positioning: Accuracy vs Speed",
            xaxis_title="Accuracy (%)",
            yaxis_title="Speed (%)",
            xaxis=dict(range=[75, 100]),
            yaxis=dict(range=[70, 100]),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Positioning analysis
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("### 🎯 Market Positioning Analysis")
        st.markdown("- **ADA**: Premium position with best accuracy AND speed")
        st.markdown("- **FICO Falcon**: High accuracy but slower processing")
        st.markdown("- **SAS Fraud**: Balanced but lower accuracy")
        st.markdown("- **ADA dominates** the top-right quadrant (high accuracy + high speed)")
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif slide == "🎯 Recommendations":
        st.markdown('<div class="slide-title">Strategic Recommendations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("### 🚀 Go-to-Market Strategy")
            st.markdown("**1. Target High-Value Segments**")
            st.markdown("- Financial institutions requiring real-time processing")
            st.markdown("- E-commerce platforms with high transaction volumes")
            st.markdown("- Regulated industries needing explainability")
            
            st.markdown("**2. Competitive Messaging**")
            st.markdown("- Emphasize speed advantage (5x faster than FICO)")
            st.markdown("- Highlight cost savings (33% vs FICO, 23% vs SAS)")
            st.markdown("- Showcase explainability for compliance")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("### 📈 Growth Opportunities")
            st.markdown("**3. Market Expansion**")
            st.markdown("- Target mid-market companies priced out by FICO/SAS")
            st.markdown("- Focus on emerging markets with cost sensitivity")
            st.markdown("- Leverage cloud-native architecture advantage")
            
            st.markdown("**4. Product Development**")
            st.markdown("- Continue improving accuracy beyond 92%")
            st.markdown("- Enhance explainability features")
            st.markdown("- Develop industry-specific solutions")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Final recommendation
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Key Recommendation")
        st.markdown("**Position ADA as the 'Next-Generation Fraud Detection Platform'**")
        st.markdown("- Superior performance across all metrics")
        st.markdown("- Modern, cloud-native architecture")
        st.markdown("- Lower total cost of ownership")
        st.markdown("- Better regulatory compliance through explainability")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 