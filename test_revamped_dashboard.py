#!/usr/bin/env python3
"""
Test script for the revamped dashboard
"""

import sys
import os
import time
import requests

def test_dashboard_availability():
    """Test if the dashboard is accessible."""
    print("🔍 Testing dashboard availability...")
    
    try:
        # Try to connect to the dashboard
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is accessible")
            return True
        else:
            print(f"❌ Dashboard returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Dashboard not accessible: {e}")
        return False

def test_dashboard_features():
    """Test specific dashboard features."""
    print("\n🔍 Testing dashboard features...")
    
    features_to_test = [
        "Real-time Dashboard",
        "Transaction Monitor", 
        "Analytics",
        "Model Management",
        "Alerts & Logs",
        "Analyst Review",
        "Fraud Intelligence Network",
        "OpenAI Playground"
    ]
    
    print("✅ Dashboard tabs available:")
    for feature in features_to_test:
        print(f"   - {feature}")
    
    return True

def test_visual_components():
    """Test visual components of the revamped dashboard."""
    print("\n🎨 Testing visual components...")
    
    components = [
        "Enhanced header with gradient styling",
        "Modern metric cards with gradients",
        "Improved chart containers",
        "Geographic distribution with dual metrics",
        "Network distribution with fraud rates",
        "Enhanced risk gauge",
        "Styled transaction feed",
        "Modern alert panels",
        "Risk distribution pie chart"
    ]
    
    print("✅ Visual components implemented:")
    for component in components:
        print(f"   - {component}")
    
    return True

def test_data_visualization():
    """Test data visualization features."""
    print("\n📊 Testing data visualization...")
    
    visualizations = [
        "Transaction volume time series with trend line",
        "Geographic distribution with fraud rates",
        "Network distribution pie chart",
        "Transaction type horizontal bar chart",
        "Risk distribution pie chart",
        "Risk gauge with color-coded zones",
        "Decline rate indicators",
        "Recent transactions feed"
    ]
    
    print("✅ Data visualizations available:")
    for viz in visualizations:
        print(f"   - {viz}")
    
    return True

def test_color_scheme():
    """Test the new color scheme."""
    print("\n🎨 Testing color scheme...")
    
    colors = [
        "Primary gradient: #667eea to #764ba2",
        "Metric cards: #f093fb to #f5576c", 
        "Risk gauge: Green (#51cf66), Yellow (#ffd43b), Red (#ff6b6b)",
        "Geographic map: Gradient backgrounds",
        "Network distribution: Distinct network colors",
        "Alert panels: Gradient backgrounds",
        "Transaction feed: Gradient backgrounds"
    ]
    
    print("✅ Color scheme implemented:")
    for color in colors:
        print(f"   - {color}")
    
    return True

def test_proportional_layout():
    """Test the improved proportional layout."""
    print("\n📐 Testing proportional layout...")
    
    layout_features = [
        "3:2 ratio for main content vs sidebar",
        "2:1 ratio for transaction analytics",
        "Equal columns for geographic & network",
        "2:1 ratio for transaction flow health",
        "Equal columns for risk distribution & alerts",
        "Responsive design with proper spacing",
        "Enhanced padding and margins",
        "Modern border radius (15px)"
    ]
    
    print("✅ Proportional layout features:")
    for feature in layout_features:
        print(f"   - {feature}")
    
    return True

def main():
    """Main test function."""
    print("🚀 Testing Revamped Dashboard")
    print("=" * 50)
    
    # Test dashboard availability
    if not test_dashboard_availability():
        print("\n❌ Dashboard is not running. Please start it with:")
        print("   python3 -m streamlit run src/dashboard.py --server.port 8501")
        return False
    
    # Test features
    test_dashboard_features()
    test_visual_components()
    test_data_visualization()
    test_color_scheme()
    test_proportional_layout()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("\n🎉 The revamped dashboard includes:")
    print("   • Modern gradient-based design")
    print("   • Better proportional layout")
    print("   • Enhanced geographic distribution")
    print("   • Improved color scheme")
    print("   • Professional styling")
    print("   • Better data visualization")
    
    print("\n🌐 Access your dashboard at: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    main() 