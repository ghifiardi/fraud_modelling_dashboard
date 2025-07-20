#!/usr/bin/env python3
"""
Quick Start Script for Blockchain Fraud Detection Dashboard
Runs the complete blockchain fraud detection system
"""

import sys
import os
import subprocess
import time
import threading
import signal
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'flask',
        'flask-cors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def start_api_server():
    """Start the blockchain API server"""
    print("🚀 Starting Blockchain API Server...")
    
    try:
        # Change to src directory
        src_dir = Path(__file__).parent / "src"
        os.chdir(src_dir)
        
        # Start API server
        api_process = subprocess.Popen([
            sys.executable, "blockchain_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:5001/health", timeout=5)
            if response.status_code == 200:
                print("✅ Blockchain API Server is running on http://localhost:5001")
                return api_process
            else:
                print("❌ API server responded with error")
                return None
        except requests.exceptions.RequestException:
            print("❌ Could not connect to API server")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start API server: {str(e)}")
        return None

def start_dashboard():
    """Start the blockchain dashboard"""
    print("🎯 Starting Blockchain Dashboard...")
    
    try:
        # Change back to project root
        project_root = Path(__file__).parent
        os.chdir(project_root)
        
        # Start dashboard
        dashboard_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "blockchain_fraud_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
        print("✅ Blockchain Dashboard is starting...")
        print("🌐 Dashboard will be available at: http://localhost:8501")
        return dashboard_process
        
    except Exception as e:
        print(f"❌ Failed to start dashboard: {str(e)}")
        return None

def run_test_suite():
    """Run the blockchain test suite"""
    print("🧪 Running Blockchain Test Suite...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_blockchain_integration.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed!")
            print("Test output:")
            print(result.stdout)
            print("Test errors:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to run tests: {str(e)}")
        return False

def cleanup(api_process, dashboard_process):
    """Cleanup processes on exit"""
    print("\n🛑 Shutting down...")
    
    if dashboard_process:
        dashboard_process.terminate()
        print("✅ Dashboard stopped")
    
    if api_process:
        api_process.terminate()
        print("✅ API server stopped")

def main():
    """Main function"""
    print("🔗 Blockchain Fraud Detection Dashboard")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Ask user what to do
    print("\n🎯 What would you like to do?")
    print("1. Run complete system (API + Dashboard)")
    print("2. Run tests only")
    print("3. Start API server only")
    print("4. Start dashboard only")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    api_process = None
    dashboard_process = None
    
    try:
        if choice == "1":
            # Run complete system
            print("\n🚀 Starting complete blockchain fraud detection system...")
            
            # Start API server
            api_process = start_api_server()
            if not api_process:
                print("❌ Failed to start API server. Exiting.")
                return 1
            
            # Start dashboard
            dashboard_process = start_dashboard()
            if not dashboard_process:
                print("❌ Failed to start dashboard. Exiting.")
                return 1
            
            print("\n🎉 System is running!")
            print("📊 Dashboard: http://localhost:8501")
            print("🔌 API Server: http://localhost:5001")
            print("\nPress Ctrl+C to stop the system...")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        
        elif choice == "2":
            # Run tests only
            success = run_test_suite()
            return 0 if success else 1
        
        elif choice == "3":
            # Start API server only
            api_process = start_api_server()
            if not api_process:
                print("❌ Failed to start API server.")
                return 1
            
            print("\n🔌 API Server is running on http://localhost:5001")
            print("Press Ctrl+C to stop...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        
        elif choice == "4":
            # Start dashboard only
            dashboard_process = start_dashboard()
            if not dashboard_process:
                print("❌ Failed to start dashboard.")
                return 1
            
            print("\n📊 Dashboard is running on http://localhost:8501")
            print("Note: API server is not running. Some features may not work.")
            print("Press Ctrl+C to stop...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            return 1
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    finally:
        cleanup(api_process, dashboard_process)
    
    print("👋 Goodbye!")
    return 0

if __name__ == "__main__":
    exit(main()) 