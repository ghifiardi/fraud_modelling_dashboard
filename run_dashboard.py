#!/usr/bin/env python3
"""
Dashboard Startup Script
Starts all monitoring components for the AI Fraud Detection System
"""

import subprocess
import sys
import os
import time
import threading
import signal
import argparse
from pathlib import Path

class DashboardManager:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def start_fastapi_server(self):
        """Start the FastAPI server."""
        print("🚀 Starting FastAPI server...")
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "src.api_server:app", 
                "--host", "0.0.0.0", 
                "--port", "8000",
                "--reload"
            ])
            self.processes.append(("FastAPI Server", process))
            print("✅ FastAPI server started on http://localhost:8000")
            return process
        except Exception as e:
            print(f"❌ Failed to start FastAPI server: {e}")
            return None
    
    def start_streamlit_dashboard(self):
        """Start the Streamlit dashboard."""
        print("🚀 Starting Streamlit dashboard...")
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "src/dashboard.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0"
            ])
            self.processes.append(("Streamlit Dashboard", process))
            print("✅ Streamlit dashboard started on http://localhost:8501")
            return process
        except Exception as e:
            print(f"❌ Failed to start Streamlit dashboard: {e}")
            return None
    
    def start_react_frontend(self):
        """Start the React frontend (if available)."""
        frontend_path = Path("frontend")
        if frontend_path.exists() and (frontend_path / "package.json").exists():
            print("🚀 Starting React frontend...")
            try:
                # Check if node_modules exists, if not run npm install
                if not (frontend_path / "node_modules").exists():
                    print("📦 Installing React dependencies...")
                    subprocess.run(["npm", "install"], cwd=frontend_path, check=True)
                
                process = subprocess.Popen([
                    "npm", "start"
                ], cwd=frontend_path)
                self.processes.append(("React Frontend", process))
                print("✅ React frontend started on http://localhost:3000")
                return process
            except Exception as e:
                print(f"❌ Failed to start React frontend: {e}")
                return None
        else:
            print("⚠️ React frontend not found, skipping...")
            return None
    
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        required_packages = [
            "streamlit", "fastapi", "uvicorn", "pandas", 
            "numpy", "scikit-learn", "plotly", "joblib"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Please install them using: pip install -r requirements.txt")
            return False
        
        print("✅ All required packages are installed")
        return True
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print("\n🛑 Shutting down dashboard components...")
            self.running = False
            self.stop_all()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def stop_all(self):
        """Stop all running processes."""
        for name, process in self.processes:
            try:
                print(f"🛑 Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                print(f"⚠️ Force killing {name}...")
                process.kill()
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
    
    def monitor_processes(self):
        """Monitor running processes and restart if needed."""
        while self.running:
            time.sleep(10)
            for i, (name, process) in enumerate(self.processes):
                if process.poll() is not None:
                    print(f"⚠️ {name} has stopped, restarting...")
                    # Restart logic could be added here
                    self.processes.pop(i)
                    break
    
    def run(self, components=None):
        """Run the dashboard with specified components."""
        print("🛡️ AI Fraud Detection Dashboard")
        print("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            return
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Start components based on selection
        if components is None or "api" in components:
            self.start_fastapi_server()
        
        if components is None or "streamlit" in components:
            self.start_streamlit_dashboard()
        
        if components is None or "react" in components:
            self.start_react_frontend()
        
        # Wait a bit for processes to start
        time.sleep(3)
        
        # Print dashboard URLs
        print("\n🌐 Dashboard URLs:")
        print("FastAPI Server: http://localhost:8000")
        print("API Documentation: http://localhost:8000/docs")
        print("Streamlit Dashboard: http://localhost:8501")
        print("React Frontend: http://localhost:3000")
        print("\nPress Ctrl+C to stop all services")
        
        # Monitor processes
        try:
            self.monitor_processes()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            self.stop_all()

def main():
    parser = argparse.ArgumentParser(description="Start AI Fraud Detection Dashboard")
    parser.add_argument(
        "--components", 
        nargs="+", 
        choices=["api", "streamlit", "react"],
        help="Components to start (default: all)"
    )
    parser.add_argument(
        "--install-deps", 
        action="store_true",
        help="Install dependencies before starting"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("📦 Installing Python dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        # Install React dependencies if frontend exists
        frontend_path = Path("frontend")
        if frontend_path.exists() and (frontend_path / "package.json").exists():
            print("📦 Installing React dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_path, check=True)
    
    # Start dashboard
    manager = DashboardManager()
    manager.run(args.components)

if __name__ == "__main__":
    main() 