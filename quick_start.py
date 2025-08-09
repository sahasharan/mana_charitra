#!/usr/bin/env python3
"""
Mana Charitra - One-Click Quick Start Script
Ultra-optimized setup for beginners with zero experience
"""

import os
import sys
import subprocess
import platform
import webbrowser
import time
from pathlib import Path

class QuickStart:
    """One-click setup and launch for Mana Charitra"""
    
    def __init__(self):
        self.project_name = "Mana Charitra"
        self.colors = {
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m', 
            'RED': '\033[91m',
            'BLUE': '\033[94m',
            'CYAN': '\033[96m',
            'MAGENTA': '\033[95m',
            'END': '\033[0m',
            'BOLD': '\033[1m'
        }
    
    def print_colored(self, message, color='END'):
        """Print colored terminal output"""
        if platform.system() == "Windows":
            # Windows CMD doesn't support ANSI colors well
            print(message)
        else:
            print(f"{self.colors.get(color, '')}{message}{self.colors['END']}")
    
    def print_banner(self):
        """Print attractive startup banner"""
        banner = f"""
{self.colors['CYAN']}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🚀 MANA CHARITRA - QUICK START 🚀                          ║
║  📚 మన చరిత్ర - తెలుగు చరిత్ర చాట్‌బాట్ 📚                    ║
║                                                              ║
║  ⚡ One-click setup for beginners                           ║
║  🔧 Automated installation and configuration                ║
║  🌟 Zero experience required!                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{self.colors['END']}
"""
        print(banner)
    
    def check_python(self):
        """Quick Python version check"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.print_colored(f"✅ Python {version.major}.{version.minor} detected - Perfect!", 'GREEN')
            return True
        else:
            self.print_colored(f"❌ Python {version.major}.{version.minor} - Need Python 3.8+", 'RED')
            self.print_colored("Download Python from: https://python.org/downloads", 'YELLOW')
            return False
    
    def install_requirements(self):
        """Install required packages with progress"""
        self.print_colored("📦 Installing required packages...", 'YELLOW')
        
        packages = [
            "streamlit>=1.28.0",
            "google-generativeai>=0.3.0", 
            "sentence-transformers>=2.2.2",
            "faiss-cpu>=1.7.4",
            "PyPDF2>=3.0.1",
            "pandas>=2.0.3",
            "numpy>=1.24.3"
        ]
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install packages
            for i, package in enumerate(packages, 1):
                self.print_colored(f"📥 Installing {package.split('>=')[0]}... ({i}/{len(packages)})", 'CYAN')
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
            
            self.print_colored("✅ All packages installed successfully!", 'GREEN')
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_colored(f"❌ Installation failed: {e}", 'RED')
            return False
    
    def create_app_file(self):
        """Create the main application file with error handling"""
        self.print_colored("📝 Creating application files...", 'YELLOW')
        
        # Check if app.py already exists
        if Path("app.py").exists():
            self.print_colored("✅ app.py already exists", 'GREEN')
            return True
        
        self.print_colored("❌ app.py not found!", 'RED')
        self.print_colored("Please ensure you have the main application file (app.py)", 'YELLOW')
        return False
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = ["data", "logs", "config", ".streamlit"]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        self.print_colored("✅ Project directories created", 'GREEN')
    
    def create_secrets_template(self):
        """Create secrets template file"""
        secrets_dir = Path(".streamlit")
        secrets_file = secrets_dir / "secrets.toml"
        
        if not secrets_file.exists():
            secrets_content = '''# Mana Charitra - API Keys Configuration
# Get your FREE Gemini API key from: https://makersuite.google.com

GEMINI_API_KEY = "your_gemini_api_key_here"

# Instructions:
# 1. Visit https://makersuite.google.com
# 2. Sign in with Google account
# 3. Click "Get API Key" → "Create API Key"
# 4. Replace "your_gemini_api_key_here" with your actual key
# 5. Save this file
'''
            
            with open(secrets_file, 'w') as f:
                f.write(secrets_content)
            
            self.print_colored("✅ Secrets template created", 'GREEN')
    
    def create_config_file(self):
        """Create Streamlit config file"""
        config_dir = Path(".streamlit")
        config_file = config_dir / "config.toml"
        
        if not config_file.exists():
            config_content = '''[theme]
primaryColor = "#2E8B57"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false
'''
            
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            self.print_colored("✅ Configuration file created", 'GREEN')
    
    def get_api_key_interactively(self):
        """Get API key from user with helpful instructions"""
        self.print_colored("\n🔑 API KEY SETUP", 'MAGENTA')
        self.print_colored("=" * 50, 'CYAN')
        
        print("\nTo use Mana Charitra, you need a FREE Gemini API key:")
        print("1. Visit: https://makersuite.google.com")
        print("2. Sign in with your Google account") 
        print("3. Click 'Get API Key' → 'Create API Key'")
        print("4. Copy the generated key")
        
        # Open browser automatically
        try:
            self.print_colored("\n🌐 Opening Google AI Studio in your browser...", 'YELLOW')
            webbrowser.open("https://makersuite.google.com")
            time.sleep(2)
        except:
            pass
        
        print("\nOptions:")
        print("1. Enter your API key now")
        print("2. Skip and enter later manually")
        
        choice = input("\nChoose option (1 or 2): ").strip()
        
        if choice == "1":
            api_key = input("\nPaste your Gemini API key here: ").strip()
            
            if api_key and len(api_key) > 20:  # Basic validation
                # Update secrets file
                secrets_file = Path(".streamlit/secrets.toml")
                
                # Read current content
                with open(secrets_file, 'r') as f:
                    content = f.read()
                
                # Replace placeholder with actual key
                updated_content = content.replace('your_gemini_api_key_here', api_key)
                
                # Write back
                with open(secrets_file, 'w') as f:
                    f.write(updated_content)
                
                self.print_colored("✅ API key saved successfully!", 'GREEN')
                return True
            else:
                self.print_colored("⚠️ Invalid API key. You can add it later.", 'YELLOW')
        
        self.print_colored("ℹ️ You can add your API key later in .streamlit/secrets.toml", 'BLUE')
        return False
    
    def test_installation(self):
        """Quick test of the installation"""
        self.print_colored("🔍 Testing installation...", 'YELLOW')
        
        try:
            # Test critical imports
            import streamlit
            import google.generativeai
            import sentence_transformers
            import faiss
            
            self.print_colored("✅ All components working correctly!", 'GREEN')
            return True
            
        except ImportError as e:
            self.print_colored(f"❌ Import error: {e}", 'RED')
            return False
    
    def create_launch_script(self):
        """Create platform-specific launch script"""
        if platform.system() == "Windows":
            script_content = '''@echo off
echo 🚀 Starting Mana Charitra...
echo 📚 మన చరిత్ర ప్రారంభమవుతోంది...
echo.
echo Opening in your default browser...
echo Press Ctrl+C to stop the application
echo.

python -m streamlit run app.py --server.port 8501

pause
'''
            script_name = "start_mana_charitra.bat"
        else:
            script_content = '''#!/bin/bash
echo "🚀 Starting Mana Charitra..."
echo "📚 మన చరిత్ర ప్రారంభమవుతోంది..."
echo
echo "Opening in your default browser..."
echo "Press Ctrl+C to stop the application"
echo

python3 -m streamlit run app.py --server.port 8501

echo "Press any key to exit..."
read -n 1 -s
'''
            script_name = "start_mana_charitra.sh"
        
        with open(script_name, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        if platform.system() != "Windows":
            os.chmod(script_name, 0o755)
        
        self.print_colored(f"✅ Launch script created: {script_name}", 'GREEN')
        return script_name
    
    def launch_application(self):
        """Launch the Streamlit application"""
        self.print_colored("\n🚀 LAUNCHING MANA CHARITRA", 'MAGENTA')
        self.print_colored("=" * 50, 'CYAN')
        
        # Check if app.py exists
        if not Path("app.py").exists():
            self.print_colored("❌ app.py not found! Cannot launch.", 'RED')
            return False
        
        try:
            self.print_colored("🌟 Starting the Telugu History Chatbot...", 'YELLOW')
            self.print_colored("📱 Opening in your browser automatically...", 'CYAN')
            
            # Launch Streamlit
            subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.port", "8501",
                "--server.headless", "true"
            ])
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Open browser
            try:
                webbrowser.open("http://localhost:8501")
            except:
                pass
            
            self.print_colored("✅ Application launched successfully!", 'GREEN')
            self.print_colored("🌐 URL: http://localhost:8501", 'BLUE')
            
            return True
            
        except Exception as e:
            self.print_colored(f"❌ Launch failed: {e}", 'RED')
            return False
    
    def print_success_message(self):
        """Print final success message with instructions"""
        success_msg = f"""
{self.colors['GREEN']}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🎉 SUCCESS! MANA CHARITRA IS READY! 🎉                    ║
║                                                              ║
║  📱 Your Telugu History Chatbot is now running at:          ║
║  🌐 http://localhost:8501                                    ║
║                                                              ║
║  💡 What you can do now:                                     ║
║  • Ask questions about Telugu history                       ║
║  • Upload PDF documents                                      ║
║  • Share local stories and folklore                         ║
║  • View statistics and analytics                            ║
║                                                              ║
║  📝 Example questions to try:                               ║
║  • వరంగల్ కోట చరిత్ర చెప్పండి                              ║
║  • తిరుమల దేవాలయ ప్రాముఖ్యత ఏమిటి?                       ║
║  • కాకతీయ రాజవంశం గురించి తెలియజేయండి                    ║
║                                                              ║
║  🛑 To stop: Press Ctrl+C in this terminal                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{self.colors['END']}
"""
        print(success_msg)
    
    def print_troubleshooting(self):
        """Print troubleshooting information"""
        troubleshooting = f"""
{self.colors['YELLOW']}🔧 TROUBLESHOOTING TIPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ Application won't start?
  → Check if Python 3.8+ is installed
  → Ensure all packages are installed
  → Try: python -m streamlit run app.py

❓ API errors?
  → Check your Gemini API key in .streamlit/secrets.toml
  → Get free key from: https://makersuite.google.com
  
❓ Telugu text not showing?
  → Your browser should auto-load Telugu fonts
  → Try refreshing the page

❓ Need help?
  → Check README.md for detailed instructions
  → Visit GitHub repository for support

💡 For next time, just run:
  → Windows: start_mana_charitra.bat
  → Linux/Mac: ./start_mana_charitra.sh{self.colors['END']}
"""
        print(troubleshooting)
    
    def run_quick_start(self):
        """Main quick start process"""
        self.print_banner()
        
        # Step 1: Check Python
        if not self.check_python():
            input("Press Enter to exit...")
            return
        
        # Step 2: Install packages
        self.print_colored("\n📦 INSTALLING PACKAGES", 'MAGENTA')
        self.print_colored("=" * 50, 'CYAN')
        
        if not self.install_requirements():
            self.print_colored("❌ Package installation failed!", 'RED')
            input("Press Enter to exit...")
            return
        
        # Step 3: Setup project
        self.print_colored("\n⚙️ SETTING UP PROJECT", 'MAGENTA')
        self.print_colored("=" * 50, 'CYAN')
        
        self.setup_directories()
        self.create_secrets_template()
        self.create_config_file()
        
        # Step 4: Check if main app exists
        if not self.create_app_file():
            self.print_colored("❌ Main application file missing!", 'RED')
            self.print_colored("Please ensure app.py is in the current directory", 'YELLOW')
            input("Press Enter to exit...")
            return
        
        # Step 5: API Key setup
        api_key_set = self.get_api_key_interactively()
        
        # Step 6: Test installation
        if not self.test_installation():
            self.print_colored("❌ Installation test failed!", 'RED')
            self.print_troubleshooting()
            input("Press Enter to exit...")
            return
        
        # Step 7: Create launch script
        self.print_colored("\n🔧 CREATING SHORTCUTS", 'MAGENTA')
        self.print_colored("=" * 50, 'CYAN')
        script_name = self.create_launch_script()
        
        # Step 8: Launch application
        if api_key_set:
            launch_now = input("\n🚀 Launch Mana Charitra now? (y/n): ").strip().lower()
            if launch_now in ['y', 'yes', '']:
                if self.launch_application():
                    self.print_success_message()
                    
                    # Keep the terminal open
                    try:
                        self.print_colored("💡 Keep this terminal open while using the app", 'YELLOW')
                        self.print_colored("Press Ctrl+C to stop the application", 'CYAN')
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        self.print_colored("\n👋 Stopping Mana Charitra... Thank you!", 'GREEN')
                else:
                    self.print_troubleshooting()
            else:
                self.print_colored(f"✅ Setup complete! Run {script_name} to start later.", 'GREEN')
        else:
            self.print_colored("⚠️ Add your API key first, then run the launch script", 'YELLOW')
            self.print_colored(f"Launch script created: {script_name}", 'GREEN')
        
        input("\nPress Enter to exit...")

def main():
    """Main entry point"""
    try:
        quick_start = QuickStart()
        quick_start.run_quick_start()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the troubleshooting guide or seek help.")

if __name__ == "__main__":
    main()