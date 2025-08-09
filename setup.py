#!/usr/bin/env python3
"""
Mana Charitra - Automated Setup Script
Advanced setup with error handling and optimizations
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path
import json

class ManaCharitraSetup:
    """Advanced setup manager with error handling and optimizations"""
    
    def __init__(self):
        self.project_name = "mana_charitra"
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "data"
        self.config_dir = self.base_dir / "config"
        self.logs_dir = self.base_dir / "logs"
        
        # Colors for terminal output
        self.COLORS = {
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'RED': '\033[91m',
            'BLUE': '\033[94m',
            'CYAN': '\033[96m',
            'END': '\033[0m',
            'BOLD': '\033[1m'
        }
    
    def print_colored(self, message: str, color: str = 'END') -> None:
        """Print colored message to terminal"""
        print(f"{self.COLORS.get(color, '')}{message}{self.COLORS['END']}")
    
    def print_header(self) -> None:
        """Print setup header"""
        self.print_colored("=" * 60, 'CYAN')
        self.print_colored("🚀 MANA CHARITRA - TELUGU HISTORY CHATBOT SETUP 🚀", 'BOLD')
        self.print_colored("📚 మన చరిత్ర - తెలుగు చరిత్ర చాట్‌బాట్ సెటప్ 📚", 'BLUE')
        self.print_colored("=" * 60, 'CYAN')
        print()
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        self.print_colored("🐍 Checking Python version...", 'YELLOW')
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.print_colored(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!", 'GREEN')
            return True
        else:
            self.print_colored(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+", 'RED')
            return False
    
    def create_directories(self) -> None:
        """Create necessary directories"""
        self.print_colored("📁 Creating project directories...", 'YELLOW')
        
        directories = [self.data_dir, self.config_dir, self.logs_dir]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            self.print_colored(f"✅ Created: {directory}", 'GREEN')
    
    def install_dependencies(self) -> bool:
        """Install required Python packages"""
        self.print_colored("📦 Installing dependencies...", 'YELLOW')
        
        try:
            # Upgrade pip first
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install requirements
            requirements = [
                "streamlit>=1.28.0",
                "google-generativeai>=0.3.0",
                "sentence-transformers>=2.2.2",
                "faiss-cpu>=1.7.4",
                "PyPDF2>=3.0.1",
                "pandas>=2.0.3",
                "numpy>=1.24.3"
            ]
            
            for requirement in requirements:
                self.print_colored(f"📥 Installing {requirement}...", 'CYAN')
                subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
                self.print_colored(f"✅ Installed {requirement}", 'GREEN')
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_colored(f"❌ Failed to install dependencies: {e}", 'RED')
            return False
    
    def create_config_files(self) -> None:
        """Create configuration files"""
        self.print_colored("⚙️ Creating configuration files...", 'YELLOW')
        
        # Streamlit config
        streamlit_config = """
[theme]
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
"""
        
        streamlit_dir = self.base_dir / ".streamlit"
        streamlit_dir.mkdir(exist_ok=True)
        
        with open(streamlit_dir / "config.toml", 'w') as f:
            f.write(streamlit_config)
        
        # Secrets template
        secrets_template = """
# Mana Charitra Configuration
# Add your Gemini API key here

GEMINI_API_KEY = "your_gemini_api_key_here"

# Optional: Add other API keys
# OPENAI_API_KEY = "your_openai_key_here"
# HUGGINGFACE_API_KEY = "your_hf_key_here"
"""
        
        with open(streamlit_dir / "secrets.toml", 'w') as f:
            f.write(secrets_template)
        
        self.print_colored("✅ Configuration files created", 'GREEN')
        self.print_colored("⚠️  Remember to add your GEMINI_API_KEY in .streamlit/secrets.toml", 'YELLOW')
    
    def create_env_file(self) -> None:
        """Create environment file"""
        self.print_colored("🌍 Creating environment file...", 'YELLOW')
        
        env_content = """# Mana Charitra Environment Variables
# Copy this to .env and add your actual API keys

GEMINI_API_KEY=your_gemini_api_key_here
PYTHONPATH=.
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
"""
        
        with open(self.base_dir / ".env.example", 'w') as f:
            f.write(env_content)
        
        self.print_colored("✅ Environment template created (.env.example)", 'GREEN')
    
    def create_launch_scripts(self) -> None:
        """Create launch scripts for different platforms"""
        self.print_colored("🚀 Creating launch scripts...", 'YELLOW')
        
        # Windows batch file
        windows_script = """@echo off
echo Starting Mana Charitra - Telugu History Chatbot...
echo మన చరిత్ర ప్రారంభమవుతోంది...

python -m streamlit run app.py --server.port 8501 --server.headless true

pause
"""
        
        with open(self.base_dir / "run_windows.bat", 'w') as f:
            f.write(windows_script)
        
        # Linux/Mac shell script
        unix_script = """#!/bin/bash
echo "🚀 Starting Mana Charitra - Telugu History Chatbot..."
echo "📚 మన చరిత్ర ప్రారంభమవుతోంది..."

python3 -m streamlit run app.py --server.port 8501 --server.headless true

echo "Press any key to exit..."
read -n 1 -s
"""
        
        with open(self.base_dir / "run_unix.sh", 'w') as f:
            f.write(unix_script)
        
        # Make Unix script executable
        if platform.system() != "Windows":
            os.chmod(self.base_dir / "run_unix.sh", 0o755)
        
        self.print_colored("✅ Launch scripts created", 'GREEN')
    
    def create_docker_files(self) -> None:
        """Create Docker files for containerized deployment"""
        self.print_colored("🐳 Creating Docker files...", 'YELLOW')
        
        dockerfile = """# Mana Charitra - Telugu History Chatbot Docker Image
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p data logs config

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
        
        with open(self.base_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        
        # Docker Compose
        docker_compose = """version: '3.8'

services:
  mana-charitra:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
        
        with open(self.base_dir / "docker-compose.yml", 'w') as f:
            f.write(docker_compose)
        
        self.print_colored("✅ Docker files created", 'GREEN')
    
    def create_deployment_configs(self) -> None:
        """Create deployment configurations"""
        self.print_colored("☁️ Creating deployment configurations...", 'YELLOW')
        
        # Heroku Procfile
        with open(self.base_dir / "Procfile", 'w') as f:
            f.write("web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0\n")
        
        # Railway deployment
        railway_config = {
            "build": {
                "builder": "NIXPACKS"
            },
            "deploy": {
                "startCommand": "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0",
                "healthcheckPath": "/_stcore/health",
                "healthcheckTimeout": 100,
                "restartPolicyType": "ON_FAILURE",
                "restartPolicyMaxRetries": 10
            }
        }
        
        with open(self.base_dir / "railway.json", 'w') as f:
            json.dump(railway_config, f, indent=2)
        
        # Streamlit Cloud config
        streamlit_cloud_config = """# Streamlit Cloud Deployment Configuration
# Place this in your GitHub repository root

[theme]
primaryColor = "#2E8B57"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
enableCORS = false
enableXsrfProtection = false
"""
        
        with open(self.base_dir / "streamlit_cloud_config.toml", 'w') as f:
            f.write(streamlit_cloud_config)
        
        self.print_colored("✅ Deployment configurations created", 'GREEN')
    
    def create_readme(self) -> None:
        """Create comprehensive README file"""
        self.print_colored("📝 Creating README file...", 'YELLOW')
        
        readme_content = """# 📚 Mana Charitra - Telugu History Chatbot
*మన చరిత్ర - తెలుగు చరిత్ర చాట్‌బాట్*

## 🎯 Overview
Mana Charitra is an AI-powered chatbot that provides historical and cultural information about places in Andhra Pradesh and Telangana in Telugu language. Users can ask questions, contribute stories, and upload documents to build a comprehensive knowledge base.

## ✨ Features
- 💬 Chat in Telugu about historical places
- 📤 Upload PDF/text documents
- ✍️ Contribute local stories and folklore
- 🔍 Advanced vector search with RAG
- 📊 Analytics and statistics
- 🎨 Modern Telugu-friendly UI

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Gemini API key (free from Google AI Studio)

### Installation

1. **Clone or download this project**
```bash
git clone <repository-url>
cd mana_charitra
```

2. **Run the setup script**
```bash
python setup.py
```

3. **Add your Gemini API key**
Edit `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_actual_api_key_here"
```

4. **Launch the application**

**Windows:**
```bash
run_windows.bat
```

**Linux/Mac:**
```bash
./run_unix.sh
```

**Manual:**
```bash
streamlit run app.py
```

## 🔧 Configuration

### API Keys
1. Get a free Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add it to `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_key_here"
```

### Environment Variables
Copy `.env.example` to `.env` and configure:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t mana-charitra .
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key mana-charitra
```

## ☁️ Cloud Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add `GEMINI_API_KEY` in secrets

### Railway
1. Connect GitHub repository
2. Add environment variable: `GEMINI_API_KEY`
3. Deploy automatically

### Heroku
```bash
git init
heroku create your-app-name
heroku config:set GEMINI_API_KEY=your_key
git add .
git commit -m "Initial commit"
git push heroku main
```

## 📁 Project Structure
```
mana_charitra/
├── app.py                 # Main application
├── setup.py              # Automated setup
├── requirements.txt      # Python dependencies
├── data/                 # Database and vector store
├── config/              # Configuration files
├── logs/                # Application logs
├── .streamlit/          # Streamlit configuration
│   ├── config.toml
│   └── secrets.toml
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose
├── Procfile            # Heroku deployment
├── railway.json        # Railway deployment
└── README.md           # This file
```

## 🎮 Usage

### Chat Interface
1. Open the application
2. Type questions in Telugu about historical places
3. Example: "వరంగల్ కోట చరిత్ర చెప్పండి"

### Upload Documents
1. Go to "📤 దస్త్రం అప్‌లోడ్" tab
2. Upload PDF or text files
3. Documents are automatically processed and added to knowledge base

### Contribute Stories
1. Go to "✍️ మీ కథ" tab
2. Share local stories and folklore
3. Stories are added to the community knowledge base

### View Statistics
1. Go to "📊 గణాంకాలు" tab
2. See analytics and popular places
3. Track recent additions

## 🔧 Advanced Configuration

### Vector Store Settings
Edit `config/config.py` to customize:
```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Change embedding model
VECTOR_DIM = 384                      # Vector dimensions
CHUNK_SIZE = 512                      # Text chunk size
MAX_TOKENS = 8192                     # Max response tokens
```

### Database Configuration
SQLite database is automatically created in `data/mana_charitra.db`
- Optimized with WAL mode for better performance
- Automatic indexing for fast queries
- Built-in duplicate prevention

## 🛠️ Development

### Project Architecture
- **Frontend:** Streamlit with Telugu font support
- **Backend:** Google Gemini Pro for LLM
- **Vector Store:** FAISS with sentence-transformers
- **Database:** SQLite with optimizations
- **Caching:** Smart LRU cache for performance

### Key Components
1. `OptimizedVectorStore` - FAISS-based similarity search
2. `OptimizedDatabase` - SQLite with connection pooling
3. `ManaCharitra` - Main chatbot class
4. `SmartCache` - Memory-efficient caching
5. `OptimizedDocumentProcessor` - PDF and text processing

### Performance Optimizations
- Batch embedding processing
- LRU caching for responses
- Connection pooling for database
- Parallel document processing
- Memory-efficient vector operations

## 🧪 Testing

### Manual Testing
1. Test chat functionality with Telugu queries
2. Upload sample PDF documents
3. Add test stories through the interface
4. Verify statistics are updating

### Example Test Queries
```
వరంగల్ కోట చరిత్ర చెప్పండి
తిరుమల దేవాలయ ప్రాముఖ్యత ఏమిటి?
కాకతీయ రాజవంశం గురించి తెలియజేయండి
హైదరాబాద్ చరిత్ర చెప్పండి
```

## 📊 Monitoring

### Logs
Application logs are stored in `logs/` directory:
- `app.log` - General application logs
- `error.log` - Error logs
- `performance.log` - Performance metrics

### Health Checks
- Built-in Streamlit health endpoint: `/_stcore/health`
- Database connectivity checks
- Vector store validation
- API key verification

## 🔒 Security

### API Key Management
- Store API keys in secrets, not code
- Use environment variables for deployment
- Never commit secrets to version control

### Data Privacy
- All data stored locally by default
- No external data transmission except API calls
- User contributions are stored securely

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**2. FAISS Installation Issues**
```bash
# Use CPU version
pip install faiss-cpu
# For GPU (if available)
pip install faiss-gpu
```

**3. Telugu Font Issues**
- Install Noto Sans Telugu font on your system
- Browser should automatically load web fonts

**4. API Key Errors**
- Verify Gemini API key is correct
- Check API quotas and limits
- Ensure key has proper permissions

**5. Memory Issues**
```python
# Reduce cache sizes in config
MAX_CACHE_SIZE = 500
CHUNK_SIZE = 256
```

**6. Performance Issues**
- Clear vector store: Delete `data/vector_store.faiss`
- Restart application
- Check available memory

### Debug Mode
Set debug mode in Streamlit:
```bash
streamlit run app.py --logger.level=debug
```

## 🤝 Contributing

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

### Improving Telugu Content
1. Add more historical data
2. Improve Telugu language processing
3. Add regional variations
4. Contribute folklore and stories

## 📞 Support

### Community
- GitHub Issues for bug reports
- Discussions for feature requests
- Wiki for documentation

### Professional Support
- Commercial licensing available
- Custom development services
- Training and consultation

## 📜 License

This project is open source under MIT License.
See `LICENSE` file for details.

## 🙏 Acknowledgments

- Google Gemini for LLM capabilities
- Streamlit for the amazing framework
- FAISS for vector similarity search
- Telugu Wikipedia for initial data
- Community contributors

## 🔄 Updates

### Version History
- v1.0.0 - Initial release with core features
- v1.1.0 - Added user contributions
- v1.2.0 - Performance optimizations
- v1.3.0 - Docker and cloud deployment

### Roadmap
- [ ] Voice input/output in Telugu
- [ ] Mobile app version
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] API endpoints for developers

---

**మన చరిత్రను కలిసి నిర్మిద్దాం! 🇮🇳**

*Built with ❤️ for Telugu heritage preservation*
"""
        
        with open(self.base_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        self.print_colored("✅ README.md created", 'GREEN')
    
    def create_gitignore(self) -> None:
        """Create .gitignore file"""
        self.print_colored("📝 Creating .gitignore...", 'YELLOW')
        
        gitignore_content = """# Mana Charitra - Git Ignore File

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.coverage
.pytest_cache/

# Environments
.env
.venv
env/
venv/
ENV/

# Streamlit
.streamlit/secrets.toml

# Database files
*.db
*.sqlite
*.sqlite3

# Vector store files
*.faiss
*.pkl
*.pickle

# Logs
logs/
*.log

# Data files (keep structure but not content)
data/*.faiss
data/*.pkl
data/*.db

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
.cache/

# API keys and secrets
config/secrets.json
.env.local
.env.production

# Model cache
sentence_transformers_cache/
transformers_cache/
"""
        
        with open(self.base_dir / ".gitignore", 'w') as f:
            f.write(gitignore_content)
        
        self.print_colored("✅ .gitignore created", 'GREEN')
    
    def verify_installation(self) -> bool:
        """Verify the installation is working"""
        self.print_colored("🔍 Verifying installation...", 'YELLOW')
        
        try:
            # Test imports
            import streamlit
            import google.generativeai
            import sentence_transformers
            import faiss
            import PyPDF2
            import pandas
            import numpy
            
            self.print_colored("✅ All required packages are installed", 'GREEN')
            
            # Test data directory structure
            required_dirs = [self.data_dir, self.config_dir, self.logs_dir]
            for directory in required_dirs:
                if directory.exists():
                    self.print_colored(f"✅ Directory exists: {directory}", 'GREEN')
                else:
                    self.print_colored(f"❌ Missing directory: {directory}", 'RED')
                    return False
            
            return True
            
        except ImportError as e:
            self.print_colored(f"❌ Import error: {e}", 'RED')
            return False
    
    def get_api_key_instructions(self) -> None:
        """Display API key setup instructions"""
        self.print_colored("🔑 API Key Setup Instructions", 'CYAN')
        print()
        self.print_colored("1. Visit Google AI Studio:", 'YELLOW')
        self.print_colored("   https://makersuite.google.com/app/apikey", 'BLUE')
        print()
        self.print_colored("2. Create a new API key (it's FREE!)", 'YELLOW')
        print()
        self.print_colored("3. Copy the API key", 'YELLOW')
        print()
        self.print_colored("4. Add it to .streamlit/secrets.toml:", 'YELLOW')
        self.print_colored('   GEMINI_API_KEY = "your_actual_api_key_here"', 'BLUE')
        print()
        self.print_colored("⚠️  Keep your API key secure and never share it!", 'RED')
    
    def run_setup(self) -> None:
        """Run the complete setup process"""
        self.print_header()
        
        # Check Python version
        if not self.check_python_version():
            self.print_colored("❌ Setup failed: Incompatible Python version", 'RED')
            return
        
        # Create directories
        self.create_directories()
        
        # Install dependencies
        if not self.install_dependencies():
            self.print_colored("❌ Setup failed: Dependency installation failed", 'RED')
            return
        
        # Create configuration files
        self.create_config_files()
        self.create_env_file()
        self.create_launch_scripts()
        self.create_docker_files()
        self.create_deployment_configs()
        self.create_readme()
        self.create_gitignore()
        
        # Verify installation
        if self.verify_installation():
            self.print_colored("🎉 Setup completed successfully!", 'GREEN')
            print()
            self.get_api_key_instructions()
            print()
            self.print_colored("🚀 To start the application:", 'CYAN')
            if platform.system() == "Windows":
                self.print_colored("   run_windows.bat", 'BLUE')
            else:
                self.print_colored("   ./run_unix.sh", 'BLUE')
            self.print_colored("   OR", 'YELLOW')
            self.print_colored("   streamlit run app.py", 'BLUE')
        else:
            self.print_colored("❌ Setup verification failed", 'RED')

if __name__ == "__main__":
    setup = ManaCharitraSetup()
    setup.run_setup()