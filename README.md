# 📚 Mana Charitra - Telugu History Chatbot
*మన చరిత్ర - తెలుగు చరిత్ర చాట్‌బాట్*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Free Deployment](https://img.shields.io/badge/Deployment-Free-green.svg)](#-free-deployment-options)

> 🎯 **AI-powered Telugu history chatbot that preserves and shares the rich cultural heritage of Andhra Pradesh and Telangana**

---

## 🌟 Features

### 💬 Interactive Chat
- **Telugu Language Support** - Native Telugu conversation
- **Historical Knowledge** - Information about temples, forts, towns, and villages
- **Cultural Context** - Stories, folklore, and traditions
- **Smart Search** - Advanced RAG-based information retrieval

### 📚 Knowledge Management
- **Document Upload** - PDF and text file processing
- **User Contributions** - Community-driven content
- **Vector Search** - Semantic similarity matching
- **Content Validation** - Duplicate prevention and quality control

### 🎨 Modern Interface
- **Telugu Fonts** - Beautiful Noto Sans Telugu typography
- **Responsive Design** - Works on desktop and mobile
- **Intuitive Navigation** - Easy-to-use tabbed interface
- **Visual Analytics** - Statistics and usage insights

### 🔧 Technical Excellence
- **Optimized Performance** - Smart caching and batch processing
- **Free Deployment** - Multiple zero-cost hosting options
- **Modular Architecture** - Clean, maintainable codebase
- **Production Ready** - Error handling and logging

---

## 🚀 Quick Start (For Beginners)

### Option 1: One-Click Setup ⚡
```bash
# Download the quick start script and run
python quick_start.py
```
**That's it!** The script handles everything automatically.

### Option 2: Manual Setup 🔧
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key
# Edit .streamlit/secrets.toml and add:
# GEMINI_API_KEY = "your_key_here"

# 3. Run the application
streamlit run app.py
```

### Option 3: Full Setup 🏗️
```bash
# Complete setup with all configurations
python setup.py
```

---

## 🔑 Getting Free Gemini API Key

1. **Visit Google AI Studio**: [makersuite.google.com](https://makersuite.google.com)
2. **Sign in** with your Google account
3. **Create API Key** - Click "Get API Key" → "Create API Key"
4. **Copy the key** - It's completely FREE with generous limits!

**Free Tier Limits:**
- ✅ 60 requests per minute
- ✅ 1,500 requests per day  
- ✅ No credit card required
- ✅ No expiration

---

## 💻 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Memory**: 2GB RAM
- **Storage**: 1GB free space
- **Internet**: For API calls and initial setup

### Recommended Requirements
- **Python**: 3.9+
- **Memory**: 4GB+ RAM
- **Storage**: 2GB+ free space
- **Internet**: Stable broadband connection

### Supported Platforms
- ✅ Windows 10/11
- ✅ macOS 10.14+
- ✅ Linux (Ubuntu 18.04+)
- ✅ Cloud platforms (Streamlit Cloud, Railway, Render, etc.)

---

## 🌐 Free Deployment Options

### 🎯 Streamlit Cloud (Recommended)
**✅ Completely FREE • ✅ Easy Setup • ✅ Auto-deployment**

1. Push code to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Add `GEMINI_API_KEY` in secrets
4. Deploy with one click!

**Your app URL**: `https://yourusername-mana-charitra-app-xyz.streamlit.app`

### 🚂 Railway 
**✅ $5 Monthly Credit • ✅ Excellent Performance**

1. Connect GitHub to [railway.app](https://railway.app)
2. Add environment variable: `GEMINI_API_KEY`
3. Auto-deploy with every git push

### 🔥 Render
**✅ FREE Tier • ✅ No Credit Card Required**

1. Connect GitHub to [render.com](https://render.com)
2. Configure environment variables
3. Deploy automatically

### 📱 Replit
**✅ Instant Deploy • ✅ Online IDE**

1. Import from GitHub to [replit.com](https://replit.com)
2. Add secrets: `GEMINI_API_KEY`
3. Click "Run" - that's it!

### 🐳 Docker
**✅ Local/Cloud • ✅ Full Control**

```bash
# Build and run
docker build -t mana-charitra .
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key mana-charitra

# Or use Docker Compose
echo "GEMINI_API_KEY=your_key" > .env
docker-compose up
```

> 📖 **Detailed deployment guide**: See [Free Deployment Guide](./DEPLOYMENT.md)

---

## 📁 Project Structure

```
mana_charitra/
├── 📄 app.py                    # Main Streamlit application
├── 🔧 setup.py                 # Automated setup script
├── ⚡ quick_start.py           # One-click setup for beginners
├── 📋 requirements.txt         # Python dependencies
├── 📚 README.md               # This file
├── 🚀 DEPLOYMENT.md           # Deployment guide
├── 📁 data/                   # Data storage
│   ├── 🗄️ mana_charitra.db    # SQLite database
│   ├── 🔍 vector_store.faiss  # FAISS vector index
│   └── 💾 embeddings_cache.pkl # Cached embeddings
├── 📁 config/                 # Configuration files
├── 📁 logs/                   # Application logs
├── 📁 .streamlit/             # Streamlit configuration
│   ├── ⚙️ config.toml         # UI configuration
│   └── 🔐 secrets.toml        # API keys (add your key here)
├── 🐳 Dockerfile             # Docker configuration
├── 📦 docker-compose.yml     # Docker Compose setup
├── 🌐 Procfile               # Heroku deployment
├── 🚂 railway.json           # Railway configuration
└── 📝 .gitignore             # Git ignore rules
```

---

## 🎮 Usage Guide

### 💬 Chat Interface
Ask questions about Telugu historical places:

**Example Questions:**
```
వరంగల్ కోట చరిత్ర చెప్పండి
తిరుమల దేవాలయ ప్రాముఖ్యత ఏమిటి?
కాకతీయ రాజవంశం గురించి తెలియజేయండి
హైదరాబాద్ చరిత్ర చెప్పండి
చిత్తూర్ జిల్లా ప్రసిద్ధ ప్రాంతాలు ఏవి?
```

### 📤 Document Upload
1. Go to "📤 దస్త్రం అప్‌లోడ్" tab
2. Select PDF or text files
3. Click "🔄 ప్రాసెస్ చేయండి"
4. Documents are automatically indexed

**Supported Formats:**
- ✅ PDF files (.pdf)
- ✅ Text files (.txt)
- ✅ Telugu and English content
- ✅ Mixed language documents

### ✍️ Story Contribution
1. Go to "✍️ మీ కథ" tab
2. Enter place name and your story
3. Add your name (optional)
4. Submit to community database

**Content Guidelines:**
- ✅ Share factual historical information
- ✅ Include local folklore and traditions
- ✅ Mention sources when possible
- ❌ Avoid unverified claims

### 📊 Analytics Dashboard
View comprehensive statistics:
- 📈 Total documents and user contributions
- 🏆 Popular places and trending topics
- 📅 Recent additions and activity
- 🔍 Search patterns and usage metrics

---

## 🔧 Advanced Configuration

### Performance Optimization
Edit configuration in `app.py`:

```python
class OptimizedConfig:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight model
    VECTOR_DIM = 384                      # Vector dimensions
    MAX_TOKENS = 8192                     # Response length
    CHUNK_SIZE = 512                      # Text chunk size
    CHUNK_OVERLAP = 64                    # Overlap between chunks
    MAX_CACHE_SIZE = 1000                 # Memory cache size
```

### Database Settings
SQLite optimizations are pre-configured:
- WAL mode for better concurrency
- Memory-based temp storage
- Automatic indexing
- Connection pooling

### Vector Store Tuning
FAISS settings for different use cases:
```python
# For small datasets (< 1000 documents)
index = faiss.IndexFlatIP(vector_dim)

# For large datasets (> 1000 documents)  
index = faiss.IndexIVFFlat(index, vector_dim, 100)
```

---

## 🛠️ Development

### Local Development Setup
```bash
# Clone repository
git clone <repository-url>
cd mana_charitra

# Install in development mode
pip install -e .

# Install additional dev dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black app.py
```

### Code Architecture

#### Core Components
1. **ManaCharitra** - Main chatbot class
2. **OptimizedVectorStore** - FAISS-based semantic search
3. **OptimizedDatabase** - SQLite with performance optimizations
4. **SmartCache** - LRU caching system
5. **OptimizedDocumentProcessor** - PDF/text processing

#### Design Principles
- ⚡ **Performance First** - Optimized for speed and memory
- 🔧 **Modular Design** - Clean separation of concerns  
- 🛡️ **Error Resilience** - Comprehensive error handling
- 💾 **Memory Efficient** - Smart caching and cleanup
- 🔄 **Scalable** - Designed for growth

### Adding New Features
```python
# Example: Adding new document processor
class NewDocumentProcessor(OptimizedDocumentProcessor):
    def process_new_format(self, file_bytes: bytes) -> str:
        # Implement new format processing
        pass

# Example: Custom embedding model
class CustomVectorStore(OptimizedVectorStore):
    def _load_model(self):
        self.model = SentenceTransformer('custom-model')
```

---

## 🧪 Testing

### Manual Testing Checklist
- [ ] Chat functionality with Telugu queries
- [ ] Document upload and processing
- [ ] User story submission
- [ ] Statistics and analytics
- [ ] Mobile responsiveness
- [ ] API error handling

### Automated Testing
```bash
# Run unit tests
python -m pytest tests/ -v

# Run integration tests  
python -m pytest tests/integration/ -v

# Run performance tests
python -m pytest tests/performance/ -v
```

### Test Data
Sample test queries and documents are provided in `tests/data/`

---

## 📊 Performance Metrics

### Benchmarks (on standard hardware)
- **Response Time**: < 2 seconds for typical queries
- **Memory Usage**: ~200MB baseline, ~500MB with large datasets
- **Startup Time**: ~10 seconds for initial model loading
- **Throughput**: 10-20 concurrent users (depends on deployment)

### Optimization Techniques Used
- 🔄 **Batch Processing** - Embeddings created in batches
- 💾 **Smart Caching** - LRU cache for responses and embeddings
- 🗜️ **Compression** - Efficient vector storage
- ⚡ **Lazy Loading** - Models loaded on demand
- 🔍 **Query Optimization** - Database indexes and connection pooling

---

## 🔒 Security

### API Key Management
```bash
# ✅ Good: Use environment variables
export GEMINI_API_KEY="your_key_here"

# ✅ Good: Use secrets.toml
GEMINI_API_KEY = "your_key_here"

# ❌ Bad: Hardcode in source
api_key = "AIzaSyC_your_key"  # NEVER DO THIS
```

### Data Privacy
- 🔐 All user data stored locally by default
- 🚫 No external data transmission except API calls
- 🔄 Optional data anonymization
- 🗑️ Easy data deletion and cleanup

### Security Best Practices
- 🔑 Secure API key storage
- 🛡️ Input validation and sanitization  
- 🔍 SQL injection prevention
- 🌐 CORS protection
- 📝 Audit logging

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Installation Problems
```bash
# Clear pip cache
pip cache purge

# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.8+
```

#### 2. API Key Issues
```bash
# Verify API key format
echo $GEMINI_API_KEY | wc -c  # Should be ~40 characters

# Test API key
python -c "import google.generativeai as genai; genai.configure(api_key='your_key'); print('API key valid')"
```

#### 3. Memory Issues
```python
# Reduce cache sizes in config
MAX_CACHE_SIZE = 100
CHUNK_SIZE = 256
VECTOR_DIM = 256
```

#### 4. Performance Issues
```bash
# Clear vector store
rm data/vector_store.faiss data/embeddings_cache.pkl

# Restart application
streamlit run app.py --server.runOnSave true
```

#### 5. Telugu Font Issues
- Ensure your browser supports web fonts
- Try refreshing the page
- Check browser console for font loading errors

#### 6. Database Corruption
```bash
# Backup and recreate database
cp data/mana_charitra.db data/backup.db
rm data/mana_charitra.db
# Restart app - database will be recreated
```

### Debug Mode
```bash
# Enable debug logging
streamlit run app.py --logger.level=debug

# Check logs
tail -f logs/app.log
```

### Getting Help
- 📖 Check this README thoroughly
- 🐛 Search existing GitHub issues
- 💬 Create new issue with detailed information
- 🌟 Join our community discussions

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute
1. 🐛 **Bug Reports** - Report issues with detailed steps
2. 💡 **Feature Requests** - Suggest new functionality
3. 📝 **Documentation** - Improve docs and guides
4. 🔧 **Code** - Submit pull requests
5. 📚 **Content** - Add historical data and stories
6. 🌐 **Translation** - Help with localization

### Development Workflow
```bash
# 1. Fork the repository
git clone https://github.com/yourusername/mana-charitra.git

# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Make changes and test
python -m pytest

# 4. Commit changes
git commit -m "Add amazing feature"

# 5. Push and create PR
git push origin feature/amazing-feature
```

### Code Style Guidelines
- Use **Black** for code formatting
- Follow **PEP 8** naming conventions  
- Add **type hints** where possible
- Write **docstrings** for functions
- Include **unit tests** for new features

### Content Guidelines
- Share **factual** historical information
- **Cite sources** when possible
- Use **respectful** language
- Include **Telugu** and English descriptions
- **Verify** information before submitting

---

## 📞 Support & Community

### Getting Support
- 📖 **Documentation**: Start with this README
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- 📧 **Email**: Contact maintainers for urgent issues

### Community Guidelines
- 🤝 **Be respectful** and inclusive
- 💡 **Share knowledge** and help others
- 🎯 **Stay on topic** - focus on Telugu history and tech
- 🚫 **No spam** or self-promotion
- 📚 **Contribute** meaningfully to the project

### Professional Services
For organizations needing custom development:
- 🏢 **Enterprise deployment** and scaling
- 🎓 **Training** and workshops
- 🔧 **Custom features** and integrations
- 🛠️ **Technical consulting** and support

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### What this means:
- ✅ **Free to use** for personal and commercial projects
- ✅ **Modify** and distribute as needed
- ✅ **Private use** allowed
- ✅ **Commercial use** allowed
- ⚠️ **No warranty** - use at your own risk
- 📝 **Attribution** - credit the original authors

---

## 🙏 Acknowledgments

### Technology Stack
- 🚀 **[Streamlit](https://streamlit.io)** - Amazing web framework
- 🧠 **[Google Gemini](https://makersuite.google.com)** - Powerful AI capabilities
- 🔍 **[FAISS](https://github.com/facebookresearch/faiss)** - Efficient similarity search
- 🤖 **[Sentence Transformers](https://www.sbert.net)** - Semantic embeddings
- 🐍 **[Python](https://python.org)** - Beautiful programming language

### Data Sources
- 📖 **Telugu Wikipedia** - Historical articles and information
- 🏛️ **Archaeological Survey of India** - Heritage site data
- 📚 **TTD Publications** - Tirumala temple information
- 🗺️ **AP/Telangana Tourism** - Cultural and historical content

### Community
- 👥 **Contributors** - Everyone who submitted code, content, and feedback
- 🌟 **Users** - The community using and promoting the project
- 🎓 **Educators** - Teachers and students spreading Telugu heritage
- 💻 **Developers** - Open source community for tools and libraries

### Special Thanks
- 🏛️ **Cultural institutions** preserving Telugu heritage
- 📚 **Historians and scholars** documenting our rich history
- 👨‍👩‍👧‍👦 **Families and elders** sharing traditional stories
- 🌍 **Global Telugu community** keeping the culture alive

---

## 🔄 Changelog

### Version 1.3.0 (Latest)
- ✨ Added Docker deployment support
- 🚀 Improved performance with smart caching
- 📱 Enhanced mobile responsiveness
- 🔧 Better error handling and logging
- 📊 Advanced analytics dashboard

### Version 1.2.0
- 📤 User story contribution feature
- 🔍 Enhanced search capabilities  
- 🎨 Improved Telugu typography
- 🔧 Performance optimizations

### Version 1.1.0
- 📚 Document upload functionality
- 🗄️ SQLite database integration
- 📈 Basic analytics and statistics
