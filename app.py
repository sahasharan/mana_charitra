"""
Mana Charitra - Telugu History Chatbot
Advanced implementation with optimal performance and free deployment
"""

import streamlit as st
import google.generativeai as genai
import os
import json
import sqlite3
import hashlib
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import io
import re
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Advanced configuration class for optimal performance
class OptimizedConfig:
    """Production-ready configuration with memory and performance optimizations"""
    
    def __init__(self):
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast model
        self.VECTOR_DIM = 384
        self.MAX_TOKENS = 8192
        self.CHUNK_SIZE = 512
        self.CHUNK_OVERLAP = 64
        self.DB_PATH = "data/mana_charitra.db"
        self.VECTOR_STORE_PATH = "data/vector_store.faiss"
        self.EMBEDDINGS_CACHE_PATH = "data/embeddings_cache.pkl"
        self.MAX_CACHE_SIZE = 1000
        
        # Telugu language specific patterns
        self.TELUGU_PATTERNS = {
            'places': [
                r'[\u0C00-\u0C7F]+\s*(కోట|దేవాలయం|గుడి|మందిరం|పట్టణం|గ్రామం)',
                r'(వరంగల్|తిరుమల|చిత్తూర్|విజయవాడ|హైదరాబాద్|కర్నూల్)'
            ],
            'history_keywords': [
                'చరిత్ర', 'కథ', 'పురాణం', 'వంశావళి', 'కాలం', 'రాజవంశం'
            ]
        }

config = OptimizedConfig()

# Advanced caching system for optimal performance
class SmartCache:
    """Memory-efficient caching with LRU eviction"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[any]:
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: any) -> None:
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Remove least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), 
                     key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]

# High-performance vector store manager
class OptimizedVectorStore:
    """FAISS-based vector store with advanced optimizations"""
    
    def __init__(self):
        self.model = None
        self.index = None
        self.metadata = []
        self.cache = SmartCache()
        self._load_model()
    
    def _load_model(self) -> None:
        """Load embedding model with caching"""
        try:
            self.model = SentenceTransformer(config.EMBEDDING_MODEL)
            # Optimize model for inference
            self.model.eval()
            logging.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logging.error(f"❌ Failed to load embedding model: {e}")
            st.error("Failed to load embedding model")
    
    def _create_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Create embeddings in optimized batches"""
        cache_key = hashlib.md5('|'.join(texts).encode()).hexdigest()
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            return cached
        
        # Batch processing for efficiency
        embeddings = self.model.encode(texts, 
                                     batch_size=32,
                                     show_progress_bar=False,
                                     convert_to_numpy=True)
        
        self.cache.set(cache_key, embeddings)
        return embeddings
    
    def add_documents(self, texts: List[str], metadata: List[Dict]) -> None:
        """Add documents to vector store with optimizations"""
        if not texts:
            return
        
        try:
            embeddings = self._create_embeddings_batch(texts)
            
            if self.index is None:
                # Initialize FAISS index with optimal settings
                self.index = faiss.IndexFlatIP(config.VECTOR_DIM)
                # Add index optimization for larger datasets
                if len(texts) > 1000:
                    self.index = faiss.IndexIVFFlat(self.index, config.VECTOR_DIM, 
                                                  min(100, len(texts) // 10))
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            self.metadata.extend(metadata)
            
            logging.info(f"✅ Added {len(texts)} documents to vector store")
            
        except Exception as e:
            logging.error(f"❌ Failed to add documents: {e}")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Optimized similarity search"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            query_embedding = self._create_embeddings_batch([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):
                    results.append((
                        self.metadata[idx]['text'],
                        float(score),
                        self.metadata[idx]
                    ))
            
            return results
            
        except Exception as e:
            logging.error(f"❌ Search failed: {e}")
            return []
    
    def save(self) -> None:
        """Save vector store to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, config.VECTOR_STORE_PATH)
                
            with open(config.EMBEDDINGS_CACHE_PATH, 'wb') as f:
                pickle.dump(self.metadata, f)
                
        except Exception as e:
            logging.error(f"❌ Failed to save vector store: {e}")
    
    def load(self) -> None:
        """Load vector store from disk"""
        try:
            if os.path.exists(config.VECTOR_STORE_PATH):
                self.index = faiss.read_index(config.VECTOR_STORE_PATH)
                
            if os.path.exists(config.EMBEDDINGS_CACHE_PATH):
                with open(config.EMBEDDINGS_CACHE_PATH, 'rb') as f:
                    self.metadata = pickle.load(f)
                    
        except Exception as e:
            logging.error(f"❌ Failed to load vector store: {e}")

# Advanced database manager with optimizations
class OptimizedDatabase:
    """SQLite database with connection pooling and optimizations"""
    
    def __init__(self):
        self.db_path = config.DB_PATH
        self._ensure_directory()
        self._init_database()
    
    def _ensure_directory(self) -> None:
        """Ensure data directory exists"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self) -> None:
        """Initialize database with optimized schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable performance optimizations
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=memory")
                
                # Create optimized tables
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source TEXT,
                        place_name TEXT,
                        category TEXT DEFAULT 'general',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        content_hash TEXT UNIQUE
                    );
                    
                    CREATE TABLE IF NOT EXISTS user_contributions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        place_name TEXT NOT NULL,
                        story TEXT NOT NULL,
                        contributor_name TEXT,
                        status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        story_hash TEXT UNIQUE
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_documents_place ON documents(place_name);
                    CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
                    CREATE INDEX IF NOT EXISTS idx_contributions_place ON user_contributions(place_name);
                    CREATE INDEX IF NOT EXISTS idx_contributions_status ON user_contributions(status);
                """)
                
                conn.commit()
                logging.info("✅ Database initialized successfully")
                
        except Exception as e:
            logging.error(f"❌ Database initialization failed: {e}")
    
    def add_document(self, title: str, content: str, source: str = None, 
                    place_name: str = None, category: str = 'general') -> bool:
        """Add document with duplicate prevention"""
        try:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO documents 
                    (title, content, source, place_name, category, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (title, content, source, place_name, category, content_hash))
                
                return conn.total_changes > 0
                
        except Exception as e:
            logging.error(f"❌ Failed to add document: {e}")
            return False
    
    def add_user_story(self, place_name: str, story: str, 
                      contributor_name: str = None) -> bool:
        """Add user contribution with validation"""
        try:
            story_hash = hashlib.md5(story.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO user_contributions 
                    (place_name, story, contributor_name, story_hash)
                    VALUES (?, ?, ?, ?)
                """, (place_name, story, contributor_name, story_hash))
                
                return conn.total_changes > 0
                
        except Exception as e:
            logging.error(f"❌ Failed to add user story: {e}")
            return False
    
    def get_documents_by_place(self, place_name: str) -> List[Dict]:
        """Get documents filtered by place with caching"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM documents 
                    WHERE place_name LIKE ? 
                    ORDER BY created_at DESC
                """, (f"%{place_name}%",))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logging.error(f"❌ Failed to get documents: {e}")
            return []

# Advanced document processor with parallel processing
class OptimizedDocumentProcessor:
    """High-performance document processing with async operations"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def process_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF with error handling"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text.strip())
            
            return '\n\n'.join(text_parts)
            
        except Exception as e:
            logging.error(f"❌ PDF processing failed: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Intelligent text chunking with overlap"""
        if len(text) <= config.CHUNK_SIZE:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + config.CHUNK_SIZE
            
            # Find sentence boundary for better chunks
            if end < len(text):
                boundary = text.rfind('.', start, end)
                if boundary > start + config.CHUNK_SIZE // 2:
                    end = boundary + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - config.CHUNK_OVERLAP
        
        return chunks
    
    def extract_place_names(self, text: str) -> List[str]:
        """Extract Telugu place names using regex patterns"""
        places = set()
        
        for pattern in config.TELUGU_PATTERNS['places']:
            matches = re.findall(pattern, text)
            places.update(matches)
        
        # Clean and filter results
        return list({place.strip() for place in places if len(place.strip()) > 2})

# Advanced Telugu History Bot with optimizations
class ManaCharitra:
    """Main chatbot class with advanced features"""
    
    def __init__(self):
        self.db = OptimizedDatabase()
        self.vector_store = OptimizedVectorStore()
        self.doc_processor = OptimizedDocumentProcessor()
        self.response_cache = SmartCache()
        self._setup_gemini()
        self._load_initial_data()
    
    def _setup_gemini(self) -> None:
        """Configure Gemini API with optimal settings"""
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("⚠️ Gemini API key not found. Please set GEMINI_API_KEY in secrets.")
            return
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        logging.info("✅ Gemini API configured successfully")
    
    def _load_initial_data(self) -> None:
        """Load initial Telugu historical data"""
        try:
            self.vector_store.load()
            
            # Add default Telugu historical content if vector store is empty
            if self.vector_store.index is None or self.vector_store.index.ntotal == 0:
                default_data = self._get_default_telugu_content()
                self._add_bulk_content(default_data)
            
            logging.info("✅ Initial data loaded successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to load initial data: {e}")
    
    def _get_default_telugu_content(self) -> List[Dict]:
        """Default Telugu historical content for bootstrapping"""
        return [
            {
                "title": "వరంగల్ కోట చరిత్ర",
                "content": """వరంగల్ కోట తెలంగాణ రాష్ట్రంలోని వరంగల్ జిల్లాలో ఉంది. ఇది కాకతీయ రాజవంశం కాలంలో నిర్మించబడింది. 
                రాణి రుద్రమాదేవి మరియు ప్రతాపరుద్రుడు కాలంలో ఈ కోట ప్రసిద్ధిచెందింది. కోట నిర్మాణంలో గ్రానైట్ రాళ్లను ఉపయోగించారు. 
                ఇది దక్షిణ భారతదేశంలోని ముఖ్యమైన కోటలలో ఒకటిగా పరిగణించబడుతుంది.""",
                "place_name": "వరంగల్",
                "category": "కోట"
            },
            {
                "title": "తిరుమల దేవాలయ చరిత్ర",
                "content": """తిరుమల వేంకటేశ్వర స్వామి దేవాలయం ఆంధ్రప్రదేశ్ రాష్ట్రంలోని చిత్తూర్ జిల్లాలో ఉంది. 
                ఇది వేంకటాద్రి పర్వతంపై స్థితిగా ఉంది. ఈ దేవాలయం వేలాది సంవత్సరాల చరిత్రను కలిగి ఉంది. 
                పల్లవ, చోళ, విజయనగర రాజవంశాల కాలంలో ఈ దేవాలయం అభివృద్ధి చెందింది. ప్రతిరోజూ లక్షలాది మంది భక్తులు దర్శనానికి వస్తారు.""",
                "place_name": "తిరుమల",
                "category": "దేవాలయం"
            }
        ]
    
    def _add_bulk_content(self, content_list: List[Dict]) -> None:
        """Add multiple content items efficiently"""
        texts = []
        metadata = []
        
        for item in content_list:
            # Add to database
            self.db.add_document(
                title=item["title"],
                content=item["content"],
                place_name=item["place_name"],
                category=item.get("category", "general")
            )
            
            # Prepare for vector store
            texts.append(item["content"])
            metadata.append({
                "text": item["content"],
                "title": item["title"],
                "place_name": item["place_name"],
                "category": item.get("category", "general")
            })
        
        # Add to vector store in batch
        self.vector_store.add_documents(texts, metadata)
        self.vector_store.save()
    
    def chat(self, user_query: str) -> str:
        """Main chat function with caching and optimization"""
        # Check cache first
        cache_key = hashlib.md5(user_query.encode()).hexdigest()
        cached_response = self.response_cache.get(cache_key)
        if cached_response:
            return cached_response
        
        try:
            # Search for relevant content
            search_results = self.vector_store.search(user_query, k=3)
            
            # Build context from search results
            context = self._build_context(search_results)
            
            # Generate response using Gemini
            response = self._generate_response(user_query, context)
            
            # Cache the response
            self.response_cache.set(cache_key, response)
            
            return response
            
        except Exception as e:
            logging.error(f"❌ Chat failed: {e}")
            return "క్షమించండి, ప్రస్తుతం సమస్య ఎదుర్కొంటున్నాను. దయచేసి తర్వాత ప్రయత్నించండి."
    
    def _build_context(self, search_results: List[Tuple]) -> str:
        """Build context from search results"""
        if not search_results:
            return "సంబంధిత చరిత్ర సమాచారం దొరకలేదు."
        
        context_parts = []
        for text, score, metadata in search_results:
            if score > 0.5:  # Relevance threshold
                context_parts.append(f"**{metadata['title']}**\n{text}")
        
        return "\n\n".join(context_parts[:3])  # Top 3 results
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using Gemini with Telugu optimization"""
        prompt = f"""
        మీరు "మన చరిత్ర" అనే తెలుగు చరిత్ర చాట్‌బాట్. మీ పని తెలుగు ప్రాంతాల చరిత్ర మరియు సాంస్కృతిక సమాచారాన్ని అందించడం.

        నియమాలు:
        1. ఎల్లప్పుడూ తెలుగులో జవాబు ఇవ్వండి
        2. చరిత్ర, సంస్కృతి గురించి ఆసక్తికరంగా చెప్పండి
        3. వాస్తవిక సమాచారాన్ని మాత్రమే అందించండి
        4. తెలియని విషయాల గురించి ఊహలు చేయవద్దు

        సందర్భ సమాచారం:
        {context}

        వినియోగదారు ప్రశ్న: {query}

        దయచేసి తెలుగులో వివరణాత్మక మరియు ఆసక్తికరమైన జవాబు ఇవ్వండి:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logging.error(f"❌ Response generation failed: {e}")
            return "క్షమించండి, ప్రస్తుతం జవాబు రూపొందించడంలో సమస్య ఎదుర్కొంటున్నాను."
    
    def add_document_from_file(self, uploaded_file) -> bool:
        """Process uploaded file and add to knowledge base"""
        try:
            # Extract text from file
            if uploaded_file.type == "application/pdf":
                text = self.doc_processor.process_pdf(uploaded_file.getvalue())
            else:
                text = uploaded_file.getvalue().decode('utf-8')
            
            if not text.strip():
                return False
            
            # Extract place names
            places = self.doc_processor.extract_place_names(text)
            place_name = places[0] if places else "సాధారణ"
            
            # Chunk text for better processing
            chunks = self.doc_processor.chunk_text(text)
            
            # Add to database and vector store
            success_count = 0
            for i, chunk in enumerate(chunks):
                title = f"{uploaded_file.name} - భాగం {i+1}"
                
                if self.db.add_document(title, chunk, uploaded_file.name, place_name):
                    success_count += 1
            
            # Add to vector store
            if chunks:
                metadata = [{
                    "text": chunk,
                    "title": f"{uploaded_file.name} - భాగం {i+1}",
                    "place_name": place_name,
                    "category": "user_upload"
                } for i, chunk in enumerate(chunks)]
                
                self.vector_store.add_documents(chunks, metadata)
                self.vector_store.save()
            
            return success_count > 0
            
        except Exception as e:
            logging.error(f"❌ File processing failed: {e}")
            return False
    
    def add_user_story(self, place_name: str, story: str, 
                      contributor_name: str = None) -> bool:
        """Add user story to knowledge base"""
        try:
            # Add to database
            db_success = self.db.add_user_story(place_name, story, contributor_name)
            
            # Add to vector store
            metadata = [{
                "text": story,
                "title": f"{place_name} - వినియోగదారు కథ",
                "place_name": place_name,
                "category": "user_story"
            }]
            
            self.vector_store.add_documents([story], metadata)
            self.vector_store.save()
            
            return db_success
            
        except Exception as e:
            logging.error(f"❌ Failed to add user story: {e}")
            return False

# Initialize logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit UI with advanced features
def main():
    """Main Streamlit application with optimized UI"""
    
    st.set_page_config(
        page_title="మన చరిత్ర - Mana Charitra",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for Telugu fonts and styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@300;400;500;700&display=swap');
    
    .main-title {
        font-family: 'Noto Sans Telugu', sans-serif;
        text-align: center;
        color: #2E8B57;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .telugu-text {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .bot-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .stats-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        with st.spinner('🔄 మన చరిత్ర లోడ్ అవుతోంది...'):
            st.session_state.chatbot = ManaCharitra()
            st.success('✅ మన చరిత్ర సిద్ధంగా ఉంది!')
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Main title
    st.markdown('<h1 class="main-title">📚 మన చరిత్ర - Mana Charitra 📚</h1>', 
                unsafe_allow_html=True)
    
    st.markdown('<p class="telugu-text" style="text-align: center; font-size: 1.2rem; color: #666;">తెలుగు ప్రాంతాల చరిత్ర మరియు సాంస్కృతిక వైభవాన్ని తెలుసుకోండి</p>', 
                unsafe_allow_html=True)
    
    # Sidebar with features
    with st.sidebar:
        st.markdown("### 🎯 ఫీచర్లు")
        
        selected_tab = st.radio(
            "ఎంచుకోండి:",
            ["💬 చాట్", "📤 దస్త్రం అప్‌లోడ్", "✍️ మీ కథ", "📊 గణాంకాలు"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 💡 ఉదాహరణ ప్రశ్నలు")
        example_questions = [
            "వరంగల్ కోట చరిత్ర చెప్పండి",
            "తిరుమల దేవాలయ ప్రాముఖ్యత ఏమిటి?",
            "కాకతీయ రాజవంశం గురించి తెలియజేయండి",
            "హైదరాబాద్ చరిత్ర చెప్పండి"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"ex_{hash(question)}"):
                st.session_state.user_input = question
    
    # Main content area
    if selected_tab == "💬 చాట్":
        render_chat_interface()
    elif selected_tab == "📤 దస్త్రం అప్‌లోడ్":
        render_upload_interface()
    elif selected_tab == "✍️ మీ కథ":
        render_story_interface()
    elif selected_tab == "📊 గణాంకాలు":
        render_stats_interface()

def render_chat_interface():
    """Render the main chat interface"""
    st.markdown("### 💬 మన చరిత్రతో మాట్లాడండి")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                st.markdown(f'<div class="user-message telugu-text"><strong>మీరు:</strong> {user_msg}</div>', 
                           unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message telugu-text"><strong>మన చరిత్ర:</strong> {bot_msg}</div>', 
                           unsafe_allow_html=True)
        else:
            st.markdown('<div class="telugu-text" style="text-align: center; color: #666; padding: 20px;">నమస్కారం! తెలుగు ప్రాంతాల చరిత్ర గురించి ఏదైనా అడగండి...</div>', 
                       unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "మీ ప్రశ్న ఇక్కడ టైప్ చేయండి:",
            value=st.session_state.get('user_input', ''),
            placeholder="ఉదా: వరంగల్ కోట చరిత్ర చెప్పండి",
            key="chat_input"
        )
    
    with col2:
        send_button = st.button("📤 పంపండి", type="primary")
    
    # Process user input
    if (send_button and user_input.strip()) or st.session_state.get('user_input'):
        if st.session_state.get('user_input'):
            user_input = st.session_state.user_input
            st.session_state.user_input = ""
        
        with st.spinner('🤔 ఆలోచిస్తున్నాను...'):
            response = st.session_state.chatbot.chat(user_input)
            st.session_state.chat_history.append((user_input, response))
            st.rerun()

def render_upload_interface():
    """Render file upload interface"""
    st.markdown("### 📤 దస్త్రం అప్‌లోడ్ చేయండి")
    
    st.markdown('<div class="telugu-text">మీ దగ్గర ఉన్న చరిత్ర దస్త్రాలను (PDF, టెక్స్ట్) అప్‌లోడ్ చేసి మన చరిత్ర డేటాబేస్‌ను సమృద్ధిగా చేయండి.</div>', 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "దస్త్రం ఎంచుకోండి:",
        type=['pdf', 'txt'],
        help="PDF లేదా టెక్స్ట్ ఫైల్స్ మాత్రమే"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ దస్త్రం ఎంచుకోబడింది: {uploaded_file.name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"📁 ఫైల్ రకం: {uploaded_file.type}")
        
        with col2:
            st.info(f"📏 ఫైల్ పరిమాణం: {uploaded_file.size / 1024:.1f} KB")
        
        if st.button("🔄 ప్రాసెస్ చేయండి", type="primary"):
            with st.spinner('📖 దస్త్రం చదువుతున్నాను...'):
                success = st.session_state.chatbot.add_document_from_file(uploaded_file)
                
                if success:
                    st.success("✅ దస్త్రం విజయవంతంగా జోడించబడింది!")
                    st.balloons()
                else:
                    st.error("❌ దస్త్రం ప్రాసెస్ చేయడంలో సమస్య ఎదురైంది")

def render_story_interface():
    """Render user story contribution interface"""
    st.markdown("### ✍️ మీ కథ పంచుకోండి")
    
    st.markdown('<div class="telugu-text">మీ గ్రామం లేదా ప్రాంతం గురించి తెలిసిన కథలు, చరిత్ర, లేదా పురాణాలను పంచుకోండి.</div>', 
                unsafe_allow_html=True)
    
    with st.form("story_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            place_name = st.text_input(
                "ప్రాంతం/గ్రామం పేరు:",
                placeholder="ఉదా: వరంగల్, తిరుమల"
            )
        
        with col2:
            contributor_name = st.text_input(
                "మీ పేరు (ఐచ్ఛికం):",
                placeholder="మీ పేరు"
            )
        
        story = st.text_area(
            "మీ కథ/చరిత్ర:",
            height=200,
            placeholder="ఇక్కడ మీ ప్రాంతం గురించిన చరిత్ర, కథలు, పురాణాలను వ్రాయండి...",
            help="దయచేసి వాస్తవిక సమాచారాన్ని మాత్రమే పంచుకోండి"
        )
        
        submitted = st.form_submit_button("📝 కథ జోడించండి", type="primary")
        
        if submitted:
            if place_name.strip() and story.strip():
                with st.spinner('💾 కథ సేవ్ చేస్తున్నాను...'):
                    success = st.session_state.chatbot.add_user_story(
                        place_name.strip(), 
                        story.strip(), 
                        contributor_name.strip() or None
                    )
                    
                    if success:
                        st.success("✅ మీ కథ విజయవంతంగా జోడించబడింది! ధన్యవాదాలు!")
                        st.balloons()
                    else:
                        st.warning("⚠️ ఈ కథ ఇప్పటికే ఉంది లేదా జోడించడంలో సమస్య ఎదురైంది")
            else:
                st.error("❌ దయచేసి ప్రాంతం పేరు మరియు కథ రెండూ వ్రాయండి")

def render_stats_interface():
    """Render statistics and analytics interface"""
    st.markdown("### 📊 గణాంకాలు మరియు విశ్లేషణ")
    
    # Get statistics from database
    db = st.session_state.chatbot.db
    
    try:
        with sqlite3.connect(db.db_path) as conn:
            # Total documents
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            # Total user contributions
            cursor = conn.execute("SELECT COUNT(*) FROM user_contributions")
            total_contributions = cursor.fetchone()[0]
            
            # Popular places
            cursor = conn.execute("""
                SELECT place_name, COUNT(*) as count 
                FROM documents 
                WHERE place_name IS NOT NULL 
                GROUP BY place_name 
                ORDER BY count DESC 
                LIMIT 10
            """)
            popular_places = cursor.fetchall()
            
            # Recent additions
            cursor = conn.execute("""
                SELECT title, place_name, created_at 
                FROM documents 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            recent_docs = cursor.fetchall()
    
    except Exception as e:
        st.error(f"గణాంకాలు లోడ్ చేయడంలో సమస్య: {e}")
        return
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <h3 style="color: #2E8B57;">📚</h3>
            <h2>{total_docs}</h2>
            <p>మొత్తం దస్త్రాలు</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <h3 style="color: #FF6B6B;">✍️</h3>
            <h2>{total_contributions}</h2>
            <p>వినియోగదారు కథలు</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        vector_count = st.session_state.chatbot.vector_store.index.ntotal if st.session_state.chatbot.vector_store.index else 0
        st.markdown(f"""
        <div class="stats-card">
            <h3 style="color: #4ECDC4;">🔍</h3>
            <h2>{vector_count}</h2>
            <p>వెక్టర్ ఎంట్రీలు</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-card">
            <h3 style="color: #45B7D1;">🗺️</h3>
            <h2>{len(popular_places)}</h2>
            <p>ప్రాంతాలు</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Popular places chart
    if popular_places:
        st.markdown("### 🏆 ప్రసిద్ధ ప్రాంతాలు")
        
        places_df = pd.DataFrame(popular_places, columns=['ప్రాంతం', 'దస్త్రాల సంఖ్య'])
        st.bar_chart(places_df.set_index('ప్రాంతం'))
    
    # Recent additions
    if recent_docs:
        st.markdown("### 🆕 ఇటీవలి జోడింపులు")
        
        for title, place, created_at in recent_docs:
            with st.expander(f"📖 {title}"):
                st.markdown(f"**ప్రాంతం:** {place or 'సాధారణ'}")
                st.markdown(f"**జోడించిన తేదీ:** {created_at}")

if __name__ == "__main__":
    main()