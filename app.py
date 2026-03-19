"""
Streamlit Frontend for Advanced RAG System
Provides web interface for document upload, query processing, and chat
"""

import streamlit as st
import os
import sys
import time
from typing import List, Dict, Any
import tempfile
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from config import config
from rag_pipeline import RAGPipeline
from utils import format_source_display, validate_file_type

# Configure Streamlit page
st.set_page_config(
    page_title=config.STREAMLIT_TITLE,
    page_icon="🤖",
    layout=config.STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'rag_pipeline' not in st.session_state:
        with st.spinner("Initializing RAG Pipeline..."):
            try:
                st.session_state.rag_pipeline = RAGPipeline()
                st.session_state.initialized = True
            except Exception as e:
                st.session_state.initialized = False
                st.session_state.error = str(e)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = {}

def display_header():
    """Display application header"""
    st.markdown('<h1 class="main-header">🤖 Advanced RAG System</h1>', unsafe_allow_html=True)
    st.markdown("---")

def display_sidebar():
    """Display sidebar with controls and information"""
    with st.sidebar:
        st.header("🔧 Controls")
        
        # System status
        if st.session_state.get('initialized', False):
            health = st.session_state.rag_pipeline.health_check()
            status_color = "🟢" if health["overall_status"] == "healthy" else "🟡"
            st.markdown(f"**System Status:** {status_color} {health['overall_status'].title()}")
        else:
            st.markdown("**System Status:** 🔴 Not Initialized")
        
        st.markdown("---")
        
        # Document upload section
        st.subheader("📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, MD)",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True,
            help="Supported formats: PDF, Text, Markdown"
        )
        
        if uploaded_files:
            if st.button("📥 Process Documents", type="primary"):
                process_uploaded_documents(uploaded_files)
        
        st.markdown("---")
        
        # System management
        st.subheader("⚙️ System Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Stats"):
                refresh_system_stats()
        
        with col2:
            if st.button("🗑️ Clear All Data"):
                if st.session_state.get('initialized', False):
                    clear_all_data()
        
        st.markdown("---")
        
        # Configuration info
        st.subheader("ℹ️ Configuration")
        config_info = {
            "Chunk Size": config.CHUNK_SIZE,
            "Chunk Overlap": config.CHUNK_OVERLAP,
            "Top K Retrieval": config.TOP_K_RETRIEVAL,
            "Top K Rerank": config.TOP_K_RERANK,
            "Confidence Threshold": config.CONFIDENCE_THRESHOLD,
            "Groq Model": config.GROQ_MODEL
        }
        
        for key, value in config_info.items():
            st.markdown(f"**{key}:** {value}")

def process_uploaded_documents(uploaded_files):
    """Process uploaded documents"""
    if not st.session_state.get('initialized', False):
        st.error("RAG Pipeline not initialized")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_files = []
    failed_files = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Ingest document
            result = st.session_state.rag_pipeline.ingest_document(tmp_file_path)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            if result["success"]:
                processed_files.append({
                    "name": uploaded_file.name,
                    "chunks": result.get("chunks_created", 0)
                })
            else:
                failed_files.append({
                    "name": uploaded_file.name,
                    "error": result.get("error", "Unknown error")
                })
        
        except Exception as e:
            failed_files.append({
                "name": uploaded_file.name,
                "error": str(e)
            })
    
    # Display results
    if processed_files:
        st.success(f"Successfully processed {len(processed_files)} files")
        for file_info in processed_files:
            st.markdown(f"✅ **{file_info['name']}** - {file_info['chunks']} chunks")
    
    if failed_files:
        st.error(f"Failed to process {len(failed_files)} files")
        for file_info in failed_files:
            st.markdown(f"❌ **{file_info['name']}** - {file_info['error']}")
    
    # Update session state
    st.session_state.uploaded_files.extend([f.name for f in uploaded_files])
    refresh_system_stats()
    
    # Clear the file uploader
    st.rerun()

def clear_all_data():
    """Clear all system data"""
    if st.session_state.get('initialized', False):
        result = st.session_state.rag_pipeline.clear_all_data()
        if result["success"]:
            st.success("All data cleared successfully")
            st.session_state.chat_history = []
            st.session_state.uploaded_files = []
            refresh_system_stats()
        else:
            st.error(f"Error clearing data: {result['message']}")

def refresh_system_stats():
    """Refresh system statistics"""
    if st.session_state.get('initialized', False):
        st.session_state.system_stats = st.session_state.rag_pipeline.get_system_stats()

def display_system_stats():
    """Display system statistics"""
    if not st.session_state.get('system_stats'):
        return
    
    stats = st.session_state.system_stats
    
    st.subheader("📊 System Statistics")
    
    # Pipeline stats
    pipeline_stats = stats.get("pipeline_stats", {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", pipeline_stats.get("total_queries", 0))
    
    with col2:
        success_rate = 0
        if pipeline_stats.get("total_queries", 0) > 0:
            success_rate = (pipeline_stats.get("successful_queries", 0) / pipeline_stats.get("total_queries", 1)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        st.metric("Blocked Queries", pipeline_stats.get("blocked_queries", 0))
    
    with col4:
        avg_time = pipeline_stats.get("avg_response_time", 0)
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    # Document stats
    ingestion_stats = stats.get("ingestion_stats", {})
    vector_store_stats = ingestion_stats.get("vector_store_stats", {})
    
    if isinstance(vector_store_stats, dict) and "document_count" in vector_store_stats:
        st.markdown(f"**Documents in Store:** {vector_store_stats['document_count']}")
    
    # Uploaded files
    if st.session_state.uploaded_files:
        st.markdown("**Recently Uploaded Files:**")
        for filename in st.session_state.uploaded_files[-5:]:  # Show last 5
            st.markdown(f"• {filename}")

def display_chat_interface():
    """Display main chat interface"""
    st.subheader("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', 
                        unsafe_allow_html=True)
            
            # Display sources if available
            if message.get("sources"):
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f'''
                        <div class="source-card">
                            <strong>Source {source.get("doc_id", i+1)}:</strong><br>
                            <em>{source.get("source", "Unknown")}</em><br>
                            <small>Score: {source.get("score", 0):.3f}</small><br>
                            <strong>Content Preview:</strong><br>
                            {source.get("content", "")}
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Debug: Show retrieval stats
                    if message.get("retrieval_stats"):
                        st.markdown("### 🔍 Retrieval Debug Info")
                        stats = message["retrieval_stats"]
                        st.json(stats)
                        
                        # Show top retrieved chunks for debugging
                        if message.get("debug_chunks"):
                            st.markdown("### 📄 Retrieved Chunks (Debug)")
                            for i, chunk in enumerate(message["debug_chunks"][:3]):  # Show top 3
                                st.markdown(f"**Chunk {i+1}** (Score: {chunk.get('score', 0):.3f})")
                                st.markdown(f"```\n{chunk.get('content', '')[:500]}...\n```")
                                st.markdown("---")
            
            # Display confidence if available
            if message.get("confidence"):
                confidence_data = message["confidence"]
                confidence_level = confidence_data.get("confidence", "unknown")
                confidence_class = f"confidence-{confidence_level}"
                avg_score = confidence_data.get("avg_score", 0)
                
                st.markdown(f'''
                <div class="metric-card">
                    <strong>Confidence:</strong> 
                    <span class="{confidence_class}">{confidence_level.title()}</span>
                    (Avg Score: {avg_score:.3f})
                </div>
                ''', unsafe_allow_html=True)
    
    # Query input
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your question:",
            placeholder="Ask about your uploaded documents...",
            key="query_input"
        )
    
    with col2:
        use_adaptive = st.checkbox("Adaptive", value=True, help="Use adaptive retrieval")
    
    if st.button("🔍 Send Query", type="primary") and query:
        process_query(query, use_adaptive)

def process_query(query: str, use_adaptive: bool):
    """Process user query"""
    if not st.session_state.get('initialized', False):
        st.error("RAG Pipeline not initialized")
        return
    
    # Add user message to chat
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now().isoformat()
    })
    
    # Process query with loading indicator
    with st.spinner("Processing your query..."):
        try:
            result = st.session_state.rag_pipeline.process_query(query, use_adaptive)
            
            # Add assistant response to chat
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result.get("answer", "No response generated"),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence_data"),
                "processing_time": result.get("processing_time", 0),
                "query_rewritten": result.get("query_rewritten", False),
                "retrieval_stats": result.get("retrieval_stats", {}),
                "debug_chunks": result.get("debug_chunks", []),
                "timestamp": datetime.now().isoformat()
            })
            
            # Show processing info in sidebar
            with st.sidebar:
                st.markdown("**Last Query Info:**")
                st.markdown(f"• Processing Time: {result.get('processing_time', 0):.2f}s")
                st.markdown(f"• Sources Found: {len(result.get('sources', []))}")
                if result.get("query_rewritten"):
                    st.markdown("• Query was rewritten for better results")
        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"I apologize, but I encountered an error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    # Rerun to update chat display
    st.rerun()

def display_health_check():
    """Display detailed health check"""
    if not st.session_state.get('initialized', False):
        st.error("RAG Pipeline not initialized")
        return
    
    health = st.session_state.rag_pipeline.health_check()
    
    st.subheader("🏥 System Health Check")
    
    # Overall status
    status_color = "🟢" if health["overall_status"] == "healthy" else "🟡"
    st.markdown(f"**Overall Status:** {status_color} {health['overall_status'].title()}")
    
    # Component status
    st.markdown("### Component Status")
    for component, status in health["components"].items():
        color = "🟢" if status == "healthy" else "🟡" if status == "limited" else "🔴"
        st.markdown(f"**{component.replace('_', ' ').title()}:** {color} {status}")
    
    # Issues
    if health["issues"]:
        st.markdown("### Issues Found")
        for issue in health["issues"]:
            st.markdown(f"⚠️ {issue}")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Check initialization
    if not st.session_state.get('initialized', False):
        st.error("❌ Failed to initialize RAG Pipeline")
        if 'error' in st.session_state:
            st.error(f"Error: {st.session_state['error']}")
        
        st.markdown("### Troubleshooting:")
        st.markdown("1. Ensure all required packages are installed")
        st.markdown("2. Check your API keys in environment variables")
        st.markdown("3. Verify internet connection for model downloads")
        return
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Statistics", "🏥 Health"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_system_stats()
    
    with tab3:
        display_health_check()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Advanced RAG System - Powered by Groq & LangChain"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
