import pytest
import numpy as np
import faiss
from unittest.mock import MagicMock, patch
from tools.faiss_engine_tool000 import FaissSearchEngine

@pytest.fixture
def mock_engine():
    """Fixture to create a mocked instance of FaissSearchEngine."""
    engine = FaissSearchEngine()
    engine.db_manager = MagicMock()  # Mock the database manager
    engine.index = None  # No FAISS index yet
    engine.chunks_list = []  # Empty chunk list
    return engine

# ---- Test: Initialization ----
def test_initialization(mock_engine):
    """Test if the FaissSearchEngine initializes correctly."""
    assert mock_engine.model is not None
    assert mock_engine.tokenizer is not None
    assert mock_engine.db_manager is not None
    assert mock_engine.index is None
    assert mock_engine.chunks_list == []

# ---- Test: _load_index_from_db ----
def test_load_index_from_db(mock_engine):
    """Test if index loads correctly from database."""
    mock_engine.db_manager.load_index.return_value = (None, [])
    mock_engine._load_index_from_db()
    assert mock_engine.index is None
    assert mock_engine.chunks_list == []

# ---- Test: _save_index_to_db ----
def test_save_index_to_db(mock_engine):
    """Test saving the FAISS index to the database."""
    mock_engine.index = faiss.IndexFlatIP(128)
    mock_engine.chunks_list = ["Sample text"]
    mock_engine._save_index_to_db()
    mock_engine.db_manager.save_index.assert_called_once()

# ---- Test: load_data ----
def test_load_data(mock_engine):
    """Test loading data from PDFs without updating the FAISS index."""
    mock_engine.db_manager.get_tracked_pdfs.return_value = {}
    mock_engine.db_manager.save_pdf_content.return_value = None

    pdf_texts = {"test.pdf": "This is a sample text"}
    
    with patch("os.path.exists", return_value=True), \
         patch("src.pdf_reader.PDFReader.load_from_folder", return_value=pdf_texts), \
         patch("os.path.getmtime", return_value=1710000000):  # Mock file modification time
        updated_files = mock_engine.load_data("dummy_path")
        assert "test.pdf" in updated_files

# ---- Test: update_index ----
def test_update_index(mock_engine):
    """Test updating the FAISS index with new data."""
    mock_engine.index = faiss.IndexFlatIP(128)
    pdf_texts = {"test.pdf": "This is a sample text"}
    before_update = len(mock_engine.chunks_list)

    with patch("src.faiss_engine.FaissSearchEngine._process_texts", return_value=("chunk", np.random.rand(1, 128))):
        mock_engine.update_index(pdf_texts)
        assert len(mock_engine.chunks_list) > before_update

# ---- Test: search ----
def test_search(mock_engine):
    """Test searching within the FAISS index."""
    mock_engine.index = faiss.IndexFlatIP(128)
    mock_engine.chunks_list = ["Sample chunk"]
    query_vector = np.random.rand(1, 128).astype("float32")
    
    mock_engine.model.encode = MagicMock(return_value=query_vector)
    with patch.object(mock_engine.index, "search", return_value=(np.array([[0.9]]), np.array([[0]]))):
        results = mock_engine.search("Sample query")
        assert len(results) == 1
        assert results[0][0] == "Sample chunk"

# ---- Test: _process_texts ----
def test_process_texts(mock_engine):
    """Test processing and encoding text chunks."""
    texts = ["This is a test sentence.", "Another example."]
    mock_engine.model.encode = MagicMock(return_value=np.random.rand(2, 128))
    chunks, vectors = mock_engine._process_texts(texts)
    assert len(chunks) == len(texts)
    assert vectors.shape == (2, 128)

# ---- Test: _sliding_window_tokenization ----
def test_sliding_window_tokenization(mock_engine):
    """Test the sliding window tokenization method."""
    text = "This is a longer test sentence that needs tokenization."
    chunks = mock_engine._sliding_window_tokenization(text, window_size=5, stride=3)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)

# ---- Test: _initialize_index ----
def test_initialize_index(mock_engine):
    """Test FAISS index initialization."""
    vectors = np.random.rand(5, 128).astype("float32")
    mock_engine._initialize_index(vectors)
    assert isinstance(mock_engine.index, faiss.IndexFlatIP)
    assert mock_engine.index.ntotal == 5

# ---- Test: search when index is None ----
def test_search_without_index(mock_engine):
    """Test search behavior when the FAISS index is not initialized."""
    mock_engine.index = None
    results = mock_engine.search("test query")
    assert results == []
