import os
import logging
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer
from ingestion.pdf_reader import PDFReader
from db.db_manager import DBManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class FaissSearchEngine:
    """
    FAISS-based search engine for storing and retrieving text using an SQLite database.
    """

    def __init__(self, model_name="multi-qa-MiniLM-L6-cos-v1", db_path="faiss_db.sqlite"):
        """
        Initializes the search engine with a SentenceTransformer model and SQLite database.
        
        Args:
            model_name (str): Name of the SentenceTransformer model.
            db_path (str): Path to the SQLite database file.
        """
        logging.info("Initializing FaissSearchEngine...")

        self.model = SentenceTransformer(model_name)
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")
        self.db_manager = DBManager(db_path)
        self.index = None
        self.chunks_list = []
        self._load_index_from_db()

        logging.info("FaissSearchEngine initialized successfully.")

    def _load_index_from_db(self):
        """Loads the FAISS index and text chunks from the database."""
        self.index, self.chunks_list = self.db_manager.load_index()
        if self.index:
            logging.info("FAISS index successfully loaded from database.")
        else:
            logging.warning("No FAISS index found in database. A new one will be created.")

    def _save_index_to_db(self):
        """Saves the FAISS index and text chunks to the database."""
        self.db_manager.save_index(self.index, self.chunks_list)

    def load_data(self, folder_path):
        """
        Loads new or updated PDF files but does not update the FAISS index yet.
        
        Args:
            folder_path (str): Path to the folder containing PDF files.
        
        Returns:
            dict: A dictionary of updated file names and their text contents.
        """
        if not os.path.exists(folder_path):
            logging.error(f"The path '{folder_path}' does not exist!")
            return {}

        tracked_pdfs = self.db_manager.get_tracked_pdfs()
        pdf_texts = PDFReader.load_from_folder(folder_path)

        if not pdf_texts:
            logging.warning("No valid PDF files found for processing!")
            return {}

        logging.info("Checking for new or updated files...")
        updated_files = {}

        for pdf_path, pdf_text in pdf_texts.items():
            filename = os.path.basename(pdf_path)
            last_modified = os.path.getmtime(pdf_path)

            if filename not in tracked_pdfs or tracked_pdfs[filename] < last_modified:
                updated_files[filename] = pdf_text
                self.db_manager.save_pdf_content(filename, pdf_text, last_modified)

        return updated_files

    def update_index(self, pdf_texts):
        """
        Updates the FAISS index with new PDF contents.
        
        Args:
            pdf_texts (dict): Dictionary containing PDF file names and their contents.
        """
        if not pdf_texts:
            logging.info("No updates found. Index remains unchanged.")
            return

        for filename, pdf_text in pdf_texts.items():
            texts = pdf_text.split("\n")
            new_chunks, new_text_vectors = self._process_texts(texts)

            if self.index is None:
                self._initialize_index(new_text_vectors)
                self.chunks_list = new_chunks
            else:
                unique_chunks = [chunk for chunk in new_chunks if chunk not in self.chunks_list]
                unique_vectors = [vec for chunk, vec in zip(new_chunks, new_text_vectors) if chunk not in self.chunks_list]

                self.chunks_list.extend(unique_chunks)
                if unique_vectors:
                    self.index.add(np.array(unique_vectors))

        self._save_index_to_db()
        logging.info("Index updated.")

    def _process_texts(self, texts, window_size=50, stride=40):
        """
        Segments texts and generates their embeddings.
        
        Args:
            texts (list): List of text strings.
            window_size (int): Token window size.
            stride (int): Stride size for overlapping windows.
        
        Returns:
            tuple: (List of text chunks, NumPy array of embeddings).
        """
        chunks_list = []
        all_texts = []

        for text in texts:
            if text.strip():
                chunks = self._sliding_window_tokenization(text, window_size, stride)
                chunks_list.extend(chunks)
                all_texts.extend(chunks)

        if not all_texts:
            logging.warning("No valid texts found for processing!")
            return [], np.array([])

        all_text_vectors = self.model.encode(all_texts, batch_size=16, convert_to_numpy=True)
        return chunks_list, all_text_vectors

    def _sliding_window_tokenization(self, text, window_size=50, stride=40):
        """
        Splits text using a sliding window approach.
        
        Args:
            text (str): Input text.
            window_size (int): Token window size.
            stride (int): Stride for sliding window.
        
        Returns:
            list: List of tokenized text chunks.
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) < window_size:
            return [self.tokenizer.convert_tokens_to_string(tokens)]

        return [
            self.tokenizer.convert_tokens_to_string(tokens[i : i + window_size])
            for i in range(0, len(tokens) - window_size + 1, stride)
        ]

    def _initialize_index(self, text_vectors):
        """
        Initializes a new FAISS index and stores it in the database.
        
        Args:
            text_vectors (np.ndarray): NumPy array of text embeddings.
        """
        text_vectors /= np.linalg.norm(text_vectors, axis=1, keepdims=True)
        dimension = text_vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(text_vectors)
        self._save_index_to_db()
        logging.info("FAISS index created and stored in the database.")

    def search(self, query, top_k=5, min_similarity=0.8):
        """
        Performs a search in the FAISS index and returns the most similar results.
    
        Args:
            query (str): Search query.
            top_k (int): Number of top matches to return.
            min_similarity (float): Minimum similarity score for results.
    
        Returns:
            list: List of matching text results.
        """
        if self.index is None or not self.chunks_list:
            logging.error("No index or textual data available. Please load the data first!")
            return []

        query_vector = np.array([self.model.encode(query)]).astype("float32")
        query_vector /= np.linalg.norm(query_vector)

        similarities, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks_list) and similarities[0][i] >= min_similarity:
                results.append(self.chunks_list[idx])

        return results if results else []
