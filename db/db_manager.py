import sqlite3
import faiss
import logging
import pickle
import numpy as np


class DBManager:
    """Class to handle SQLite database operations for storing FAISS index and PDF metadata."""

    def __init__(self, db_path="faiss_db.sqlite"):
        """Initialize the database and ensure the table exists."""
        self.db_path = db_path
        self._initialize_db()

    def load_index(self):
        """Load FAISS index and chunks_list from SQLite database."""
        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT index_data, chunks_list FROM faiss_index ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()

            if result and result[0]:
                index_data = np.frombuffer(result[0], dtype=np.uint8)
                index = faiss.deserialize_index(index_data)
                chunks_list = pickle.loads(result[1]) if result[1] else []
                logging.info("FAISS index and chunks_list successfully loaded from database.")
                return index, chunks_list

            logging.warning("No FAISS index found in database. A new one will be created.")
            return None, []

    def _initialize_db(self):
        """Create the necessary tables if they don't exist."""
        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS faiss_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    index_data BLOB,
                    chunks_list BLOB
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pdf_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE,
                    content TEXT,
                    last_modified REAL
                )
                """
            )
            conn.commit()

    def save_pdf_content(self, filename, content, last_modified):
        """Save or update PDF content only if there is a change."""
        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content, last_modified FROM pdf_files WHERE filename = ?", (filename,))
            result = cursor.fetchone()

            if result:
                old_content, old_modified = result
                if old_modified >= last_modified:
                    logging.info(f"No change in '{filename}', update not required.")
                    return
                if old_content == content:
                    logging.info(f"Content of '{filename}' has not changed, update not required.")
                    return

            cursor.execute(
                "INSERT OR REPLACE INTO pdf_files (filename, content, last_modified) VALUES (?, ?, ?)",
                (filename, content, last_modified)
            )
            conn.commit()
            logging.info(f"PDF '{filename}' has been updated.")

    def get_tracked_pdfs(self):
        """Retrieve all stored PDFs and their last modified timestamps."""
        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, last_modified FROM pdf_files")
            pdfs = {row[0]: row[1] for row in cursor.fetchall()}
            return pdfs

    def save_index(self, index, chunks_list):
        """Save FAISS index and chunks_list to the SQLite database."""
        if index is None:
            logging.warning("No FAISS index to save!")
            return

        index_data = faiss.serialize_index(index)
        chunks_list_data = pickle.dumps(chunks_list)

        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM faiss_index")  # Delete old index
            cursor.execute("INSERT INTO faiss_index (index_data, chunks_list) VALUES (?, ?)", (index_data, chunks_list_data))
            conn.commit()
            logging.info("FAISS index and chunks_list successfully saved to the database.")
