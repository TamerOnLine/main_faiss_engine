import sqlite3
import datetime
import logging
from tools.faiss_engine_tool import FaissSearchEngine
from model.ollama_model import OllamaHandler  # استيراد Ollama


class Agent:
    """Agent that integrates FaissSearchEngine with Ollama AI model."""

    def __init__(self, db_path="chat_history.db"):
        """
        Initialize the agent with FaissSearchEngine and Ollama.

        Args:
            db_path (str): Path to the SQLite database.
        """
        logging.info("Initializing Agent...")
        self.engine = FaissSearchEngine()
        self.ollama = OllamaHandler()  # تهيئة Ollama
        self.db_path = db_path
        self.init_db()
        logging.info("Agent initialized successfully.")

    def init_db(self):
        """Create necessary tables in the database if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    question TEXT,
                    processed_question TEXT,
                    answer TEXT
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS error_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    error_message TEXT,
                    query TEXT
                )
                """
            )

            conn.commit()

    def log_interaction(self, query, processed_query, response):
        """
        Log conversations.

        Args:
            query (str): The user's original query.
            processed_query (str): The query after processing with Ollama.
            response (str): The agent's response.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO chat_log (timestamp, question, processed_question, answer) 
                VALUES (?, ?, ?, ?)
                """,
                (timestamp, query, processed_query, response),
            )
            conn.commit()


    def process_query(self, query):
        """
        Process the user query using Ollama AI and FaissSearchEngine, while logging each step.
        """
        try:
            logging.info("🔍 [Step 1] Received user query: %s", query)

            # 🔹 **تحليل الاستعلام باستخدام Ollama**
            logging.info("🧠 [Step 2] Sending query to Ollama for optimization...")
            processed_query = self.ollama.explain_question_mark(
                f"Rewrite this query in a more effective way for search: {query}"
            )
            logging.info("✅ [Step 3] Optimized query from Ollama: %s", processed_query)

            # 🔹 **البحث في FAISS**
            logging.info("📂 [Step 4] Searching FAISS for similar documents...")
            results = self.engine.search(processed_query, top_k=3, min_similarity=0.4)

            if results:
                response = "\n".join([f"{i+1}. {text}" for i, text in enumerate(results, 1)])
                logging.info("✅ [Step 5] FAISS search completed. Found %d results.", len(results))
            else:
                response = "⚠ No relevant results found."
                logging.warning("⚠ [Step 5] FAISS search completed. No results found.")

            # 🔹 **تسجيل التفاعل في قاعدة البيانات**
            logging.info("📝 [Step 6] Logging interaction to database...")
            self.log_interaction(query, processed_query, response)

            logging.info("🎯 [Step 7] Returning response to user.")
            return response

        except Exception as e:
            logging.error("❌ [ERROR] An error occurred: %s", str(e))
            return "An error occurred while processing your request."





