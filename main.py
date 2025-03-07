import logging
from pathlib import Path
from agents.agent import Agent
from tools.faiss_engine_tool import FaissSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define the path for the PDF folder
BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / "data" / "pdf"

def main() -> None:
    """Runs an interactive search session using the enhanced Agent with FAISS."""
    logging.info("Initializing Agent...")
    agent = Agent()

    # Initialize FAISS engine
    logging.info("Checking for new PDFs to load into FAISS...")
    engine = FaissSearchEngine()

    # Ensure the folder exists
    if not PDF_FOLDER.exists():
        logging.warning("PDF folder does not exist. Creating it now...")
        PDF_FOLDER.mkdir(parents=True, exist_ok=True)

    # Get list of PDFs
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if pdf_files:
        logging.info(f"Found {len(pdf_files)} PDF files. Loading into FAISS...")
        pdf_data = engine.load_data(str(PDF_FOLDER))
        engine.update_index(pdf_data)
        logging.info("FAISS index updated successfully.")

        # **عرض بعض البيانات المخزنة في FAISS لمعرفة محتوى البحث**
        logging.info("\n🔍 **Sample data from FAISS index:**")
        for i, chunk in enumerate(engine.chunks_list[:5]):  # عرض 5 عينات فقط
            logging.info(f"{i+1}. {chunk[:300]}...")  # عرض أول 300 حرف فقط
    else:
        logging.warning("No PDF files found. Please add documents to improve search results.")

    while True:
        query_text = input("\nEnter search text (or type 'exit' to quit): ").strip()

        if query_text.lower() == "exit":
            logging.info("Session terminated.")
            break

        if not query_text:
            logging.warning("Warning: You cannot enter an empty text. Please try again.")
            continue

        response = agent.process_query(query_text)
        print(response)

if __name__ == "__main__":
    main()
