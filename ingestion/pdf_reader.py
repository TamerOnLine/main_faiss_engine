import os
import logging
import pdfplumber
from PyPDF2 import PdfReader


class PDFReader:
    """Class to read text from PDF files with a fallback to pdfplumber."""

    def __init__(self, path: str):
        """
        Initialize PDFReader with a file path.

        Args:
            path (str): The file path of the PDF.
        """
        self.path = path

    def read_text(self) -> str:
        """
        Extract text from the PDF file using PyPDF2 and fallback to pdfplumber.

        Returns:
            str: Extracted text from the PDF file.
        """
        text = ""

        try:
            with open(self.path, "rb") as file:
                pdf = PdfReader(file)
                text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages if page.extract_text()
                )

            if not text.strip():
                logging.warning(
                    "No text extracted from %s using PyPDF2. Falling back to pdfplumber.",
                    self.path,
                )
                with pdfplumber.open(self.path) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        except Exception as error:
            logging.error("Error reading %s: %s", self.path, error)

        return text.strip()

    @staticmethod
    def load_from_folder(folder_path: str) -> dict:
        """
        Load all PDF files from a specified folder and extract their texts.

        Args:
            folder_path (str): The path to the folder containing PDFs.

        Returns:
            dict: A dictionary where keys are file paths and values are extracted texts.
        """
        if not os.path.exists(folder_path):
            logging.error("Path '%s' does not exist!", folder_path)
            return {}

        pdf_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")
        ]

        if not pdf_files:
            logging.warning("No PDF files found in the specified folder.")
            return {}

        logging.info("Found %d PDF file(s) in %s.", len(pdf_files), folder_path)

        pdf_texts = {}
        for pdf_path in pdf_files:
            logging.info("Processing file: %s", pdf_path)
            reader = PDFReader(pdf_path)
            text = reader.read_text()
            if text:
                pdf_texts[pdf_path] = text
            else:
                logging.warning("File %s is empty or contains no extractable text.", pdf_path)

        return pdf_texts
