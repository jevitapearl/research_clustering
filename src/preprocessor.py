import pandas as pd
import PyPDF2

class PDFPreprocessor:
    def __init__(self, uploaded_files):
        self.files = uploaded_files

    def extract_text_from_pdf(self, file_obj):
        """Reads a PDF file object and returns text."""
        try:
            pdf_reader = PyPDF2.PdfReader(file_obj)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception:
            return ""

    def process(self):
        """Iterates through uploaded files and returns a DataFrame."""
        documents = []
        filenames = []

        for uploaded_file in self.files:
            file_type = uploaded_file.name.split('.')[-1].lower()
            text = ""

            # Reset file pointer to beginning just in case
            uploaded_file.seek(0)

            if file_type == 'pdf':
                text = self.extract_text_from_pdf(uploaded_file)
            elif file_type == 'txt':
                text = str(uploaded_file.read(), "utf-8")

            # Simple cleaning: remove excessive newlines
            text = text.replace('\n', ' ').strip()

            # Only add if we actually found text (min 50 chars)
            if len(text) > 50:
                documents.append(text)
                filenames.append(uploaded_file.name)

        return pd.DataFrame({
            "Filename": filenames,
            "Content": documents
        })