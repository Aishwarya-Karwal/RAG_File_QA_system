import os
import fitz

def load_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def load_pdf_file(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        return load_text_file(file_path)
    elif ext == '.pdf':
        return load_pdf_file(file_path)
    else:
        raise ValueError(f"Unsupported file type : {ext}")