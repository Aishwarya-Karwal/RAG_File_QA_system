import os
import fitz

def load_text_file(file_path: str) -> list[dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        return [{
            "text":text,
            "metadata":{
                "source":os.path.basename(file_path),
                "page_number" : 1
            }
        }]
    
def load_pdf_file(file_path: str) -> list[dict]:
    doc = fitz.open(file_path)
    documents = []
    for page_num, page in enumerate(doc, start = 1):
        text = page.get_text().strip()
        if(text):
            documents.append({
                "text":text,
                "metadata":{
                    "source":os.path.basename(file_path),
                    "page_number":page_num
                }
            })
    return documents

def load_document(file_path: str) -> list[dict]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        return load_text_file(file_path)
    elif ext == '.pdf':
        return load_pdf_file(file_path)
    else:
        raise ValueError(f"Unsupported file type : {ext}")
    

# if __name__ == "__main__":
#     sample_pdf = r"D:\Books DSA\coding-interview-patterns-nail-your-next-coding-interview.pdf"
#     output = load_document(sample_pdf)
#     print(output[:5])