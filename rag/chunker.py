from typing import List, Dict

def chunk_text(text_lst: str, file_name: str, chunk_size: int = 2000, overlap: int = 300) -> List[Dict]:
    """
    Splits text into chunks with overlap and attaches metadata.
    
    Args:
        text_lst (list of dict): The full document text with metadata like source doc and page number
        chunk_size (int): Max characters per chunk
        overlap (int): Overlap between chunks to preserve context
    
    Returns:
        List[dict]: List of dict - {"text": chunk_text, "metadata": {"file_name":..., "page_number":..., "chunk_index":...}}
    """

    if chunk_size <= overlap:
        raise ValueError("chunk size must be greater than overlap")
    
    chunks = []
    for page in text_lst:
        start = 0
        text = page['text']
        meta = page['metadata']
        page_number = meta.get("page_number", None)
        text_length = len(text)
        chunk_index = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]

            metadata = {
                "source" : file_name,
                "page_number": page_number,
                "chunk_id": f"{file_name}_page{page_number}_chunk{chunk_index}"
            }

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": metadata
                }
            )
            chunk_index += 1
            start += (chunk_size - overlap) # move ahead but keep overlap

    return chunks