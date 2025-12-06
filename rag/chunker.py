from typing import List

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks with overlap.
    
    Args:
        text (str): The full document text
        chunk_size (int): Max characters per chunk
        overlap (int): Overlap between chunks to preserve context
    
    Returns:
        List[str]: List of text chunks
    """

    if chunk_size <= overlap:
        raise ValueError("chunk size must be greater than overlap")
    
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += (chunk_size - overlap) # move ahead but keep overlap

    return chunks