# RAG File QA System

A powerful Retrieval-Augmented Generation (RAG) system for file-based Question Answering. This system enables you to load documents (PDF, text, etc.), process them through chunking and embedding, and query them intelligently using Large Language Models (LLMs).

## Overview

The RAG File QA System combines document retrieval with generative AI to provide accurate answers based on your document corpus. Instead of relying solely on an LLM's training data, this system retrieves relevant content from your files and uses it to generate contextually accurate answers.

## Features

- 📄 **Multi-Format Support**: Load documents in PDF, TXT, and other text formats
- 🔗 **Intelligent Chunking**: Split documents into optimal chunks for better retrieval
- 🧠 **Vector Embeddings**: Convert text into embeddings for semantic search
- 🤖 **LLM Integration**: Query documents using state-of-the-art language models
- 🔍 **Semantic Search**: Find relevant content based on meaning, not just keywords
- ⚡ **Efficient Processing**: Optimized pipeline for fast document processing and querying

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Aishwarya-Karwal/RAG_File_QA_system.git
cd RAG_File_QA_system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from rag_system import RAGFileQA

# Initialize the system
rag = RAGFileQA()

# Load documents
rag.load_documents('path/to/documents')

# Ask a question
answer = rag.query("Your question here")
print(answer)
```

### Loading Documents

```python
# Load a single document
rag.load_document('document.pdf')

# Load multiple documents from a directory
rag.load_documents('docs_folder/')
```

### Querying

```python
# Simple query
result = rag.query("What is the main topic?")

# Query with custom parameters
result = rag.query(
    question="What is the main topic?",
    top_k=5,  # Number of relevant chunks to retrieve
    temperature=0.7
)
```

## Architecture

The system follows this pipeline:

1. **Document Loading**: Read and parse input files
2. **Chunking**: Split documents into manageable chunks
3. **Embedding**: Convert chunks into vector embeddings
4. **Indexing**: Store embeddings in a searchable index
5. **Retrieval**: Find relevant chunks based on query
6. **Generation**: Use LLM to generate answers from retrieved context

## Configuration

Create a `config.py` file to customize the system:

```python
# Document settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM settings
LLM_MODEL = "gemini-flash-latest"
TEMPERATURE = 0.7
MAX_TOKENS = 500

# Retrieval settings
TOP_K = 5
```

## Supported File Formats

- `.pdf` - PDF documents
- `.txt` - Plain text files
- `.md` - Markdown files
- `.docx` - Word documents (with additional dependencies)

## Requirements

Core dependencies:
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector similarity search
- `pypdf` - PDF processing
- `python-dotenv` - Environment variable management

See `requirements.txt` for complete list with versions.

## API Reference

### RAGFileQA Class

#### Methods

- `load_document(file_path)` - Load a single document
- `load_documents(directory_path)` - Load all documents from a directory
- `query(question, top_k=5, temperature=0.7)` - Ask a question and get an answer
- `clear_index()` - Clear the current index
- `get_relevant_chunks(question, top_k=5)` - Get relevant chunks without generating answer

## Performance Tips

1. **Chunk Size**: Use appropriate chunk size based on your documents (500-1500 tokens works well)
2. **Embedding Model**: Choose lightweight models for speed or powerful models for accuracy
3. **Batch Processing**: Process multiple documents together for efficiency
4. **Indexing**: Use FAISS or similar for fast vector search

## Troubleshooting

### Out of Memory Issues
- Reduce chunk size
- Process documents in smaller batches
- Use a more lightweight embedding model

### Slow Query Times
- Verify FAISS index is properly built
- Check chunk size settings
- Consider using GPU acceleration for embeddings

### Low Answer Quality
- Increase `top_k` for more context
- Adjust chunk size and overlap
- Use a more powerful LLM model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Embeddings powered by [Sentence Transformers](https://www.sbert.net/)
- Vector search by [FAISS](https://github.com/facebookresearch/faiss)

## Support

For issues and questions:
- Open an [Issue](https://github.com/Aishwarya-Karwal/RAG_File_QA_system/issues)
- Check existing documentation
- Review examples in the `examples/` directory

## Roadmap

- [ ] Web interface
- [ ] Support for more file formats (Excel, CSV)
- [ ] Multi-language support
- [ ] Advanced caching mechanisms
- [ ] Streaming responses

---

**Built with ❤️ using Python**
