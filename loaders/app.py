from loaders.document_loader import load_document

def main():
    text = load_document(r"D:\Books DSA\coding-interview-patterns-nail-your-next-coding-interview.pdf")
    print(text)

# so that running `python -m loaders.app` still works too
if __name__ == "__main__":
    main()
