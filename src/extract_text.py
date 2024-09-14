import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Example usage:
if __name__ == "__main__":
    text = extract_text_from_pdf("/Users/mustavikhan/Desktop/chatbot/data/book.pdf")
    with open("/Users/mustavikhan/Desktop/chatbot/data/extracted_physics_book.txt", "w") as f:
        f.write(text)