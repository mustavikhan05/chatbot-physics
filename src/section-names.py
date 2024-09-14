import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def extract_chapters_and_sections_with_content(text):
    """
    Extracts chapters, sections, and their content.
    Formats them directly into a dictionary format.
    """
    # Regular expression to detect chapters and sections
    chapter_pattern = re.compile(r'(Chapter\s*(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty))\s+([\w\s]+)', re.IGNORECASE)
    section_pattern = re.compile(r'(\d{1,2}\.\d{1,2}\.?)\s+([\w\s]+)')

    chapters_and_sections = {}
    current_section = None
    current_content = []
    
    # Split text by lines and parse sections
    lines = text.splitlines()
    for line in lines:
        line = line.strip()

        # Check for chapters
        chapter_match = chapter_pattern.match(line)
        if chapter_match:
            if current_section and current_content:
                # Save the previous section's content
                chapters_and_sections[current_section] = " ".join(current_content).strip()
                current_content = []

            # Create the chapter key
            chapter = chapter_match.group(1).capitalize() + " " + chapter_match.group(3).strip()
            current_section = chapter
            chapters_and_sections[current_section] = []
            continue

        # Check for sections and subsections
        section_match = section_pattern.match(line)
        if section_match:
            section_number = section_match.group(1)  # E.g., 12.1 or 12.1.
            section_title = section_match.group(2).strip()  # The section title

            # Save the previous section's content
            if current_section and current_content:
                chapters_and_sections[current_section] = " ".join(current_content).strip()
                current_content = []

            # Combine the section number and title
            current_section = f"{section_number} {section_title}"
            chapters_and_sections[current_section] = []
            continue

        # Skip over examples
        if line.lower().startswith('example'):
            continue

        # Collect content under the current section
        if current_section:
            current_content.append(line.strip())

    # Add the final section's content
    if current_section and current_content:
        chapters_and_sections[current_section] = " ".join(current_content).strip()

    return chapters_and_sections

def extract_from_specific_page_range(pdf_path, start_page, end_page):
    """Extracts text from a specific range of pages in the PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(start_page, end_page):
        page = doc.load_page(page_num)  # load each page in range
        text += page.get_text("text")
    return text

if __name__ == "__main__":
    # Path to your PDF file
    pdf_path = "/Users/mustavikhan/Desktop/chatbot/data/book.pdf"

    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Extract chapters, sections, and content
    chapters_and_sections_with_content = extract_chapters_and_sections_with_content(text)

    # If Chapter 12 and onwards are missing, handle page-specific extraction for them
    if 'Chapter twelve' not in chapters_and_sections_with_content:
        # Manually specify the page range where Chapter 12 onwards begins
        chapter_12_text = extract_from_specific_page_range(pdf_path, start_page=120, end_page=140)
        # Append the extracted text to the main text and re-run the extraction for those chapters
        text += chapter_12_text
        chapters_and_sections_with_content.update(extract_chapters_and_sections_with_content(chapter_12_text))

    # Output the formatted chapters and sections for direct use
    output_path = "/Users/mustavikhan/Desktop/chatbot/data/chapters_and_sections_for_chunking.txt"
    with open(output_path, "w") as f:
        f.write("chapters = {\n")
        for section, content in chapters_and_sections_with_content.items():
            # Format the section and content for pasting into the dictionary
            f.write(f'    "{section}": """{content}""",\n\n')
        f.write("}\n")

    print(f"Formatted chapters and sections have been saved to '{output_path}'.")