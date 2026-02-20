import zipfile
import xml.etree.ElementTree as ET
import sys
import os


def get_docx_text(path):
    """Extract text from a .docx file."""
    try:
        with zipfile.ZipFile(path, 'r') as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            namespace = {
                'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text_elements = tree.findall('.//w:t', namespace)
            return "".join(t.text for t in text_elements if t.text)
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    output_file = "all_extracted_text.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for path in ["Project Report.docx", "Task2WriteUp.docx", "WriteUp.docx"]:
            if os.path.exists(path):
                f.write(f"--- START OF {path} ---\n")
                f.write(get_docx_text(path))
                f.write(f"\n--- END OF {path} ---\n\n")
            else:
                f.write(f"File not found: {path}\n")
    print(f"Done. Output in {output_file}")
