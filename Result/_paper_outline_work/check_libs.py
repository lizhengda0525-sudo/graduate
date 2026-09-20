import importlib.util

for name in ["pypdf", "PyPDF2", "pdfplumber", "fitz", "pdfminer"]:
    spec = importlib.util.find_spec(name)
    print(name, bool(spec))
