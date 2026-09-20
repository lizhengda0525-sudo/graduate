import sys

import pypdf

pdf_path = (
    "G:/graduate/Paper/"
    "A_Nonlinear_Granger_Causality_Learning_Approach_via_Sparse_Component-Wise_MLP_"
    "for_iEEG-Based_Epileptogenic_Zone_Localization.pdf"
)
out_path = "G:/graduate/Result/_paper_outline_work/paper_text.txt"

reader = pypdf.PdfReader(pdf_path)
with open(out_path, "w", encoding="utf-8") as handle:
    for index, page in enumerate(reader.pages):
        handle.write(f"===== PAGE {index + 1} =====\n")
        handle.write(page.extract_text() or "")
        handle.write("\n")
print("pages", len(reader.pages))
print("saved", out_path)
