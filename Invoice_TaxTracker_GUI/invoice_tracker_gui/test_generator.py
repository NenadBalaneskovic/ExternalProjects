from fpdf import FPDF
import random
import os

VENDORS = ["AcmeCorp", "Globex", "Initech", "Umbrella",
           "Soylent", "Stark Industries"]
CATEGORIES = ["Office", "Travel", "Meals", "Supplies",
              "Consulting", "Software"]


def generate_test_pdf(filename, vendor, amount, category, tampered=False,
                      month="January"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Invoice Document", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Vendor: {vendor}", ln=True)
    pdf.cell(200, 10, txt=f"Total Amount: ${amount:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Category: {category}", ln=True)
    pdf.cell(200, 10, txt=f"Month: {month}", ln=True)

    if tampered:
        pdf.cell(200, 10, txt="Note: This document has been altered.", ln=True)
        pdf.cell(200, 10, txt="Font mismatch detected.", ln=True)

    os.makedirs("test_docs", exist_ok=True)
    path = os.path.join("test_docs", filename)
    pdf.output(path)
    return path


def generate_batch(n=10, tamper_ratio=0.3, month="January"):
    paths = []
    for i in range(n):
        vendor = random.choice(VENDORS)
        category = random.choice(CATEGORIES)
        amount = round(random.uniform(10, 500), 2)
        tampered = random.random() < tamper_ratio
        filename = f"test_invoice_{month}_{i+1}_{'tampered' if tampered else 'clean'}.pdf"
        path = generate_test_pdf(filename, vendor, amount, category,
                                 tampered, month)
        paths.append(path)
    return paths
