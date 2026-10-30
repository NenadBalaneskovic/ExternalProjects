import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

def normalize_amount(value: str) -> float:
    value = value.replace("€", "").replace(".", "").replace(",", ".").strip()
    try:
        return float(value)
    except Exception:
        return 0.0

def extract_invoice_id(lines):
    for i, line in enumerate(lines):
        if "Rechnungsnummer" in line:
            match = re.search(r"Rechnungsnummer\s*[:\-]?\s*(\S+)", line)
            if match:
                return match.group(1).strip()
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r"\S+", next_line):
                    return next_line
    return ""

def extract_amount(lines, keyword="Zahlbetrag", window=5):
    for i, line in enumerate(lines):
        if keyword.lower() in line.lower():
            for j in range(i, min(i + window, len(lines))):
                match = re.search(r"([0-9]+(?:[.,][0-9]{2})?) ?€", lines[j])
                if match:
                    return normalize_amount(match.group(1))
    return 0.0

def extract_total_price_hybrid(page, raw_text):
    """
    Hybrid extractor:
    1. Try raw text near 'Gesamtpreis'
    2. Fallback to word-level tokens
    3. Fallback to OCR (English default) if nothing plausible found
    """
    lines = raw_text.splitlines()
    euro_pat = re.compile(r"([0-9]{1,6}(?:[.,][0-9]{2})?) ?€|([0-9]{5,})€")

    # Step 1: raw text window
    for i, ln in enumerate(lines):
        if "Gesamtpreis" in ln:
            window = lines[i:i+8]
            candidates = []
            for w in window:
                m = euro_pat.search(w)
                if m:
                    raw_val = m.group(1) or m.group(2)
                    val = normalize_amount(raw_val)
                    candidates.append(val)
            if candidates:
                return max(candidates)

    # Step 2: word-level tokens
    words = page.get_text("words")
    for idx, w in enumerate(words):
        token = w[4]
        if re.match(r"^\d{5,}€$", token):  # catches '90893€'
            return normalize_amount(token.replace("€", ""))

    # Step 3: OCR fallback (English default)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    try:
        ocr_text = pytesseract.image_to_string(img)  # defaults to English
    except Exception:
        return None

    # Look for any large euro values in OCR text
    m = re.search(r"\b([0-9]{5,}) ?€", ocr_text)
    if m:
        raw_val = m.group(1)
        return normalize_amount(raw_val)

    return None

def parse_document(path):
    doc = fitz.open(path)
    extracted = {
        "vendor": "",
        "amount": 0.0,
        "total_price": None,
        "ocr_total_price": None,
        "invoice_id": "",
        "date": "",
        "raw_text": ""
    }

    vendor_candidates = []

    for page in doc:
        text = page.get_text("text")
        extracted["raw_text"] += text + "\n"
        lines = text.splitlines()

        if not extracted["invoice_id"]:
            extracted["invoice_id"] = extract_invoice_id(lines)

        if extracted["amount"] == 0.0:
            extracted["amount"] = extract_amount(lines)

        if extracted["total_price"] is None:
            extracted["total_price"] = extract_total_price_hybrid(page, extracted["raw_text"])

        # Always store OCR separately for fraud discrepancy check
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            ocr_text = pytesseract.image_to_string(img)  # English default
            m = re.search(r"\b([0-9]{5,}) ?€", ocr_text)
            if m:
                raw_val = m.group(1)
                extracted["ocr_total_price"] = normalize_amount(raw_val)
        except Exception:
            extracted["ocr_total_price"] = None

        vendor_matches = re.findall(r"Verkauft von\s+([^\n]+)", text)
        if vendor_matches:
            vendor_candidates.extend(vendor_matches)

    if vendor_candidates:
        extracted["vendor"] = max(vendor_candidates, key=len)

    return extracted
