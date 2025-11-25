import re

def normalize(value):
    try:
        value = value.replace("€", "").replace(".", "").replace(",", ".").strip()
        return float(value)
    except Exception:
        return None

def has_address_after(label, raw, max_gap_lines=6):
    lines = raw.splitlines()
    try:
        start_idx = next(i for i, ln in enumerate(lines) if re.search(rf"{label}", ln, re.IGNORECASE))
    except StopIteration:
        return False

    window = lines[start_idx+1:start_idx+1+max_gap_lines]
    addr_patterns = [
        r"[A-Za-zÄÖÜäöüß]+\s+\d+[A-Za-z]?",   # Street + number
        r"\b\d{5}\b",                         # Postal code
        r"\bFrankfurt\b|\bBerlin\b|\bMünchen\b|\bHamburg\b",  # Cities
        r"\bDE\b|\bDeutschland\b|\bGermany\b"                 # Country
    ]
    return any(re.search(p, ln) for ln in window for p in addr_patterns)

def detect_anomalies(parsed):
    anomalies = []
    raw = parsed.get("raw_text", "")
    amount = parsed.get("amount", 0)
    total_price = parsed.get("total_price", None)

    # Basic field checks
    if not parsed.get("invoice_id"):
        anomalies.append("Missing invoice number.")
    if not parsed.get("vendor"):
        anomalies.append("Missing vendor name.")
    if amount is None or amount < 0.01:
        anomalies.append("Missing or invalid amount.")

    # Gesamtpreis presence and plausibility
    if total_price is None:
        if "Gesamtpreis" in raw:
            anomalies.append("Gesamtpreis label found but amount missing or unreadable")
        else:
            anomalies.append("Gesamtpreis not found (possible formatting issue).")
    else:
        if total_price > 10000:
            anomalies.append(f"Unrealistic Gesamtpreis detected: {total_price:.2f} EUR")
        elif total_price > 1000:
            anomalies.append(f"Implausible Gesamtpreis: {total_price:.2f} EUR")

    # Suspicious formatting of Gesamtpreis
    if re.search(r"Gesamtpreis.*\b\d{5,}€\b", raw):
        anomalies.append("Suspicious Gesamtpreis formatting: missing thousands/decimal separators")

    # Mismatch between Gesamtpreis and Zahlbetrag
    if total_price is not None and amount is not None:
        if abs(total_price - amount) > 2.0:
            anomalies.append(f"Zahlbetrag ({amount:.2f}) and Gesamtpreis ({total_price:.2f}) mismatch")

    # OCR discrepancy check
    if "ocr_total_price" in parsed and parsed["ocr_total_price"] is not None:
        ocr_val = parsed["ocr_total_price"]
        if total_price is not None and abs(total_price - ocr_val) > 2.0:
            anomalies.append(
                f"Discrepancy between text layer ({total_price:.2f}) and OCR ({ocr_val:.2f}) detected"
            )
        if ocr_val > 10000:
            anomalies.append(f"OCR detected implausible Gesamtpreis: {ocr_val:.2f} EUR")

    # VAT consistency check
    vat_match = re.search(r"USt\.\s*%.*?([0-9]+(?:[.,][0-9]{2})?) ?€", raw)
    subtotal_match = re.search(r"Zwischensumme.*?([0-9]+(?:[.,][0-9]{2})?) ?€", raw)
    if vat_match and subtotal_match:
        vat = normalize(vat_match.group(1))
        subtotal = normalize(subtotal_match.group(1))
        expected_vat = round(subtotal * 0.19, 2)
        if vat is not None and subtotal is not None and abs(vat - expected_vat) > 0.5:
            anomalies.append(f"VAT mismatch: expected {expected_vat:.2f}, found {vat:.2f}")

    # Empty address blocks
    if "Rechnungsadresse" in raw and "Lieferadresse" in raw:
        billing_ok = has_address_after("Rechnungsadresse", raw)
        shipping_ok = has_address_after("Lieferadresse", raw)
        if not billing_ok or not shipping_ok:
            anomalies.append("Empty billing or shipping address block detected")

    return anomalies
