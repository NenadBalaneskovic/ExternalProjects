# 1. 🚀 Project Introduction: Tax/Invoice Tracker & Fraud Detector GUI (Parsing & OCR)

## Objective  
The Invoice and Tax Tracker + Fraud Detector project was conceived as a modular, auditable framework for managing financial documents in environments where compliance, 
transparency, and fraud prevention are paramount. In today’s business landscape, organizations face increasing pressure to ensure that their financial reporting is not 
only accurate but also defensible under regulatory scrutiny. This project addresses that need by combining robust document parsing, anomaly detection, visualization, 
and reporting into a single, user-friendly application.

At its core, the system ingests invoices in PDF format and transforms them into structured, machine-readable data. The parsing engine leverages PyMuPDF for text extraction, 
regex rules for field identification, and OCR fallbacks to capture tampered or hidden values. This hybrid approach ensures resilience against common manipulation techniques, 
such as altering only the graphical layer of a PDF while leaving the text layer intact. By normalizing amounts and preserving raw text, the parser provides both clean data for 
downstream analysis and a transparent audit trail for validation.

Once parsed, invoices are passed through the fraud detection module. This component applies a series of plausibility and consistency checks designed to surface anomalies that may 
indicate fraud or error. Rules include verifying the presence of mandatory fields, checking whether totals are realistic, comparing paid amounts against total prices, detecting 
suspicious formatting, and validating VAT calculations. The module also compares text-layer values against OCR results to catch discrepancies introduced by tampering. Each anomaly 
is reported in plain language, making the system’s findings explainable and defensible to auditors, executives, and regulators alike.

The graphical user interface, built with PyQt5, orchestrates the workflow and presents results in an accessible manner. Users can import invoices, toggle fraud detection, run batch 
or single-document analyses, and view anomalies in both textual and visual formats. A fraud heatmap translates anomaly reports into intuitive color-coded grids, allowing users to 
quickly identify which fields are problematic and how severe the issues are. Debug Mode provides transparency by exposing raw parsed data and anomaly lists, reinforcing the system’s 
commitment to auditability.

Beyond detection, the project emphasizes reporting and compliance. The exporter module enables users to save parsed results to CSV and generate tax summaries that distinguish between 
valid and flagged invoices. This functionality supports both operational workflows and regulatory reporting requirements. To facilitate demos and training, a test generator creates 
synthetic invoices with configurable ratios of clean and tampered cases, ensuring that the system can be validated under controlled conditions.

The overarching aim of the project is to demonstrate how advanced analytics and physics-inspired rigor can be embedded into consulting-ready tools that balance technical depth with 
business clarity. By foregrounding transparency, explainability, and reproducibility, the system provides a defensible framework for fraud detection and tax tracking. It is designed 
not only to catch anomalies but also to communicate them clearly, bridging the gap between technical analysis and executive decision-making. In doing so, the project showcases how 
modular AI/ML components can be integrated into client-facing environments to deliver measurable business impact while maintaining compliance and trust 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/LinearProgramming_GUI/LinearProgramming_GUI.md#8--references) 1 - 3 below). 

## 1.1 🎯 **Primary Aim**

Structure and implement a PyQt5-powered GUI for detecting fraudulent content in invoices and tax forms with built-in logic to route the problem to regex and OCR parsers prior to 
subjecting such parsed output to fraud detection modules, both from a **UX** and **reporting (compliance) architecture** perspective.

## 1.2 🧩 Modular Components and Their Roles

### 🧩 GUI Architecture Overview

The interface is divided into three main panels:
1. **Left Panel** – Document Import & Fraud Toggle  
2. **Center Panel** – Document Review & Anomaly Report  
3. **Right Panel** – Summary Table, Export Options & Heatmap

#### 🔹 1. Left Panel: Document Import & Fraud Toggle

##### 📂 File Import
- **Import PDF**: Opens a file dialog to select one or more invoice/receipt PDFs.
- **Connect Email**: Initiates email integration (e.g., IMAP or Outlook API) to fetch receipts from inbox.

##### 📋 File List Display
- **PLACE FILE LIST**: A scrollable area showing imported files with status indicators (e.g., parsed, flagged, error).

##### 🛡️ Fraud Detection Toggle
- **Enable Fraud Detection Mode**: Checkbox that activates anomaly checks and visual heatmap generation.

##### 🟢 Submit Button
- Begins parsing and analysis of all imported documents using selected options.

#### 🔸 2. Center Panel: Document Review & Anomaly Report

##### 🗂️ Tabs
- **Document Tab**: Displays raw document preview (PDF snapshot or OCR text).
- **Fields Tab**: Shows extracted fields like vendor, amount, date, category.

##### 📤 Upload Fields
- **Upload Invoice / Receipt / Other Doc**: Optional manual upload fields for specific document types.

##### 🧠 Analyzing Options
- **Extract Vendor**: Enables vendor name extraction via NLP or regex.
- **Extract Amount**: Parses total, subtotal, and tax values.
- **Extract Category**: Classifies expense type (e.g., travel, meals, office supplies).

##### 🔍 Fraud Detect Settings
- **Check Authenticity**: Runs heuristic and visual checks for tampering, mismatched totals, or duplicate entries.

##### 🔥 Visual Fraud Heatmap
- Displays a color-coded overlay indicating suspicious regions in the document:
  - **Red**: High anomaly score (e.g., altered totals)
  - **Yellow**: Moderate concern (e.g., missing fields)
  - **Blue/Green**: Low risk or clean areas

##### 📑 Anomaly Report
- Textual summary of detected issues:
  - “Invoice #123 has mismatched totals”
  - “Vendor name not recognized”
  - “Font inconsistency detected in amount field”

##### 🟢 Submit / ❌ Close Buttons
- **Submit**: Re-analyzes current document with updated settings.
- **Close**: Clears current document or exits the review panel.

#### 🔹 3. Right Panel: Summary & Export

##### 📊 Parsed Files Table
- Displays all processed documents with columns:
  - File Name
  - Vendor
  - Amount
  - Status (e.g., Clean, Flagged, Error)

##### 📤 Export Options
- **Export CSV**: Saves parsed data for accounting or audit purposes.
- **Generate Tax**: Creates a pre-filled tax form or summary report (PDF or structured format).

##### 📡 Status Indicator
- Displays real-time feedback:
  - “3 invoices flagged for review”
  - “Export successful”
  - “Parsing error in Receipt_04.pdf”

Furthermore:

#### 🟦 STATUS Indicator (Bottom Right)

##### 🔍 Purpose:
- Displays **real-time feedback** about the system’s current state or recent actions.

##### 🧠 Typical Messages:
- ✅ “Parsing complete: 5 documents processed”
- ⚠️ “2 invoices flagged for fraud review”
- ❌ “Error: Receipt_04.pdf could not be parsed”
- 📤 “Export successful: invoices_2025.csv saved”

##### 🔄 Behavior:
- Updates dynamically after each major action:
  - File import
  - Fraud detection
  - Export
  - Tax report generation

#### 🟩 Submit Button (Bottom Center)

##### 🔍 Purpose:
- Acts as a **global trigger** to run the full pipeline across all imported documents.

##### 🧠 What It Does:
- Parses all uploaded files
- Extracts vendor, amount, category (if selected)
- Runs fraud detection (if enabled)
- Updates:
  - Parsed Files table
  - Visual Fraud Heatmap
  - Anomaly Report
  - STATUS indicator

##### 🔄 Behavior:
- Can be clicked after importing multiple files
- Resets and reprocesses the entire batch with current settings

##### 🧠 Why Both Buttons Are Useful

- The **Submit button** initiates the work.
- The **STATUS box** tells the user what happened — success, error, or warnings.

##### 🧠 Smart Behaviors

- ✅ Auto-expands constraint fields based on number of variables
- ✅ Auto-selects fraud detection logic if checkbox is enabled
- ✅ Dynamically updates heatmap and anomaly report after each document is parsed
- ✅ Allows manual override of extracted fields before export

##### 🧠 Progress Bar Behavior

- **Indeterminate Mode**: While waiting for backend response (e.g., OCR or parsing)
- **Percentage Mode**: Shows completion % as each file is processed
- **Color Feedback**:
  - Green: Success
  - Yellow: Warnings
  - Red: Errors

 
## 1.3 🧠 **GUI sketch**  

In the following we address our full GUI sketch, a clean, structured layout for our PyQt5-based Tax/Invoice Tracker and Fraud detection GUI. It includes all the key modules we discussed: 
objective and constraint input, method selection (OCR parsing, fraud detection mode), result display, visualization, and diagnostics.

![TaxInvoice_GUI_sketch.png](https://github.com/NenadBalaneskovic/ExternalProjects/blob/64bf360e6cf7aba5fbf0c7e89a06dbcd8e71e2b1/Invoice_TaxTracker_GUI/Invoice_GUI_sketch_.png)

We are ready to scaffold this into actual PyQt5 code and wire up the backend logic for method selection and parser/detector routing.

---

# 2. 🔐 Algorithmic concepts

This is a comprehensive, notebook-ready Markdown explanation of major optimization, parsing and OCR techniques
our GUI should support. Each section includes mathematical foundations, algorithmic steps, and illustrative examples.

## 2.1 📁 Project Structure

```plaintext
invoice_tracker_gui/
├── main.py
├── gui.py
├── parser.py
├── fraud.py
├── heatmap.py
├── exporter.py
├── test_generator.py
└── requirements.txt
```

## 2.2 📄 `main.py`

```python
from PyQt5.QtWidgets import QApplication
from gui import InvoiceTrackerApp
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InvoiceTrackerApp()
    window.show()
    sys.exit(app.exec_())
```

## 2.3 📄 `gui.py` (with toggles and batch simulation)

```python
import os
import hashlib
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QCheckBox, QProgressBar, QTableWidget, QTableWidgetItem, QTabWidget,
    QLineEdit, QTextEdit, QFileDialog, QListWidget, QComboBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from parser import parse_document
from fraud import detect_anomalies
from exporter import export_csv, generate_tax_report
from heatmap import render_heatmap
from test_generator import generate_batch


class InvoiceTrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Invoice and Tax Tracker + Fraud Detector")
        self.setGeometry(100, 100, 1200, 700)
        self.init_ui()
        self.analysis_cache = {}  # Maps absolute file path → (parsed, anomalies)
        self.session_mode = "idle"  # "batch" or "single"

    def reset_app(self, note="Reset"):
        self.session_mode = "idle"
        self.analysis_cache.clear()
        self.file_list.clear()
        self.parsed_table.setRowCount(0)
        self.anomaly_report.clear()
        self.debug_output.clear()
        self.heatmap_canvas.figure.clf()
        self.heatmap_canvas.draw()
        self.status_label.setText(note)
        self.progress_bar.setValue(0)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left Panel
        left_panel = QVBoxLayout()
        self.import_btn = QPushButton("Import PDF")
        self.email_btn = QPushButton("Connect Email")
        self.file_list = QListWidget()
        self.fraud_toggle = QCheckBox("Enable Fraud Detection Mode")
        self.global_submit_btn = QPushButton("Submit")

        self.generate_test_btn = QPushButton("Generate Test Invoices")
        self.test_type_selector = QComboBox()
        self.test_type_selector.addItems(["Mixed", "Clean Only", "Tampered Only"])
        self.month_selector = QComboBox()
        self.month_selector.addItems([
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])

        # Debug toggle
        self.debug_toggle = QCheckBox("Enable Debug Mode")

        left_panel.addWidget(self.import_btn)
        left_panel.addWidget(self.email_btn)
        left_panel.addWidget(self.file_list)
        left_panel.addWidget(self.fraud_toggle)
        left_panel.addWidget(self.global_submit_btn)
        left_panel.addWidget(QLabel("Test Case Data File"))
        left_panel.addWidget(self.test_type_selector)
        left_panel.addWidget(QLabel("Simulate Months"))
        left_panel.addWidget(self.month_selector)
        left_panel.addWidget(self.generate_test_btn)
        left_panel.addWidget(self.debug_toggle)

        # Center Panel
        center_panel = QVBoxLayout()
        self.tabs = QTabWidget()

        # Document Tab
        doc_tab = QWidget()
        doc_layout = QVBoxLayout()
        self.upload_invoice = QLineEdit()
        self.browse_invoice_btn = QPushButton("Browse")
        self.upload_receipt = QLineEdit()
        self.browse_receipt_btn = QPushButton("Browse")
        self.upload_other = QLineEdit()
        self.browse_other_btn = QPushButton("Browse")

        doc_layout.addWidget(QLabel("Upload Invoice:"))
        doc_layout.addLayout(self._file_input_row(self.upload_invoice, self.browse_invoice_btn))
        doc_layout.addWidget(QLabel("Upload Receipt:"))
        doc_layout.addLayout(self._file_input_row(self.upload_receipt, self.browse_receipt_btn))
        doc_layout.addWidget(QLabel("Upload Other Doc:"))
        doc_layout.addLayout(self._file_input_row(self.upload_other, self.browse_other_btn))
        doc_tab.setLayout(doc_layout)

        # Fields Tab
        fields_tab = QWidget()
        fields_layout = QVBoxLayout()
        self.extract_vendor = QCheckBox("Extract Vendor")
        self.extract_amount = QCheckBox("Extract Amount")
        self.extract_category = QCheckBox("Extract Category")
        self.check_authenticity = QCheckBox("Check authenticity")
        self.heatmap_canvas = FigureCanvas(Figure(figsize=(3, 2)))
        self.anomaly_report = QTextEdit()
        self.analyze_submit_btn = QPushButton("Submit")
        self.analyze_close_btn = QPushButton("Close")

        # Debug output panel
        self.debug_output = QTextEdit()
        self.debug_output.setReadOnly(True)

        for box in [self.extract_vendor, self.extract_amount, self.extract_category, self.check_authenticity]:
            fields_layout.addWidget(box)
        fields_layout.addWidget(QLabel("Visual Fraud Heatmap"))
        fields_layout.addWidget(self.heatmap_canvas)
        fields_layout.addWidget(QLabel("Anomaly Report:"))
        fields_layout.addWidget(self.anomaly_report)
        fields_layout.addWidget(QLabel("Debug Output:"))
        fields_layout.addWidget(self.debug_output)
        fields_layout.addWidget(self.analyze_submit_btn)
        fields_layout.addWidget(self.analyze_close_btn)
        fields_tab.setLayout(fields_layout)

        self.tabs.addTab(doc_tab, "Document")
        self.tabs.addTab(fields_tab, "Fields")
        center_panel.addWidget(self.tabs)

        # Right Panel
        right_panel = QVBoxLayout()
        self.parsed_table = QTableWidget()
        self.parsed_table.setColumnCount(4)
        self.parsed_table.setHorizontalHeaderLabels(["File", "Vendor", "Amount", "Status"])
        self.export_btn = QPushButton("Export CSV")
        self.tax_btn = QPushButton("Generate Tax")
        self.status_label = QLabel("STATUS")
        self.progress_bar = QProgressBar()

        right_panel.addWidget(self.parsed_table)
        right_panel.addWidget(self.export_btn)
        right_panel.addWidget(self.tax_btn)
        right_panel.addWidget(self.status_label)
        right_panel.addWidget(self.progress_bar)

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(center_panel, 3)
        main_layout.addLayout(right_panel, 3)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.connect_signals()

    def _file_input_row(self, line_edit, button):
        layout = QHBoxLayout()
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return layout
    def connect_signals(self):
        self.import_btn.clicked.connect(lambda: (self.reset_app("Imported PDFs"), self.import_pdf()))
        self.global_submit_btn.clicked.connect(self.run_pipeline)
        self.export_btn.clicked.connect(self.export_data)
        self.tax_btn.clicked.connect(self.generate_tax)
        self.analyze_submit_btn.clicked.connect(lambda: (self.reset_app("Single-doc analysis"), self.analyze_current_doc()))
        self.generate_test_btn.clicked.connect(lambda: (self.reset_app("Generated test data"), self.generate_test_data()))
        self.browse_invoice_btn.clicked.connect(lambda: self.browse_file(self.upload_invoice))
        self.browse_receipt_btn.clicked.connect(lambda: self.browse_file(self.upload_receipt))
        self.browse_other_btn.clicked.connect(lambda: self.browse_file(self.upload_other))
        self.email_btn.clicked.connect(self.connect_email)
        self.file_list.itemClicked.connect(self.display_selected_file)

    def browse_file(self, target_field):
        path, _ = QFileDialog.getOpenFileName(self, "Select Document", "", "PDF Files (*.pdf)")
        if path:
            target_field.setText(path)

    def import_pdf(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Import PDFs", "", "PDF Files (*.pdf)")
        self.file_list.clear()
        for f in files:
            self.file_list.addItem(f)

    def run_pipeline(self):
        self.session_mode = "batch"
        self.progress_bar.setValue(0)
        self.parsed_table.setRowCount(0)
        self.anomaly_report.clear()
        self.debug_output.clear()
        self.heatmap_canvas.figure.clf()
        self.heatmap_canvas.draw()
        self.analysis_cache.clear()

        for i in range(self.file_list.count()):
            path = os.path.abspath(self.file_list.item(i).text())
            parsed = parse_document(path)
            anomalies = detect_anomalies(parsed) if self.fraud_toggle.isChecked() else []
            self.analysis_cache[path] = (parsed, anomalies)
            self.update_table(path, parsed, anomalies)
            self.progress_bar.setValue(int((i + 1) / self.file_list.count() * 100))

            if self.debug_toggle.isChecked():
                debug_text = f"FILE: {path}\nPARSED:\n{parsed}\nANOMALIES:\n{anomalies}\n\n"
                self.debug_output.append(debug_text)

        self.status_label.setText("Parsing complete.")

    def update_table(self, path, parsed, anomalies):
        row = self.parsed_table.rowCount()
        self.parsed_table.insertRow(row)
        self.parsed_table.setItem(row, 0, QTableWidgetItem(path))
        self.parsed_table.setItem(row, 1, QTableWidgetItem(parsed.get("vendor", "")))
        self.parsed_table.setItem(row, 2, QTableWidgetItem(f"{parsed.get('amount', 0):.2f} €"))
        status = "Flagged" if anomalies else "OK"
        self.parsed_table.setItem(row, 3, QTableWidgetItem(status))

    def render_heatmap(self, anomalies):
        self.heatmap_canvas.figure.clf()
        render_heatmap(self.heatmap_canvas, anomalies)

    def export_data(self):
        export_csv(self.parsed_table)
        self.status_label.setText("Export successful.")

    def generate_tax(self):
        generate_tax_report(self.parsed_table)
        self.status_label.setText("Tax report generated.")
    def analyze_current_doc(self):
        self.session_mode = "single"
        self.parsed_table.setRowCount(0)
        self.anomaly_report.clear()
        self.debug_output.clear()
        self.heatmap_canvas.figure.clf()
        self.heatmap_canvas.draw()

        path = (self.upload_invoice.text() or self.upload_receipt.text() or self.upload_other.text())
        if not path:
            self.status_label.setText("No document selected.")
            return

        path = os.path.abspath(path)
        parsed = parse_document(path)
        anomalies = detect_anomalies(parsed) if self.check_authenticity.isChecked() else []

        self.anomaly_report.setText("\n".join(anomalies))
        render_heatmap(self.heatmap_canvas, anomalies)
        self.heatmap_canvas.draw()

        self.parsed_table.insertRow(0)
        self.parsed_table.setItem(0, 0, QTableWidgetItem(path))
        self.parsed_table.setItem(0, 1, QTableWidgetItem(parsed.get("vendor", "")))
        self.parsed_table.setItem(0, 2, QTableWidgetItem(f"{parsed.get('amount', 0):.2f} €"))
        status = "Flagged" if anomalies else "OK"
        self.parsed_table.setItem(0, 3, QTableWidgetItem(status))

        if self.debug_toggle.isChecked():
            debug_text = f"FILE: {path}\nPARSED:\n{parsed}\nANOMALIES:\n{anomalies}\n\n"
            self.debug_output.setText(debug_text)

        if anomalies:
            self.status_label.setText("Invoice flagged for anomalies.")
        else:
            self.status_label.setText("Invoice appears clean.")

    def generate_test_data(self):
        self.session_mode = "batch"
        self.file_list.clear()
        mode = self.test_type_selector.currentText()
        tamper_ratio = {"Mixed": 0.4, "Clean Only": 0.0, "Tampered Only": 1.0}[mode]
        month = self.month_selector.currentText()
        paths = generate_batch(n=10, tamper_ratio=tamper_ratio, month=month)
        for path in paths:
            self.file_list.addItem(os.path.abspath(path))
        self.status_label.setText(f"Test invoices for {month} generated.")

    def display_selected_file(self, item):
        path = os.path.abspath(item.text())

        self.anomaly_report.clear()
        self.debug_output.clear()
        self.heatmap_canvas.figure.clf()
        self.heatmap_canvas.draw()
        self.parsed_table.setRowCount(0)

        parsed = parse_document(path)
        anomalies = detect_anomalies(parsed) if (
            (self.session_mode == "batch" and self.fraud_toggle.isChecked()) or
            (self.session_mode == "single" and self.check_authenticity.isChecked())
        ) else []

        self.anomaly_report.setText("\n".join(anomalies))
        render_heatmap(self.heatmap_canvas, anomalies)
        self.heatmap_canvas.draw()
        self.status_label.setText("Analyzed selected file.")

        self.parsed_table.insertRow(0)
        self.parsed_table.setItem(0, 0, QTableWidgetItem(path))
        self.parsed_table.setItem(0, 1, QTableWidgetItem(parsed.get("vendor", "")))
        self.parsed_table.setItem(0, 2, QTableWidgetItem(f"{parsed.get('amount', 0):.2f} €"))
        self.parsed_table.setItem(0, 3, QTableWidgetItem("Flagged" if anomalies else "OK"))

        if self.debug_toggle.isChecked():
            debug_text = f"FILE: {path}\nPARSED:\n{parsed}\nANOMALIES:\n{anomalies}\n\n"
            self.debug_output.setText(debug_text)

    def connect_email(self):
        self.status_label.setText("Email integration not yet implemented.")
```

## 2.4 📄 `test_generator.py`

```python
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
```

## 2.5 📄 `parser.py`

```python
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
```

## 2.6 📄 `fraud.py`

```python
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
```

## 2.7 📄 `heatmap.py`

```python
import numpy as np
import matplotlib.pyplot as plt

def render_heatmap(canvas, anomalies):
    grid = np.zeros((5, 5))

    severity_map = {
        "missing": 0.5,
        "formatting": 0.5,
        "mismatch": 1.0,
        "implausible": 1.0,
        "tampering": 1.0,
        "not found": 0.7
    }

    for anomaly in anomalies:
        a = anomaly.lower()
        intensity = next((v for k, v in severity_map.items() if k in a), 0.5)

        if "gesamtpreis" in a:
            grid[1, 0] = intensity
        elif "zahlbetrag" in a:
            grid[1, 1] = intensity
        elif "vat" in a or "ust" in a:
            grid[1, 2] = intensity
        elif "tampering" in a:
            grid[2, 0] = intensity
        elif "missing invoice" in a:
            grid[0, 0] = intensity
        elif "missing vendor" in a:
            grid[0, 1] = intensity
        elif "invalid amount" in a:
            grid[0, 2] = intensity
        elif "address" in a:
            grid[3, 0] = intensity
        else:
            grid[4, 4] = 0.3  # fallback anomaly

    ax = canvas.figure.add_subplot(111)
    ax.clear()
    ax.imshow(grid, cmap="hot", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    canvas.draw()
```

## 2.8 📄 `exporter.py`

```python
import csv
from PyQt5.QtWidgets import QFileDialog
from collections import defaultdict


def export_csv(table_widget):
    path, _ = QFileDialog.getSaveFileName(None, "Export CSV",
                                          "", "CSV Files (*.csv)")
    if not path:
        return

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        headers = [table_widget.horizontalHeaderItem(i).text() for
                   i in range(table_widget.columnCount())]
        writer.writerow(headers)

        for row in range(table_widget.rowCount()):
            row_data = [table_widget.item(row, col).text() for
                        col in range(table_widget.columnCount())]
            writer.writerow(row_data)


def generate_tax_report(table_widget):
    path, _ = QFileDialog.getSaveFileName(None, "Save Tax Report",
                                          "", "CSV Files (*.csv)")
    if not path:
        return

    vendor_totals = defaultdict(float)
    category_totals = defaultdict(float)

    for row in range(table_widget.rowCount()):
        vendor = table_widget.item(row, 1).text()
        amount = float(table_widget.item(row, 2).text())
        status = table_widget.item(row, 3).text()

        vendor_totals[vendor] += amount
        if "Flagged" not in status:
            category_totals["Valid"] += amount
        else:
            category_totals["Flagged"] += amount

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Vendor", "Total Amount"])
        for vendor, total in vendor_totals.items():
            writer.writerow([vendor, f"{total:.2f}"])

        writer.writerow([])
        writer.writerow(["Category", "Total Amount"])
        for cat, total in category_totals.items():
            writer.writerow([cat, f"{total:.2f}"])
```

## 2.9 📄 `requirements.txt`

```txt
PyQt5
fpdf
PyMuPDF
matplotlib
numpy
pytesseract
Pillow
```

---

# 3. GUI design and its user interaction flow

Let us walk through the user interaction flow for our PyQt5 GUI for Invoice and Tax document fraud detection.
I will break it down into intuitive stages:

## 3.1 `gui.py`:

### 🎛 High-Level Purpose
`gui.py` defines the **InvoiceTrackerApp**, a PyQt5 desktop application that:
- Imports and lists PDF invoices.
- Parses them (`parser.py`).
- Detects anomalies (`fraud.py`).
- Displays parsed fields and anomaly reports.
- Provides debug output and a fraud heatmap.
- Exports results to CSV and generates tax reports.
- Simulates test invoices for demos.

It’s essentially the **front-end controller** that ties together our parsing, fraud detection, visualization, and export modules.

### 🧩 Key Components

#### 1. **Class Initialization**
```python
class InvoiceTrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Invoice and Tax Tracker + Fraud Detector")
        self.setGeometry(100, 100, 1200, 700)
        self.init_ui()
        self.analysis_cache = {}  # Maps file path → (parsed, anomalies)
        self.session_mode = "idle"  # "batch" or "single"
```
- Sets up the main window.
- Initializes cache for parsed invoices and anomalies.
- Tracks whether you’re analyzing multiple files (“batch”) or one file (“single”).

#### 2. **Reset Function**
```python
def reset_app(self, note="Reset"):
    self.session_mode = "idle"
    self.analysis_cache.clear()
    self.file_list.clear()
    self.parsed_table.setRowCount(0)
    self.anomaly_report.clear()
    self.debug_output.clear()
    self.heatmap_canvas.figure.clf()
    self.heatmap_canvas.draw()
    self.status_label.setText(note)
    self.progress_bar.setValue(0)
```
- Clears all UI panels and resets state.
- Used before new imports, analysis, or test generation.

#### 3. **UI Layout**
- **Left Panel:** Import PDFs, connect email, fraud toggle, submit, test invoice generation, debug toggle.
- **Center Panel:** Tabs for:
  - **Document Tab:** Upload invoice/receipt/other files.
  - **Fields Tab:** Extraction toggles, fraud heatmap, anomaly report, debug output.
- **Right Panel:** Parsed results table, export button, tax report button, status label, progress bar.

This three-panel layout balances input, analysis, and output.

#### 4. **Signal Connections**
```python
def connect_signals(self):
    self.import_btn.clicked.connect(lambda: (self.reset_app("Imported PDFs"), self.import_pdf()))
    self.global_submit_btn.clicked.connect(self.run_pipeline)
    self.export_btn.clicked.connect(self.export_data)
    self.tax_btn.clicked.connect(self.generate_tax)
    self.analyze_submit_btn.clicked.connect(lambda: (self.reset_app("Single-doc analysis"), self.analyze_current_doc()))
    self.generate_test_btn.clicked.connect(lambda: (self.reset_app("Generated test data"), self.generate_test_data()))
    ...
```
- Connects buttons and checkboxes to their respective functions.
- Ensures UI actions trigger parsing, anomaly detection, exports, etc.

#### 5. **Pipeline Execution**
```python
def run_pipeline(self):
    self.session_mode = "batch"
    ...
    for i in range(self.file_list.count()):
        path = os.path.abspath(self.file_list.item(i).text())
        parsed = parse_document(path)
        anomalies = detect_anomalies(parsed) if self.fraud_toggle.isChecked() else []
        self.analysis_cache[path] = (parsed, anomalies)
        self.update_table(path, parsed, anomalies)
        ...
        if self.debug_toggle.isChecked():
            debug_text = f"FILE: {path}\nPARSED:\n{parsed}\nANOMALIES:\n{anomalies}\n\n"
            self.debug_output.append(debug_text)
```
- Iterates over imported PDFs.
- Calls `parse_document` (extracts vendor, amount, total price, etc.).
- Calls `detect_anomalies` if fraud mode is enabled.
- Updates table with results.
- Shows debug output if Debug Mode is toggled.

#### 6. **Single Document Analysis**
```python
def analyze_current_doc(self):
    self.session_mode = "single"
    ...
    path = (self.upload_invoice.text() or self.upload_receipt.text() or self.upload_other.text())
    parsed = parse_document(path)
    anomalies = detect_anomalies(parsed) if self.check_authenticity.isChecked() else []
    ...
    self.anomaly_report.setText("\n".join(anomalies))
    render_heatmap(self.heatmap_canvas, anomalies)
    ...
```
- Runs analysis on one selected file.
- Displays anomalies in text and heatmap.
- Updates parsed table.
- Shows debug output if enabled.

#### 7. **Test Data Generation**
```python
def generate_test_data(self):
    mode = self.test_type_selector.currentText()
    tamper_ratio = {"Mixed": 0.4, "Clean Only": 0.0, "Tampered Only": 1.0}[mode]
    month = self.month_selector.currentText()
    paths = generate_batch(n=10, tamper_ratio=tamper_ratio, month=month)
    ...
```
- Uses `test_generator.generate_batch` to create synthetic invoices.
- Lets you demo fraud detection with clean/tampered mixes.

#### 8. **Exports**
- **Export CSV:** Saves parsed table to CSV via `exporter.export_csv`.
- **Generate Tax Report:** Creates tax summary via `exporter.generate_tax_report`.

#### 9. **Debug Mode**
- Controlled by `self.debug_toggle`.
- When enabled, appends raw parsed dict and anomaly list to the debug panel.
- Gives transparency for demos and validation.

### 🧾 Summary
- **`gui.py` is the orchestration layer.**  
It ties together parsing, fraud detection, visualization, and export into a user-friendly PyQt5 interface.
- **Batch vs Single modes:** Support analyzing multiple invoices or one at a time.
- **Fraud detection toggle:** Lets you run anomaly checks only when desired.
- **Debug Mode:** Provides transparency for demos and validation.
- **Test generation:** Creates synthetic invoices for training or demo scenarios.


## 3.2 `parser.py`:

### 🎛 High-Level Purpose
`parser.py` ingests a PDF invoice, reads its text layer (and optionally OCRs the rendered page), and returns a dictionary with key fields like vendor, 
invoice ID, amount, total price, and raw text. It’s designed to be robust against tampering by combining text parsing, regex, and OCR fallback.

### 🧩 Key Functions

#### 1. `normalize_amount(value: str) -> float`
- **Purpose:** Convert euro strings into floats.
- **Process:**
  - Removes `€`.
  - Removes thousands separators (`.`).
  - Converts decimal commas to dots.
  - Strips whitespace.
  - Attempts to cast to `float`.
- **Example:** `"9,82 €"` → `9.82`.

#### 2. `extract_invoice_id(lines)`
- **Purpose:** Find the invoice number.
- **Process:**
  - Scans each line for `"Rechnungsnummer"`.
  - Uses regex to capture the following token.
  - If not inline, checks the next line.
- **Return:** Invoice ID string or empty string if not found.

#### 3. `extract_amount(lines, keyword="Zahlbetrag", window=5)`
- **Purpose:** Find the payment amount (Zahlbetrag).
- **Process:**
  - Searches for the keyword `"Zahlbetrag"`.
  - Looks up to 5 lines ahead.
  - Regex extracts euro values.
- **Return:** Float amount or `0.0` if missing.

#### 4. `extract_total_price_hybrid(page, raw_text)`
- **Purpose:** Robustly extract the Gesamtpreis (total price).
- **Hybrid strategy:**
  1. **Raw text window:** Look near `"Gesamtpreis"` in the text layer.
  2. **Word-level tokens:** Use PyMuPDF’s `get_text("words")` to catch odd tokens like `90893€`.
  3. **OCR fallback:** Render page to image, run Tesseract OCR, and scan for large euro values.
- **Return:** Float total price or `None`.

#### 5. `parse_document(path)`
- **Purpose:** Main entry point — parse a PDF file.
- **Process:**
  - Opens PDF with PyMuPDF.
  - Initializes `extracted` dict:
    ```python
    {
      "vendor": "",
      "amount": 0.0,
      "total_price": None,
      "ocr_total_price": None,
      "invoice_id": "",
      "date": "",
      "raw_text": ""
    }
    ```
  - Iterates over pages:
    - Appends text layer to `raw_text`.
    - Extracts invoice ID, amount, total price.
    - Runs OCR to populate `ocr_total_price`.
    - Collects vendor candidates via regex (`Verkauft von ...`).
  - Chooses longest vendor candidate as final vendor.
- **Return:** The `extracted` dictionary.

### 📊 Example Output

For a clean invoice:
```python
{
  "vendor": "Heinz Guenter Walsdorf",
  "amount": 9.82,
  "total_price": 9.82,
  "ocr_total_price": None,
  "invoice_id": "INV-DE-119172911-2024-7777",
  "date": "",
  "raw_text": "... full text ..."
}
```

For a tampered invoice (`Gesamtpreis 90893 €`):
```python
{
  "vendor": "Heinz Guenter Walsdorf",
  "amount": 9.82,
  "total_price": 90893.0,
  "ocr_total_price": 90893.0,
  "invoice_id": "INV-DE-119172911-2024-7777",
  "date": "",
  "raw_text": "... includes 'Gesamtpreis 90893 €' ..."
}
```

### 🎯 Executive Takeaway
- **Parser is the foundation:** It transforms messy invoice PDFs into structured fields.
- **Hybrid extraction:** Combines text layer, word tokens, and OCR to catch tampering.
- **Auditability:** Keeps `raw_text` so anomalies can be traced back to source.
- **Flexibility:** Works for both clean and manipulated invoices.

Here’s a clear **textual flow diagram** of how our `parser.py` processes an invoice PDF:

### 📄 `parser.py` Flow Diagram (Textual)

```
PDF file (input)
       │
       ▼
 ┌───────────────────────────────┐
 │ 1. Open with PyMuPDF (fitz)   │
 │    → iterate over pages        │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ 2. Extract text layer          │
 │    page.get_text("text")       │
 │    → append to raw_text        │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ 3. Regex field extraction      │
 │    - Invoice ID (Rechnungsnr.) │
 │    - Amount (Zahlbetrag)       │
 │    - Vendor ("Verkauft von")   │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ 4. Hybrid Gesamtpreis logic    │
 │    a) Scan raw text near label │
 │    b) Scan word tokens         │
 │       (catch "90893€")         │
 │    c) OCR fallback             │
 │       (render page → image →   │
 │        pytesseract → regex)    │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ 5. Normalize amounts           │
 │    "9,82 €" → 9.82             │
 │    "90893€" → 90893.0          │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ 6. Populate dictionary         │
 │ {                              │
 │   vendor,                      │
 │   amount,                      │
 │   total_price,                 │
 │   ocr_total_price,             │
 │   invoice_id,                  │
 │   date,                        │
 │   raw_text                     │
 │ }                              │
 └───────────────────────────────┘
       │
       ▼
Structured output → passed to `fraud.py` anomaly detection
```

### 🎯 Executive Narrative
- **Step 1–2:** We open the PDF and capture its text layer.  
- **Step 3:** Regex rules extract invoice ID, vendor, and Zahlbetrag.  
- **Step 4:** Gesamtpreis is extracted with a hybrid strategy: text layer, word tokens, and OCR fallback.  
- **Step 5:** All amounts are normalized into floats for comparison.  
- **Step 6:** Results are packaged into a dictionary for downstream fraud detection.  

👉 This flow shows how the parser is **modular, auditable, and tamper‑resistant** — exactly the qualities executives care about.  


## 3.3 `fraud.py`:

### 🎛 High-Level Purpose
`fraud.py` takes the structured dictionary from `parser.py` and applies a series of anomaly detection rules. It checks for missing fields, 
implausible totals, mismatches between amounts, OCR discrepancies, VAT consistency, and empty address blocks. The output is a list of anomaly messages that can be displayed in our GUI.

### 🧩 Key Functions

#### 1. `normalize(value)`
- Converts euro strings into floats for comparison.
- Same logic as in parser: strip symbols, replace separators, cast to float.

#### 2. `has_address_after(label, raw, max_gap_lines=6)`
- Looks for address‑like patterns (street + number, postal code, city, country) within a few lines after a label like “Rechnungsadresse” or “Lieferadresse”.
- Returns `True` if an address is found, `False` otherwise.

#### 3. `detect_anomalies(parsed)`
- Core function. Takes the parsed invoice dict and returns a list of anomaly strings.
- Checks include:
  - **Basic fields:** Missing invoice ID, vendor, or invalid amount.
  - **Gesamtpreis:** Missing, implausible (>1000), or unrealistic (>10000).
  - **Suspicious formatting:** Large numbers without separators (e.g., `90893€`).
  - **Mismatch:** Zahlbetrag vs Gesamtpreis differ by more than 2 EUR.
  - **OCR discrepancy:** Text layer vs OCR differ significantly, or OCR finds implausible totals.
  - **VAT consistency:** Expected VAT (19% of subtotal) vs reported VAT.
  - **Empty addresses:** Missing address content after labels.

### 📄 `fraud.py` Flow Diagram (Textual)

```
Parsed invoice dictionary (input)
       │
       ▼
 ┌───────────────────────────────┐
 │ 1. Basic field checks          │
 │   - invoice_id present?        │
 │   - vendor present?            │
 │   - amount valid?              │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ 2. Gesamtpreis plausibility    │
 │   - Missing label?             │
 │   - Implausible > 1000 EUR     │
 │   - Unrealistic > 10000 EUR    │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ 3. Formatting check            │
 │   - Regex for "90893€" style   │
 │   - Missing separators         │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ 4. Amount mismatch             │
 │   - Compare Zahlbetrag vs      │
 │     Gesamtpreis                │
 │   - Flag if difference > 2 EUR │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ 5. OCR discrepancy             │
 │   - Compare text vs OCR        │
 │   - Flag if difference > 2 EUR │
 │   - Flag OCR > 10000 EUR       │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ 6. VAT consistency             │
 │   - Extract subtotal & VAT     │
 │   - Compute expected 19%       │
 │   - Flag mismatch > 0.5 EUR    │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ 7. Address block validation    │
 │   - Check "Rechnungsadresse"   │
 │   - Check "Lieferadresse"      │
 │   - Flag if empty              │
 └───────────────────────────────┘
       │
       ▼
List of anomaly messages (output)
```

### 🎯 Executive Narrative
- **Step 1–2:** We validate basic fields and check if totals are plausible.  
- **Step 3–4:** We catch suspicious formatting and mismatches between Zahlbetrag and Gesamtpreis.  
- **Step 5:** We compare text layer vs OCR to detect hidden manipulations.  
- **Step 6:** We verify VAT consistency against expected 19%.  
- **Step 7:** We ensure addresses are present after their labels.  

👉 This flow shows how `fraud.py` enforces **auditability, plausibility, and consistency checks** — making fraud detection explainable and defensible.

Here’s the **end‑to‑end pipeline view** that combines `parser.py` and `fraud.py` into one coherent story.

### 📄 End‑to‑End Invoice Analysis Pipeline

```
PDF Invoice (input)
       │
       ▼
 ┌───────────────────────────────┐
 │ Parser (parser.py)            │
 │                               │
 │ 1. Open PDF with PyMuPDF       │
 │ 2. Extract text layer          │
 │ 3. Regex field extraction      │
 │    - Invoice ID                │
 │    - Vendor                    │
 │    - Zahlbetrag (amount)       │
 │ 4. Hybrid Gesamtpreis logic    │
 │    a) Text near "Gesamtpreis"  │
 │    b) Word tokens (catch 90893€)│
 │    c) OCR fallback             │
 │ 5. Normalize amounts           │
 │ 6. Build structured dictionary │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ Fraud Detector (fraud.py)     │
 │                               │
 │ 1. Basic field checks          │
 │    - Invoice ID, vendor, amount│
 │ 2. Gesamtpreis plausibility    │
 │    - Missing, implausible,     │
 │      unrealistic thresholds    │
 │ 3. Formatting check            │
 │    - Suspicious "90893€" style │
 │ 4. Amount mismatch             │
 │    - Zahlbetrag vs Gesamtpreis │
 │ 5. OCR discrepancy             │
 │    - Text vs OCR comparison    │
 │ 6. VAT consistency             │
 │    - Expected 19% vs reported  │
 │ 7. Address block validation    │
 │    - Billing & shipping present│
 └───────────────────────────────┘
       │
       ▼
Anomaly Report (output)
- List of flagged issues
- Debug Mode shows raw parsed data
- Heatmap visualizes anomalies
```

### 🎯 Executive Narrative
- **Step 1:** The parser transforms messy PDFs into structured fields using text, regex, and OCR.  
- **Step 2:** The fraud detector applies plausibility, consistency, and discrepancy rules.  
- **Step 3:** Results are surfaced in the GUI as anomaly reports, debug output, and heatmaps.  
- **Business Impact:** This pipeline ensures invoices are **transparent, auditable, and fraud‑resistant**, supporting compliance and reducing risk.


## 3.4 📄 `exporter.py`

### 🎛 Purpose
This module enables users to **export parsed invoice data** and **generate tax summaries** directly from the GUI’s table widget. It bridges the analysis results with external reporting formats (CSV).

### 🧩 Functions

#### 1. `export_csv(table_widget)`
- **Workflow:**
  1. Opens a save dialog (`QFileDialog.getSaveFileName`) to let the user choose where to save.
  2. Collects table headers from the GUI’s `QTableWidget`.
  3. Iterates through each row, extracting cell text.
  4. Writes headers and rows into a CSV file.
- **Outcome:** A clean CSV file with all parsed invoices, vendors, amounts, and statuses.

👉 *Executive narrative:* “This function lets us export the full invoice table for compliance or further analysis in Excel.”

#### 2. `generate_tax_report(table_widget)`
- **Workflow:**
  1. Opens a save dialog for the tax report CSV.
  2. Initializes two dictionaries:
     - `vendor_totals`: sums amounts per vendor.
     - `category_totals`: sums amounts by category (“Valid” vs “Flagged”).
  3. Iterates through table rows:
     - Adds each amount to the vendor’s total.
     - Categorizes amounts based on anomaly status.
  4. Writes two sections into the CSV:
     - Vendor totals.
     - Category totals.
- **Outcome:** A tax report summarizing vendor totals and separating flagged vs valid invoices.

👉 *Executive narrative:* “This function produces a tax summary that distinguishes clean invoices from flagged ones, supporting compliance and audit readiness.”

## 3.5 📄 `heatmap.py`

### 🎛 Purpose
This module provides a **visual fraud heatmap**. It maps anomalies to a grid and displays them with intensity colors, making fraud detection more intuitive.

### 🧩 Function

#### `render_heatmap(canvas, anomalies)`
- **Workflow:**
  1. Initializes a 5×5 grid of zeros.
  2. Defines a `severity_map`:
     - Missing → 0.5
     - Formatting → 0.5
     - Mismatch → 1.0
     - Implausible → 1.0
     - Tampering → 1.0
     - Not found → 0.7
  3. Iterates through anomalies:
     - Converts anomaly text to lowercase.
     - Matches keywords against severity map.
     - Places intensity values into specific grid cells:
       - Gesamtpreis anomalies → `[1,0]`
       - Zahlbetrag anomalies → `[1,1]`
       - VAT anomalies → `[1,2]`
       - Tampering → `[2,0]`
       - Missing invoice/vendor/amount → `[0,*]`
       - Address anomalies → `[3,0]`
       - Fallback anomalies → `[4,4]`
  4. Renders the grid with `matplotlib` using a “hot” colormap.
  5. Draws the heatmap on the provided canvas.

- **Outcome:** A color‑coded anomaly visualization where red/yellow intensity highlights suspicious fields.

👉 *Executive narrative:* “The heatmap translates anomaly text into a visual grid, so users can instantly see which fields are suspicious and how severe the anomalies are.”

## 🎯 Combined Takeaway
- **`exporter.py`:** Converts analysis results into CSVs for compliance and tax reporting.  
- **`heatmap.py`:** Provides a visual fraud map, making anomalies intuitive and executive‑friendly.  

Together, they ensure our system isn’t just detecting fraud — it’s **communicating results clearly** in both structured reports and visual dashboards.

## 3.6 📄 `test_generator.py`

### 🎛 Purpose
This module programmatically generates **synthetic invoice PDFs** for testing and demos. It allows you to simulate clean vs tampered invoices with randomized vendors, categories, and amounts.

### 🧩 Key Elements
- **Constants:**
  - `VENDORS`: sample vendor names (AcmeCorp, Globex, etc.).
  - `CATEGORIES`: sample categories (Office, Travel, Meals, etc.).

- **Function: `generate_test_pdf(...)`**
  - Uses `fpdf` to create a simple invoice PDF.
  - Writes vendor, amount, category, and month.
  - If `tampered=True`, adds extra lines like “Note: This document has been altered” and “Font mismatch detected.”
  - Saves PDF into a `test_docs` folder.
  - Returns the file path.

- **Function: `generate_batch(...)`**
  - Creates a batch of `n` invoices.
  - Randomly selects vendor, category, and amount.
  - Randomly decides if each invoice is tampered based on `tamper_ratio`.
  - Names files accordingly (`test_invoice_January_1_clean.pdf` or `...tampered.pdf`).
  - Returns a list of file paths.

### 🎯 Executive Narrative
*“This module lets us generate synthetic invoices on demand, mixing clean and tampered cases. It’s perfect for demos, training, and stress‑testing the fraud detection pipeline.”*

## 3.7 📄 `main.py`

### 🎛 Purpose
This is the **entry point** of your application. It launches the PyQt5 GUI defined in `gui.py`.

### 🧩 Workflow
- Imports `QApplication` from PyQt5 and your `InvoiceTrackerApp`.
- Creates the application object (`app = QApplication(sys.argv)`).
- Instantiates the main window (`window = InvoiceTrackerApp()`).
- Shows the window (`window.show()`).
- Starts the Qt event loop (`sys.exit(app.exec_())`).

### 🎯 Executive Narrative
*“This file is the launcher. It initializes the Qt application and opens the Invoice Tracker GUI. Everything else — parsing, fraud detection, exports — is orchestrated from here.”*

## 3.8 📄 `requirements.txt`

### 🎛 Purpose
Defines the Python dependencies needed to run your system. This ensures reproducibility and easy setup.

### 🧩 Dependencies
- **PyQt5** → GUI framework.
- **fpdf** → Generate synthetic test invoices.
- **PyMuPDF** → Parse PDF text and structure.
- **matplotlib** → Render fraud heatmaps.
- **numpy** → Grid math for heatmaps.
- **pytesseract** → OCR fallback for tampered invoices.
- **Pillow** → Image handling for OCR.

### 🎯 Executive Narrative
*“This file lists all required packages. It guarantees that anyone can install dependencies and run the system consistently.”*

## 🔗 Combined Role
- **`test_generator.py`** → Creates demo data.  
- **`main.py`** → Launches the GUI.  
- **`requirements.txt`** → Ensures environment reproducibility.  

Together, they make our system **demo‑ready, portable, and reproducible**.

In the following we conclude with the **full architectural overview** of our system — a textual diagram that ties together all the 
modules (`parser.py`, `fraud.py`, `gui.py`, `exporter.py`, `heatmap.py`, `test_generator.py`, `main.py`, plus `requirements.txt`).

## 3.9 🏗 End‑to‑End System Architecture

```
User launches app (main.py)
       │
       ▼
 ┌───────────────────────────────┐
 │ GUI Layer (gui.py)             │
 │ - PyQt5 interface               │
 │ - Import PDFs / connect email   │
 │ - Fraud toggle / Debug mode     │
 │ - Parsed table + anomaly report │
 │ - Heatmap visualization         │
 │ - Export & tax report buttons   │
 │ - Test invoice generation       │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ Parser (parser.py)             │
 │ - Extract text layer (PyMuPDF) │
 │ - Regex for invoice ID, vendor │
 │ - Zahlbetrag extraction        │
 │ - Hybrid Gesamtpreis logic      │
 │   (text, word tokens, OCR)     │
 │ - Normalize amounts            │
 │ - Build structured dictionary  │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ Fraud Detector (fraud.py)      │
 │ - Basic field checks            │
 │ - Gesamtpreis plausibility      │
 │ - Suspicious formatting         │
 │ - Zahlbetrag vs Gesamtpreis     │
 │ - OCR discrepancy               │
 │ - VAT consistency               │
 │ - Address block validation      │
 │ - Output: anomaly list          │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ Output Layer                   │
 │                                │
 │ Exporter (exporter.py)         │
 │ - Export parsed table to CSV   │
 │ - Generate vendor/category tax │
 │   report (Valid vs Flagged)    │
 │                                │
 │ Heatmap (heatmap.py)           │
 │ - Map anomalies to grid cells  │
 │ - Render severity with colors  │
 │ - Visual fraud dashboard       │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ Test Data Generator            │
 │ (test_generator.py)            │
 │ - Create synthetic invoices    │
 │ - Random vendors, categories   │
 │ - Tampered vs clean ratio      │
 │ - Demo/test dataset            │
 └───────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────┐
 │ Environment (requirements.txt) │
 │ - PyQt5, fpdf, PyMuPDF,        │
 │   matplotlib, numpy,           │
 │   pytesseract, Pillow          │
 │ - Ensures reproducibility      │
 └───────────────────────────────┘
```

## 🎯 Executive Narrative

- **GUI Layer:** The user interacts through a PyQt5 interface — importing invoices, toggling fraud detection, viewing anomalies, and exporting results.  
- **Parser:** Converts messy PDFs into structured fields using text extraction, regex, and OCR fallback.  
- **Fraud Detector:** Applies plausibility, consistency, and discrepancy rules to flag anomalies.  
- **Output Layer:** Results are communicated clearly — structured CSVs for compliance, tax summaries for reporting, and heatmaps for intuitive visualization.  
- **Test Generator:** Provides synthetic clean/tampered invoices for demos and validation.  
- **Environment:** Requirements ensure reproducibility across machines.  
- **Main Launcher:** Starts the application and ties everything together.

This architecture shows a **modular, auditable, and demo‑ready system**: from ingestion to fraud detection to reporting and visualization.

---

# 🧠 4. Run instructions

Here’s a simple and effective **Jupyter Notebook runner** that launches our PyQt5 GUI from within a notebook environment. 
This is especially useful for testing, demos, or integrating GUI workflows into broader data science pipelines.

## 📄 `run_gui.ipynb` – Jupyter Notebook Launcher

```python
# Jupyter cell 1: Setup
import sys
import os

# Ensure the GUI directory is in the path
sys.path.append(os.path.abspath("invoice_tracker_gui"))

# Optional: autoreload if you're editing modules
%load_ext autoreload
%autoreload 2
```

```python
# Jupyter cell 2: Launch the GUI
from PyQt5.QtWidgets import QApplication
from gui import InvoiceTrackerApp

app = QApplication(sys.argv)
window = InvoiceTrackerApp()
window.show()
app.exec_()
```

## 🧠 Notes

- This assumes our project folder is named `invoice_tracker_gui` and is in the same directory as the notebook.
- We must run this notebook in a **desktop-capable environment** (e.g., local Jupyter, VS Code, or JupyterLab with GUI support).
- If we are using **JupyterLab**, we have to make sure it’s not running in headless mode.


---
 

# 🔥 5. Heatmap Coloring Scheme Explained

The heatmap is a **visual diagnostic tool** that highlights potential anomalies in invoice data. Here's how it works:

### 🎨 Color Mapping
- **Red / Bright Yellow**: Indicates regions with **high anomaly intensity** — these are areas where fraud signals (like tampering, missing fields, or suspicious keywords) are concentrated.
- **Orange / Mid-Tones**: Moderate anomaly likelihood — possibly inconsistent data or borderline suspicious entries.
- **Dark Shades (Black / Deep Red)**: Low or no anomaly presence — these regions are considered clean or normal.

### 🧠 Data Simulation
The heatmap uses a synthetic 10×10 matrix (`np.random.rand(10, 10)`) to simulate anomaly distribution. When anomalies are detected:
```python
data[0:3, 0:3] += 2  # artificially intensifies top-left region
```
This creates a **clustered hotspot** to visually represent fraud detection activity.

### 🧪 What It Represents
- The heatmap doesn’t map to specific fields (like vendor or amount) — it’s a **symbolic visualization** of anomaly density.
- It’s meant to give users a **quick visual cue**: if the heatmap lights up, the document likely contains fraud indicators.  

Also, understanding the distinction between **invoice**, **receipt**, and **other document** is key to organizing and evaluating 
financial records accurately. Here’s how they differ in purpose and structure:

## 🧾 Invoice vs. Receipt vs. Other Document

| Type            | Purpose                                      | Issued By             | Key Features                              |
|-----------------|----------------------------------------------|------------------------|--------------------------------------------|
| **Invoice**     | Request for payment                          | Seller or service provider | Includes due date, itemized charges, tax, total amount |
| **Receipt**     | Proof of payment already made                | Seller or payment processor | Shows payment confirmation, method, date, and amount |
| **Other Document** | Supplementary or unrelated financial info | Varies (e.g., bank, employer) | Could be contracts, delivery notes, tax forms, etc. |


### 🔍 Invoice
- Sent **before** payment
- Used for billing and accounting
- May include payment terms (e.g., Net 30)
- Often triggers fraud detection (e.g., inflated charges)

### ✅ Receipt
- Issued **after** payment is received
- Confirms transaction completion
- Less likely to be fraudulent, but still worth verifying (e.g., duplicate charges)

### 📂 Other Document
- Catch-all for anything that’s not a formal invoice or receipt
- Examples: delivery slips, tax letters, bank statements, warranty forms
- May contain useful metadata or context for fraud analysis

## 🧠 In Our GUI
When we upload:
- **Invoice** → triggers full parsing + fraud evaluation
- **Receipt** → confirms payment and cross-checks invoice
- **Other Doc** → stored for reference, may be parsed if relevant

---

# 6. 🔗 Results and conclusions

## 6.1 📊 Start the Tax-Invoice-GUI

### ✅ Step 1: Download the folder

Download the main folder
📁 [Invoice_TaxTracker_GUI](https://github.com/NenadBalaneskovic/ExternalProjects/tree/9581a59cec81e7484abe874f71a83a900929749c/Invoice_TaxTracker_GUI)
 which has the following structure:  
 
   <img src="https://github.com/NenadBalaneskovic/ExternalProjects/blob/9581a59cec81e7484abe874f71a83a900929749c/Invoice_TaxTracker_GUI/Invoice_GUI_folder.PNG" width="400" height="200"/>

### ✅ Step 2: Run the jupyter runner

Run the jupyter file "__run_gui.ipynb__" (subfolder "__invoice_tracker_gui__") in VS Code or Jupyter notebook.

### ✅ Step 3: Interact with the Tax-Invoice-GUI

Interact with the Tax-Invoice-GUI by providing reasonable inputs in the left half-plane and pressing the "__Submit__"-button.

## 6.2 🧠 Interpretation of results

### 🧠 This Tax-Invoice-GUI...

- Accepts user inputs:
  - invoices and tax forms as PDFs
  - extraction field specifications
  
- Displays fraud detection results emerging from the successive application of different parsing and OCR methods 
and renders them as 2D-plots (heatmaps) within the GUI-plane

- Stores obtained results and characterizations as csv files (see figure below)

![Fraud_Detection_Results_csv](https://github.com/NenadBalaneskovic/ExternalProjects/blob/4a436eaba0a9111c87bf98caff3cf5c21ac61d69/Invoice_TaxTracker_GUI/tax_report_stored.PNG)

- Automatically displays the fraud detection heatmap associated with selected (extracted) invoice fields (see figure below).

![Tax_Invoice_GUI_Functionality](https://github.com/NenadBalaneskovic/ExternalProjects/blob/e6263c8a1be70e3d0f294ab777a8d7c449a2a87e/Invoice_TaxTracker_GUI/GUI_complete_results.png)

## 6.3 🏁 Final Thoughts

The Invoice and Tax Tracker + Fraud Detector project demonstrates how modular, auditable design can transform the way organizations handle financial documentation. 
By integrating parsing, anomaly detection, visualization, and reporting into a single cohesive framework, the system provides a transparent and defensible approach to invoice management. 
Each component — from the hybrid parser that combines text extraction and OCR, to the fraud detector that applies plausibility and consistency checks, to the GUI that surfaces results in 
both textual and visual formats — contributes to a pipeline that is robust, explainable, and business‑ready.

The project’s strength lies in its balance of technical rigor and practical usability. The parser ensures that even tampered invoices are captured accurately, while the fraud module translates 
complex validation rules into clear anomaly reports. The GUI empowers users to interact with the system intuitively, toggling fraud detection, running batch or single analyses, and exporting 
results for compliance. The heatmap visualization adds an executive‑friendly layer of insight, making anomalies immediately visible and understandable. Meanwhile, the exporter and tax report generator 
bridge the gap between detection and compliance, ensuring that flagged anomalies can be incorporated into structured reporting workflows. The test generator further enhances the system’s utility by 
providing synthetic datasets for demos, training, and validation.

Collectively, these modules embody the project’s overarching aim: to deliver a consulting‑ready solution that foregrounds transparency, auditability, and reproducibility. In environments where 
regulatory scrutiny and fraud risk are ever‑present, this system offers a defensible framework that not only detects anomalies but also explains them in a way that builds trust with auditors, executives,
 and clients. By embedding explainability and validation into every stage, the project showcases how advanced analytics can be deployed responsibly and effectively.

Beyond its immediate functionality, the project also serves as a proof of concept for how physics‑inspired rigor and AI/ML best practices can be integrated into client‑facing tools. It highlights the importance 
of modular design, where each component can be refined independently yet contributes to a coherent whole. It demonstrates how transparency features like Debug Mode and anomaly heatmaps can elevate technical outputs 
into governance‑ready artifacts. Most importantly, it illustrates how technical depth can be translated into business clarity, ensuring that complex detection logic is narratable and actionable at the executive level.

In conclusion, the Invoice and Tax Tracker + Fraud Detector is more than a technical prototype; it is a strategic framework for bridging compliance, fraud prevention, and executive communication. It underscores the 
value of explainable AI in consulting contexts, where trust and defensibility are as critical as detection accuracy. By combining robust parsing, anomaly detection, visualization, and reporting, the project delivers a 
comprehensive solution that is not only technically sound but also aligned with the broader goals of governance, transparency, and business impact.

---

# 7. 🧾 Invoice and Tax Tracker + Fraud Detector  
### 🛠️ User Manual

Here is a comprehensive, step-by-step **User Manual** for our **Invoice and Tax Tracker + Fraud Detector GUI**. It covers every feature, input/output behavior, and practical examples 
to help users navigate and test the system confidently.

## 🧠 Pre-Step 1: Install Tesseract OCR Engine

### 🔧 Using the official installer
1. Go to the official Tesseract GitHub page: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
2. Scroll down to the **Windows** section and download the installer from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki).
3. Download the `.exe` installer (e.g., `tesseract-ocr-w64-setup-v5.3.3.20231005.exe`).
4. Run the installer and follow the setup instructions.
5. During installation, **note the installation path** (e.g., `C:\Program Files\Tesseract-OCR`).

## 🧠 Pre-Step 2: Add Tesseract to System PATH

1. Open **Start Menu** → search for **Environment Variables** → click **Edit the system environment variables**.
2. In the **System Properties** window, click **Environment Variables**.
3. Under **System variables**, find and select `Path`, then click **Edit**.
4. Click **New**, and add the path to the Tesseract installation folder (e.g., `C:\Program Files\Tesseract-OCR`).
5. Click **OK** to save and close all dialogs.

## 🧠 Pre-Step 3: Install pytesseract via pip

Open your command prompt and run:

```bash
pip install pytesseract
```

## 🧠 Pre-Step 4: Verify Installation

Create a quick Python script to test:

```python
import pytesseract
from PIL import Image

# Optional: specify path if not in system PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = Image.open('sample_image.png')  # Replace with your image path
text = pytesseract.image_to_string(img)
print(text)
```

## ✅ You're all set!

You can now use Tesseract OCR in your Python projects.

## 📌 Overview

This application allows users to:
- Import and analyze invoices, receipts, and other financial documents
- Detect potential fraud using rule-based heuristics
- Visualize anomalies via a heatmap
- Generate test data for simulation
- Export results for tax and audit purposes

## 🖥️ GUI Layout

| Section | Description |
|--------|-------------|
| **Left Panel** | File import, email connection, fraud toggle, test data generation |
| **Center Panel** | Document upload tabs and fraud analysis options |
| **Right Panel** | Parsed results table, export buttons, status and progress bar |

## 🧭 Step-by-Step Guide

### 1. 📂 Importing Documents

#### Option A: Manual Import
- Click **“Import PDF”**
- Select one or more `.pdf` files
- Files will appear in the **file list**

#### Option B: Upload via Tabs
- Go to **“Document”** tab
- Use **“Browse”** buttons next to:
  - Upload Invoice
  - Upload Receipt
  - Upload Other Doc

> ✅ These fields are used for single-document analysis in the **“Fields”** tab.

### 2. 📧 Email Integration (Placeholder)
- Click **“Connect Email”**
- Currently displays: *“Email integration not yet implemented.”*

### 3. 🧪 Generating Test Invoices
- Select a **test type**:
  - Mixed
  - Clean Only
  - Tampered Only
- Choose a **month** from the dropdown
- Click **“Generate Test Invoices”**
- 10 synthetic PDFs will be created and added to the file list

> Example: `test_invoice_January_3_tampered.pdf`

### 4. 🕵️ Enabling Fraud Detection
- Toggle **“Enable Fraud Detection Mode”**
- When enabled, the system will:
  - Check for missing fields
  - Detect suspicious keywords (e.g., “manipuliert”)
  - Flag font mismatches or altered vendor names

### 5. 🚀 Running the Analysis
- Click **“Submit”** (left panel)
- Each file is parsed and evaluated
- Results appear in the **Parsed Table**:
  - File path
  - Vendor name
  - Amount
  - Status: `OK` or `Flagged`

> Progress bar shows completion percentage  
> Status label updates to “Parsing complete.”

### 6. 🔍 Analyzing a Single Document
- Go to **“Fields”** tab
- Select a file via the upload fields
- Check desired options:
  - Extract Vendor
  - Extract Amount
  - Extract Category
  - Check Authenticity
- Click **“Submit”**
- Output:
  - **Anomaly Report** (textual)
  - **Visual Heatmap** (fraud intensity)

### 7. 📤 Exporting Results

#### Export CSV
- Click **“Export CSV”**
- Saves the parsed table to a `.csv` file

#### Generate Tax
- Click **“Generate Tax”**
- Creates a summary of:
  - Total amount per vendor
  - Flagged vs. valid totals
- Saves to a `.csv` file

## 📈 Heatmap Interpretation

- **Bright Red/Yellow**: High anomaly density
- **Orange**: Moderate suspicion
- **Dark**: Clean zones
- Symbolic visualization — not tied to document coordinates

## 📁 Input Examples

### ✅ Clean Invoice
```
Rechnungsnummer: INV-DE-2025-0001
Verkauft von: Heinz Guenter Walsdorf
Zahlbetrag: 9,82 EUR
```

### ⚠️ Tampered Invoice
```
Verkauft von: Heinz Günter W@lsdorf
Hinweis: Dieses Dokument wurde manipuliert.
```

## 📤 Output Examples

| File                          | Vendor                     | Amount | Status   |
|-------------------------------|----------------------------|--------|----------|
| test_invoice_January_1_clean.pdf | Heinz Guenter Walsdorf     | 9.82   | OK       |
| test_invoice_January_3_tampered.pdf | Heinz Günter W@lsdorf       | 9.82   | Flagged  |

## 🧠 Tips & Notes

- Use **“Generate Test Invoices”** to validate fraud detection logic
- Use **“Fields” tab** for deep dive into a single document
- Replace `€` with `EUR` in test PDFs to avoid encoding issues
- OCR fallback is triggered if text extraction fails  

Finally, we offer a crisp **demo flow** of our GUI. It highlights the value chain from ingestion to anomaly detection and reporting:

## 🚀 Demo Flow for `InvoiceTrackerApp

👉 *“We start by importing invoices directly into the system. Each file is queued for analysis.”*`

### 1. **Import PDFs**
- Click **Import PDF** on the left panel.  
- Select a batch of invoices (clean + tampered).  
- They appear in the **File List**.

👉 *“The pipeline extracts vendor, amount, and total price, then applies fraud rules. Any anomalies are flagged immediately.”*

### 2. **Run Pipeline**
- Click **Submit**.  
- The app parses each invoice (`parser.py`) and runs anomaly checks (`fraud.py`).  
- Progress bar shows completion percentage.  
- Results populate the **Parsed Table** (right panel):
  - File path
  - Vendor
  - Amount
  - Status (“OK” or “Flagged”)


👉 *“Fraud Detection Mode highlights implausible totals, mismatched VAT, or discrepancies between text and OCR.”*

### 3. **Fraud Detection Mode**
- Toggle **Enable Fraud Detection Mode**.  
- Anomalies are listed in the **Anomaly Report** (center panel).  
- Fraud heatmap visualizes suspicious fields.


👉 *“Debug Mode provides transparency: we see exactly what was parsed and why anomalies were flagged. This is critical for auditability.”*


### 4. **Debug Mode**
- Toggle **Enable Debug Mode**.  
- Raw parsed dictionary and anomaly list appear in the **Debug Output** panel.


👉 *“We can also analyze a single invoice in detail, useful for case-by-case validation.”*

### 5. **Single Document Analysis**
- Upload one invoice via the **Document Tab**.  
- Click **Submit** in the Fields tab.  
- Anomaly report and heatmap update for that file only.


👉 *“For demos or training, we can generate synthetic test invoices with controlled tampering ratios.”*

### 6. **Test Data Generation**
- Choose **Mixed / Clean Only / Tampered Only** in the test selector.  
- Pick a month.  
- Click **Generate Test Invoices**.  
- Synthetic invoices appear in the file list.


👉 *“Finally, results can be exported for compliance or rolled up into tax reporting.”*

### 7. **Export & Tax Report**
- Click **Export CSV** → saves parsed table.  
- Click **Generate Tax** → produces a tax summary report.


## 🎯 Executive Takeaway
- **Transparency:** Debug Mode shows raw parsing and anomalies.  
- **Auditability:** Fraud rules are explicit and explainable.  
- **Flexibility:** Batch vs single analysis, synthetic test generation.  
- **Business Impact:** Immediate anomaly detection reduces fraud risk and supports tax compliance.


---

# 8. 📚 References
1. (Py-)tesseract package: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract), https://pypi.org/project/pytesseract/,
https://builtin.com/data-science/python-ocr, https://www.analyticsvidhya.com/blog/2024/04/ocr-libraries-in-python/ and [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki).
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/555418f5374782028cd14a3650caec82b4134ee5/Invoice_TaxTracker_GUI/Invoice_TaxTracker_GUI.ipynb)
3. [![LinearProgrammingOptimization Report | English](https://img.shields.io/badge/LinearProgrammingOptimization%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/a17e5fc8e0f19b55f16b61cb2e2070c67e49c01c/LinearProgramming_GUI/LinearProgramming_Methodology_Report.pdf) 
4. A. Meister , T. Sonar: "__Numerik__", 1st Ed. Springer-Spektrum (2019); S. Chapra, R. Canale: "__Numerical Methods for Engineers__", Mcgraw-Hill, 6th Edition (2010). 
5. J. Kilty, A. M. McAllister: "__Mathematical Modeling and Applied Calculus__", 1st Ed. Oxford University Press (2018).
6. U. Kockelkorn: "__Statistik für Anwender__", 1st Ed. Springer (2012), s. chapters 7 - 8.
7. Robert H. Shumway, David S. Stoffer: "__Time Series Analysis and Its Applications with R Examples__", Springer (2011).
8. Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor: "__An Introduction to Statistical Learning with Applications in Python__", Springer (2023).
9. Cornelis W. Oosterlee, Lech A. Grzelak: "__Mathematical Modeling and Computation in Finance with Exercises and Python and MATLAB Computer Codes__", World Scientific (2020).
10. Richard Szeliski: "__Computer Vision - Algorithms and Applications__", Springer (2022).
11. Anthony Scopatz, Kathryn D. Huff: "__Effective Computation in Physics - Field Guide to Research with Python__", O'Reilly Media (2015).
12. Alex Gezerlis: "__Numerical Methods in Physics with Python__", Cambridge University Press (2020).
13. Gary Hutson, Matt Jackson: "__Graph Data Modeling in Python. A practical guide__", Packt-Publishing (2023).
14. Hagen Kleinert: "__Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets__", 5th Edition, World Scientific Publishing Company (2009).
15. Peter Richmond, Jurgen Mimkes, Stefan Hutzler: "__Econophysics and Physical Economics__", Oxford University Press (2013).
16. A. Coryn , L. Bailer Jones: "__Practical Bayesian Inference A Primer for Physical Scientists__", Cambridge University Press (2017).
17. Avram Sidi: "__Practical Extrapolation Methods - Theory and Applications__", Cambridge university Press (2003).
18. Volker Ziemann: "__Physics and Finance__", Springer (2021).
19. Zhi-Hua Zhou: "__Ensemble methods, foundations and algorithms__", CRC Press (2012).
20. B. S. Everitt, et al.: "__Cluster analysis__", Wiley (2011).
21. Lior Rokach, Oded Maimon: "__Data Mining With Decision Trees - Theory and Applications__", World Scientific (2015).
22. Bernhard Schölkopf, Alexander J. Smola: "__Learning with kernels - support vector machines, regularization, optimization and beyond__", MIT Press (2009).
23. Johan A. K. Suykens: "__Regularization, Optimization, Kernels, and Support Vector Machines__", CRC Press (2014).
24. Sarah Depaoli: "__Bayesian Structural Equation Modeling__", Guilford Press (2021).
25. Rex B. Kline: "__Principles and Practice of Structural Equation Modeling__", Guilford Press (2023).
26. Ekaterina Kochmar: "__Getting Started with Natural Language Processing__", Manning (2022).
27. Jakub Langr, Vladimir Bok: "__GANs in Action__", Computer Vision Lead at Founders Factory (2019).
28. David Foster: "__Generative Deep Learning__", O'Reilly(2023).
29. Rowel Atienza: "__Advanced Deep Learning with Keras: Applying GANs and other new deep learning algorithms to the real world__", Packt Publishing (2018).
30. Josh Kalin: "__Generative Adversarial Networks Cookbook__", Packt Publishing (2018).  
31. Thomas Haslwanter: "__Hands-on Signal Analysis with Python: An Introduction__", Springer (2021).
32. Jose Unpingco: "__Python for Signal Processing__", Springer (2023).
33. R. K. Burdick, C. M. Borror, D. C. Montgomery: "__Design and Analysis of Gauge R&R Studies__", 1st Ed. SIAM (2005); 
S. H. Derakhshan , C. V. Deutsch: "__Numerical Integration of Bivariate Gaussian Distribution__", Paper 405, CCG Anual Report 13 (2011).
34. C. Paar, J. Pelzl: "__Understanding Cryptography__", Springer (2010); H. Delfs, H. Knebl: "__Introduction to Cryptography__", 3rd Ed. Springer (2015); J. Katz, Y. lindell: "__Introduction to Modern Cryptography__", 2nd Ed, CRC Press (2015); 
O. Goldreich: "__Foundations of Cryptography__", Cambridge University Press (2008); J. P. Aumasson: "__Serious Cryptography__", no starch press (2018).  
35. J. Berk, P. DeMarzo: „__Corporate Finance__“, 6th Ed., Pearson (2023); R. W. Melicher, E. A. Norton: "__Introduction to Finance__", 16th Ed. WILEY (2017); 
Anatoly B. Schmidt: "__Quantitative Finance for Physicists: An Introduction__", 1st Ed. Academic Press (2005); Alex Backwell: "__An Intuitive Introduction to Finance and Derivatives: Concepts, Terminology and Models__",
 1st Ed, Springer (2023); Michael Isichenko: "__Quantitative Portfolio Management: The Art and Science of Statistical Arbitrage__", 1st Ed., Springer (2021); John H. Cochrane: "__Asset Pricing__", Revised Ed., Princeton University Press (2005);
 Antti Ilmanen: "__Expected Returns: An Investor’s Guide to Harvesting Market Rewards__", 1st Ed., WILEY (2011); Steven E. Shreve: "__Stochastic Calculus for Finance I & II__", 1st Ed., Springer (2004); 
 Andrew Pole: "__Statistical Arbitrage: Algorithmic Trading Insights and Techniques__", 1st Ed., WILEY (2007); Mark S. Joshi: "__The Concepts and Practice of Mathematical Finance__", 2nd Ed., Cambridge University Press (2008);
Kaggle-link: competition-documentation: https://www.kaggle.com/competitions/drw-crypto-market-prediction.
36. R. Nystrom: "__Game Programming Patterns__", 1st Ed. genever benning (2014); A. A. Stepanov, D. E. Rose: "__From Mathematics to Generic Programming__", 1st Ed. Addison-Wesley (2015);
37. E. Parzen: "__Stochastic Processes__", 3rd Ed. Dover Publications (2015); S. Aloorravi: "__Metaprogramming with Python__", 1st Ed. Packt (2022); B. Klein, P. Klein: "__Funktionale Programmierung mit Python__", Hanser (2025);
K. Webel, D. Wied: "__Stochastische Prozesse__", 2. Auflage Springer (2016); L. Held: "__Methoden der statistischen Inferenz__", 1. Auflage Spektrum (2008); E. Cinlar: "__Stochastic Processes__", Dover (2013);
N. Bäuerle, U. Rieder: "__Finanzmathematik in diskreter Zeit__", Springer-Spektrum (2017); M. Albrecht, R. Maurer: "__Investment- und Risikomanagement__", 3. Auflage, Schäffer Poeschel (2008);
N. H. Bingham, R. Kiesel: "__Risk Neutral Valuation: Pricing and Hedging of Financial Derivatives__", 2. Auflage Springer (2004); T. Björk: "__Arbitrage Theory in Continuous Time__", 3rd Ed. Oxford University Press (2009);
N. J. Cutland, A. Roux: "__Derivative Pricing in Discrete Time__", Springer (2013); F. Delbaen, W. Schachermayer: "__The Mathematics of Arbitrage__", Springer (2006); 
R. J. Elliott, P. E. Kopp: "__Mathematics of Financial Markets__", 2nd Ed. Springer (2005); H. Föllmer, A. Scheid: "__A Stochastic Finance: An Introduction in Discrete Time__", 3rd Ed. de Gruyter (2011);
J. C. Hull: "__Options, Futures and Other Derivatives__", 8th Ed. Pearson (2011); J. Kremer: "__Einführung in die diskrete Finanzmathematik__", Springer (2005); 
D. Lamberton, B. Lapeyre: "__Introduction to Stochastic Calculus Applied to Finance__", Chapman & Hall (2007); D. G. Luenberger: "__Investment Science__", Oxford University Press (1998);
S. R. Pliska: "__Introduction to Mathematical Finance: Discrete Time Models__", Blackwell (2000); A. N. Shiryaev: "__Essentials of Stochastic Finance__", World Scientific (2001);
S. E. Shreve: "__Stochastic Calculus for Finance I: The Binomial Asset Pricing Model__", Springer (2004); J. Kremer: "__Portfoliotheorie, Risikomanagement und die Bewertung von Derivaten__", Springer (2011);
L. Rüschendorf: "__Mathematical Risk Analysis__", Springer (2013). 
38. A. Becker: "__Kalman Filter - From the Ground Up__", 1st Ed. private publication (2023); K. Triantafyllopoulos: "__Bayesian Inference of State Space Models__", 1st Ed. Springer (2021); 
P. Zarchan, H. Musoff: "__Fundamentals of Kalman Filtering: A Practical Approach__", 
3rd Ed. AIAA (2009); A. Sidi: "__Vector Extrapolation Methods with Applications__", 1st Ed. SIAM (2019); C. Brezinski, M. R. Zaglia: "__Extrapolation Methods - Theory and Practice__", 2nd Ed. North-Holland (2002); 
C. Gardiner, P. Zoller: "__Quantum Noise: A Handbook of Markovian and Non-Markovian Quantum Stochastic Methods with Applications to Quantum Optics__", 3rd Ed. Springer (2004); 
K. Kendre: "__Machine Learning for Quantum Noise Reduction__", https://arxiv.org/abs/2509.16242 (2025); D. C. Marinescu, G. M. Marinescu: "__Classical and Quantum Information__", 1sr Ed. Academic Press (2012); 
Liao, H et al.: "__Machine Learning for Practical Quantum Error Mitigation__", arXiv:2309.17368v2 (2024), https://arxiv.org/pdf/2309.17368; Streamlit: https://streamlit.io/; 
Mitiq-package: https://quantum-journal.org/papers/q-2022-08-11-774/, https://arxiv.org/abs/2009.04417; Extrapolation packages: https://pypi.org/project/extrapolation/  
39. A. Koop, H. Moock: "__Lineare Optimierung - Eine anwendungsorientierte Einführung in Operations Research__", 1st Ed. Spektrum (2008); 
G, B, Dantzig, M. N. Thalpa: "__Linear Programming 1: Introduction__", 1st Ed. Springer (1997) & "__Linear Programming 2: Theory and Extensions__", 1st Ed. Springer (2003); 
H. S. Kasana, K. D. Kumar: "__Introductory Operations Research, Theory and Applications__", 1st Ed. Springer (2004); D. G. Luenberger: "__Linear and Nonlinear Programming__", 2nd Ed. Kluwer (2004); 
R. J. Boucherie, A. Braaksma, H. Tijms: "__Operations Research - Introduction to Models and Methods__", 1st Ed. World Scientific (2022); 
A. J. King, S. W. Wallace: "__Modeling with Stochastic Programming__", 2nd Ed. Springer (2024); 
J. O. Royset, R. J.-B. Wets: "__An Optimization Primer__", 1st Ed. Springer (2021); cvxpy package: https://www.cvxpy.org/, https://pypi.org/project/cvxpy/;
py-packages for operations research: https://wiki.python.org/moin/PythonForOperationsResearch





















