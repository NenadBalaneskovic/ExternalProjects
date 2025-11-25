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
