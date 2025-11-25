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
