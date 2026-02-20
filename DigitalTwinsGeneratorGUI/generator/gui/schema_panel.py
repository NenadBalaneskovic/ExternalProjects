# generator/gui/schema_panel.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QCheckBox, QLabel
)
from PySide6.QtCore import Qt


class SchemaPanel(QWidget):
    """
    Panel containing the 15 predefined telemetry column checkboxes.
    Columns are grouped into:
        - Numeric Sensors
        - Categorical / Boolean
        - Auxiliary (timestamp, logs, cycle counter)

    Provides:
        get_schema() -> list of dicts describing selected columns
    """

    def __init__(self):
        super().__init__()

        self.setMinimumWidth(280)
        self._build_ui()

    # ---------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()

        # --- Numeric Sensors ---
        numeric_group = QGroupBox("Numeric Sensors")
        numeric_layout = QVBoxLayout()
        self.cb_temperature = QCheckBox("Temperature")
        self.cb_rpm = QCheckBox("Motor RPM")
        self.cb_vibration = QCheckBox("Vibration Level")
        self.cb_power = QCheckBox("Power Consumption")
        self.cb_voltage = QCheckBox("Voltage")
        self.cb_current = QCheckBox("Current")
        self.cb_pressure = QCheckBox("Pressure / Load")
        self.cb_noise = QCheckBox("Noise Level")

        for cb in [
            self.cb_temperature, self.cb_rpm, self.cb_vibration,
            self.cb_power, self.cb_voltage, self.cb_current,
            self.cb_pressure, self.cb_noise
        ]:
            numeric_layout.addWidget(cb)

        numeric_group.setLayout(numeric_layout)
        layout.addWidget(numeric_group)

        # --- Categorical / Boolean ---
        categorical_group = QGroupBox("Categorical / Boolean")
        categorical_layout = QVBoxLayout()
        self.cb_onoff = QCheckBox("Device On/Off")
        self.cb_mode = QCheckBox("Operating Mode")
        self.cb_error = QCheckBox("Error Code")
        self.cb_interlock = QCheckBox("Safety Interlock")

        for cb in [
            self.cb_onoff, self.cb_mode,
            self.cb_error, self.cb_interlock
        ]:
            categorical_layout.addWidget(cb)

        categorical_group.setLayout(categorical_layout)
        layout.addWidget(categorical_group)

        # --- Auxiliary ---
        auxiliary_group = QGroupBox("Auxiliary Columns")
        auxiliary_layout = QVBoxLayout()
        self.cb_timestamp = QCheckBox("Timestamp")
        self.cb_log = QCheckBox("Log Message")
        self.cb_cycle = QCheckBox("Cycle Counter")

        for cb in [
            self.cb_timestamp, self.cb_log, self.cb_cycle
        ]:
            auxiliary_layout.addWidget(cb)

        auxiliary_group.setLayout(auxiliary_layout)
        layout.addWidget(auxiliary_group)

        layout.addStretch()
        self.setLayout(layout)

    # ---------------------------------------------------------
    # Schema Extraction
    # ---------------------------------------------------------
    def get_schema(self):
        """
        Returns a list of dictionaries describing the selected columns.
        Each entry has:
            - name: column name
            - type: float/int/categorical/boolean/text
            - generator: name of simulation function (string)
            - optional metadata (unit, categories)

        This schema is consumed by:
            - TelemetryGenerator
            - config_writer (to write config.json)
        """
        schema = []

        # Numeric sensors
        if self.cb_temperature.isChecked():
            schema.append({
                "name": "Temperature",
                "type": "float",
                "unit": "Celsius",
                "generator": "simulate_temperature"
            })
        if self.cb_rpm.isChecked():
            schema.append({
                "name": "Motor RPM",
                "type": "int",
                "unit": "rpm",
                "generator": "simulate_rpm"
            })
        if self.cb_vibration.isChecked():
            schema.append({
                "name": "Vibration Level",
                "type": "float",
                "unit": "m/s2",
                "generator": "simulate_vibration"
            })
        if self.cb_power.isChecked():
            schema.append({
                "name": "Power Consumption",
                "type": "float",
                "unit": "W",
                "generator": "simulate_power"
            })
        if self.cb_voltage.isChecked():
            schema.append({
                "name": "Voltage",
                "type": "float",
                "unit": "V",
                "generator": "simulate_voltage"
            })
        if self.cb_current.isChecked():
            schema.append({
                "name": "Current",
                "type": "float",
                "unit": "A",
                "generator": "simulate_current"
            })
        if self.cb_pressure.isChecked():
            schema.append({
                "name": "Pressure / Load",
                "type": "float",
                "unit": "arb",
                "generator": "simulate_pressure"
            })
        if self.cb_noise.isChecked():
            schema.append({
                "name": "Noise Level",
                "type": "float",
                "unit": "dB",
                "generator": "simulate_noise"
            })

        # Categorical / Boolean
        if self.cb_onoff.isChecked():
            schema.append({
                "name": "Device On/Off",
                "type": "boolean",
                "generator": "simulate_onoff"
            })
        if self.cb_mode.isChecked():
            schema.append({
                "name": "Operating Mode",
                "type": "categorical",
                "categories": ["Idle", "Low", "High"],
                "generator": "simulate_mode"
            })
        if self.cb_error.isChecked():
            schema.append({
                "name": "Error Code",
                "type": "categorical",
                "categories": ["None", "Minor", "Major"],
                "generator": "simulate_error"
            })
        if self.cb_interlock.isChecked():
            schema.append({
                "name": "Safety Interlock",
                "type": "boolean",
                "generator": "simulate_interlock"
            })

        # Auxiliary
        if self.cb_timestamp.isChecked():
            schema.append({
                "name": "Timestamp",
                "type": "timestamp",
                "format": "ISO8601",
                "generator": "simulate_timestamp"
            })
        if self.cb_log.isChecked():
            schema.append({
                "name": "Log Message",
                "type": "text",
                "generator": "simulate_log"
            })
        if self.cb_cycle.isChecked():
            schema.append({
                "name": "Cycle Counter",
                "type": "int",
                "generator": "simulate_cycle"
            })

        return schema