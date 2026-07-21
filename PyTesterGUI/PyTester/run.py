import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "mplconfig")

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import logging
import logging.config
import yaml

# GUI entrypoint
from gui.main_window import MainWindow

# Core subsystems
from core.input_loader import InputLoader
from core.syntax_checker import SyntaxChecker
from core.safe_import import SafeImporter
from core.ast_inspector import ASTInspector
from core.docstring_extractor import DocstringExtractor
from core.annotation_extractor import AnnotationExtractor
from core.structure_registry import StructureRegistry

# Inference subsystems
from inference.static_analysis import StaticAnalyzer
from inference.semantic_analysis import SemanticAnalyzer
from inference.dynamic_probe import DynamicProbe
from inference.type_fusion import TypeFusion
from inference.schema_builder import SchemaBuilder

# Test generation subsystems
from testgen.smoke_generator import SmokeTestGenerator
from testgen.type_tests_generator import TypeTestsGenerator
from testgen.boundary_tests_generator import BoundaryTestsGenerator
from testgen.property_tests_generator import PropertyTestsGenerator
from testgen.docstring_tests_generator import DocstringTestsGenerator
from testgen.template_renderer import TemplateRenderer

# Execution subsystems
from executor.pytest_runner import PytestRunner
from executor.coverage_runner import CoverageRunner
from executor.report_collector import ReportCollector
from executor.log_capture import LogCapture

# Visualization subsystems
from visualization.plot_results import PlotResults
from visualization.plot_durations import PlotDurations
from visualization.plot_failures import PlotFailures
from visualization.plot_coverage import PlotCoverage
from visualization.png_exporter import PNGExporter


def load_settings() -> dict:
    settings_path = Path("config/settings.yaml")
    with open(settings_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_logging() -> None:
    logging.config.fileConfig("config/logging.conf")


def initialize_subsystems(settings: dict) -> dict:
    # Create shared subsystem instances FIRST
    annotation_extractor = AnnotationExtractor(settings)
    docstring_extractor = DocstringExtractor(settings)
    safe_importer = SafeImporter(settings)
    structure_registry = StructureRegistry(settings)

    return {
        "input_loader": InputLoader(settings),
        "syntax_checker": SyntaxChecker(settings),
        "safe_importer": safe_importer,
        "ast_inspector": ASTInspector(settings),
        "docstring_extractor": docstring_extractor,
        "annotation_extractor": annotation_extractor,
        "structure_registry": structure_registry,

        "static_analyzer": StaticAnalyzer(settings, annotation_extractor, docstring_extractor),
        "semantic_analyzer": SemanticAnalyzer(settings),
        "dynamic_probe": DynamicProbe(settings, safe_importer),
        "type_fusion": TypeFusion(settings),

        "schema_builder": SchemaBuilder(settings, structure_registry),

        "smoke_generator": SmokeTestGenerator(settings),
        "type_tests_generator": TypeTestsGenerator(settings),
        "boundary_tests_generator": BoundaryTestsGenerator(settings),
        "property_tests_generator": PropertyTestsGenerator(settings),
        "docstring_tests_generator": DocstringTestsGenerator(settings),
        "template_renderer": TemplateRenderer(settings),

        "pytest_runner": PytestRunner(settings),
        "coverage_runner": CoverageRunner(settings),
        "report_collector": ReportCollector(settings),
        "log_capture": LogCapture(settings),

        "plot_results": PlotResults(settings),
        "plot_durations": PlotDurations(settings),
        "plot_failures": PlotFailures(settings),
        "plot_coverage": PlotCoverage(settings),
        "png_exporter": PNGExporter(settings),
    }


def main() -> None:
    from PyQt5.QtWidgets import QApplication

    # QApplication MUST be created before any QWidget
    qt_app = QApplication(sys.argv)

    settings = load_settings()
    configure_logging()
    subsystems = initialize_subsystems(settings)

    app = MainWindow(settings, subsystems)
    app.show()

    sys.exit(qt_app.exec_())


if __name__ == "__main__":
    main()
