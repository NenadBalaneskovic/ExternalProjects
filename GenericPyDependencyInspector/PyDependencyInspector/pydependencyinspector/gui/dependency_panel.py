"""
gui/dependency_panel.py

Dependency tree panel for PyDependencyInspector.

Responsibilities:
- Display the resolved dependency tree in a QTreeWidget.
- Update dynamically when MainWindow triggers a new resolution.
- Emit selection events for the DocumentationPanel.
- Apply dark graphite + cyan styling.

This module is intentionally:
- GUI-only (no resolver logic).
- Modular and clean.
- Driven by ProjectState.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QMenu,
)
from PySide6.QtCore import Qt, Signal

from ..core.dependency_resolver import DependencyNode, ResolutionResult


class DependencyPanel(QWidget):
    """
    Displays the dependency tree.

    Signals:
    - dependencySelected(str): emitted when a dependency is clicked
    """

    dependencySelected = Signal(str)

    def __init__(self, project_state) -> None:
        super().__init__()
        self.project_state = project_state

        self._init_ui()
        self._apply_styles()

    # ------------------------------------------------------------------ #
    # UI initialization
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Package", "Version", "Type"])
        self.tree.setColumnWidth(0, 260)
        self.tree.setColumnWidth(1, 120)
        self.tree.setColumnWidth(2, 120)

        # Connect signals
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)

        layout.addWidget(self.tree)

    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QTreeWidget {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #333;
                font-size: 13px;
            }
            QTreeWidget::item:selected {
                background-color: #0e639c;
                color: white;
            }
            QTreeWidget::item {
                padding: 4px;
            }
        """)

    # ------------------------------------------------------------------ #
    # Updating the panel
    # ------------------------------------------------------------------ #

    def update_with_resolution(self, result: ResolutionResult) -> None:
        """
        Update the tree widget with a new dependency resolution result.
        """
        self.tree.clear()

        if not result or not result.root:
            return

        root_item = self._create_item(result.root)
        self.tree.addTopLevelItem(root_item)

        self._populate_children(root_item, result.root)
        self.tree.expandToDepth(1)

    def _populate_children(self, parent_item: QTreeWidgetItem, node: DependencyNode) -> None:
        for child in node.children:
            item = self._create_item(child)
            parent_item.addChild(item)
            self._populate_children(item, child)

    def _create_item(self, node: DependencyNode) -> QTreeWidgetItem:
        """
        Create a QTreeWidgetItem for a dependency node.
        """
        name = node.name or ""
        version = node.version or ""
        dep_type = node.dep_type.value

        item = QTreeWidgetItem([name, version, dep_type])

        # Store package name for documentation panel
        item.setData(0, Qt.UserRole, name)

        return item

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """
        Emit dependencySelected when a package is clicked.
        """
        pkg = item.data(0, Qt.UserRole)
        if pkg:
            self.dependencySelected.emit(pkg)

    def _on_context_menu(self, pos) -> None:
        """
        Right-click context menu for future extensibility.
        """
        item = self.tree.itemAt(pos)
        if not item:
            return

        pkg = item.data(0, Qt.UserRole)
        if not pkg:
            return

        menu = QMenu(self)

        action_doc = menu.addAction("Show Documentation")
        action_doc.triggered.connect(lambda: self.dependencySelected.emit(pkg))

        menu.exec(self.tree.mapToGlobal(pos))
