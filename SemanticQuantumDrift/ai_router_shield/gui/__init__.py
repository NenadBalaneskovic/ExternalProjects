# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  gui/__init__.py
ROLE:    Package initializer elevating main dashboard containers and dialog
         windows to the unified view namespace.
================================================================================
"""

# Explicit promotion of user interface elements to secure modular view imports
from gui.main_window import MainDashboardWindow
from gui.history_window import ChatHistoryDialog
from gui.auth_window import ApiKeyEntryDialog, AccountStatusCenterDialog

# Define the explicit public API export layout of the graphical interface package
__all__ = [
    "MainDashboardWindow",
    "ChatHistoryDialog",
    "ApiKeyEntryDialog",
    "AccountStatusCenterDialog"
]