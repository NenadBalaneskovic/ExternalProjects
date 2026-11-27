#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  run.py
ROLE:    Global execution entry point initializing environment contexts,
         asynchronous platform loops, and the main PyQt6 visualization window.
================================================================================
"""

import os
import sys
import logging
from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication

# Import out of our local View architecture context
# Note: Ensure the gui/ directory package is fully populated
from gui.main_window import MainDashboardWindow

# Setup production-grade telemetry logging format for console diagnostics
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] (%(name)s): %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("SystemRuntimeCore")


def initialize_system_environment():
    """
    Validates filesystem states and synchronizes the locally stored provider
    API keys out of the .env layout. No browser session cookies are used.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, '.env')

    # Automated synthesis of empty secure state file if missing
    if not os.path.exists(env_path):
        log.warning("Configuration layer [.env] missing. Synthesizing clean template...")
        try:
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write("# ====================================================================\n")
                f.write("# AI ROUTER & HALLUCINATION SHIELD - LOCAL API KEY STORE\n")
                f.write("# DO NOT COMMIT TO REPOSITORIES - PRIVATE CREDENTIALS\n")
                f.write("# Fill these in via the GUI's 'Account-Status' button, or paste them\n")
                f.write("# here directly after generating them on each provider's own site:\n")
                f.write("#   GEMINI_API_KEY  -> https://aistudio.google.com/apikey\n")
                f.write("#   GROQ_API_KEY    -> https://console.groq.com/keys\n")
                f.write("#   MISTRAL_API_KEY -> https://console.mistral.ai/api-keys\n")
                f.write("#   TAVILY_API_KEY  -> https://app.tavily.com\n")
                f.write("# ====================================================================\n\n")
                f.write("GEMINI_API_KEY=\n")
                f.write("GROQ_API_KEY=\n")
                f.write("MISTRAL_API_KEY=\n")
                f.write("TAVILY_API_KEY=\n")
            log.info("Clean environment template written to disk.")
        except IOError as e:
            log.critical(f"Fatal disruption writing environment wrapper layer: {str(e)}")
            sys.exit(1)
    else:
        # Inject existing variables into active process context arrays
        load_dotenv(env_path)
        log.info("Provider API-key configuration linked to execution environment.")


def main():
    """
    Main execution orchestration framework initializing the global QApplication loop
    and mounting the production dashboard interface.
    """
    log.info("Initializing AI Router & Hallucination Shield runtime instance...")

    # 1. Fire-up local disk system audits
    initialize_system_environment()

    # 2. Spawn the Qt application loop. (The former QtWebEngine-specific
    #    command-line flags, --enable-gpu-rasterization and
    #    --smart-scrollbar-blending, are no longer relevant now that the GUI
    #    has no embedded Chromium browser view, so they have been removed.)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Secures seamless styling transitions across OS platforms

    log.info("PyQt6 application subsystem online. Mounting Graphical User Interface layout views...")

    try:
        # 3. Mount the main dashboard terminal view
        dashboard = MainDashboardWindow()
        dashboard.show()

        log.info("System operational dashboard completely mounted. Event execution loop engaged.")
        # 4. Hand-off execution scope completely to the safe Qt thread runtime loop
        sys.exit(app.exec())

    except Exception as e:
        log.critical(f"Unhandled phase transition collapse encountered during startup: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()