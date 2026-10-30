# `gui_design.md`

```markdown
# GUI Design — Slurm HPC–QPU Workflow Orchestrator

This document describes the design principles, component hierarchy, layout system, and event‑loop architecture of the orchestrator’s GUI.  
The GUI is built using **PySimpleGUI**, with a strict separation between layout, components, utilities, and logic.

The design emphasizes:
- safety (no workflow execution)
- modularity
- testability
- deterministic behavior
- reproducibility across environments

---

## 1. Design Philosophy

The GUI follows four core principles:

### 1.1 Pure Layout
All GUI layout modules are **pure factories**:
- no event loop
- no workflow execution
- no side effects

This ensures deterministic behavior and easy testing.

### 1.2 Strict Separation of Concerns
The GUI is divided into:
- **components** (atomic widgets)
- **layout** (composition of components)
- **utils** (validation, dialogs)
- **main_gui** (event loop + orchestration)
- **theme_manager** (styling)

### 1.3 Safety
The GUI never:
- executes workflow code  
- imports workflow modules  
- evaluates user input  
- performs dynamic execution  

All workflow inspection is done via static AST analysis.

### 1.4 Deterministic Behavior
Every GUI action produces predictable results:
- upload → static AST analysis  
- generate → deterministic template substitution  
- preview → static multiline update  

---

## 2. GUI Architecture Overview

```
gui/
    main_gui.py
    components.py
    file_dialogs.py
    theme_manager.py
    styles.css
    utils/
        validators.py
    layout/
        main_layout.py
        workflow_analysis_panel.py
        slurm_preview_panel.py
        credentials_section.py
```

Each module has a single responsibility.

---

## 3. Component Hierarchy

### 3.1 Atomic Components (`components.py`)
Atomic components include:
- **upload_section**  
- **analysis_panel**  
- **slurm_preview_panel**  
- **credentials_section**  
- **generate_button**  

These components are reusable and stateless.

### 3.2 Layout Composition (`layout/`)
Layout modules assemble atomic components into structured GUI sections:

- `main_layout.py`  
- `workflow_analysis_panel.py`  
- `slurm_preview_panel.py`  
- `credentials_section.py`  

Each layout module returns a **pure PySimpleGUI element tree**.

---

## 4. Main Window Structure

The main window is composed of five sections:

1. **Workflow Upload Section**  
2. **Workflow Analysis Panel**  
3. **Slurm Preview Panel**  
4. **Credentials Toggle + Section**  
5. **Generate Slurm Button**

### 4.1 Visual Structure

```
+--------------------------------------------------------------+
| Slurm HPC–QPU Workflow Orchestrator                          |
+--------------------------------------------------------------+
| Workflow Upload                                              |
| [Input] [Browse] [Upload]                                   |
+--------------------------------------------------------------+
| Workflow Analysis                                            |
| [Multiline: AST + classification]                           |
+--------------------------------------------------------------+
| Slurm Preview                                                |
| [Multiline: generated script]                               |
+--------------------------------------------------------------+
| [ ] Enable manual QPU credentials                           |
| (hidden) API Key, Runtime URL                               |
+--------------------------------------------------------------+
| [Generate Slurm Script]                                     |
+--------------------------------------------------------------+
```

---

## 5. Event Loop Design (`main_gui.py`)

The event loop is intentionally simple and deterministic:

### 5.1 Event Types

- `UPLOAD_BUTTON`  
- `GENERATE_SLURM`  
- `ENABLE_CREDS`  
- `WIN_CLOSED`  

### 5.2 Event Flow

#### Upload Workflow
1. Validate path  
2. Parse AST  
3. Classify workflow  
4. Update analysis panel  

#### Generate Slurm Script
1. Re‑parse AST  
2. Re‑classify workflow  
3. Prepare substitution dictionary  
4. Generate script  
5. Update preview panel  

#### Toggle Credentials
Show/hide credential section.

---

## 6. Safety Model

The GUI enforces strict safety:

### 6.1 No Execution of User Code
The GUI never:
- imports workflow modules  
- executes workflow functions  
- evaluates workflow strings  
- loads dynamic code  

### 6.2 Static Analysis Only
All workflow inspection uses:
- Python AST  
- static import detection  
- static call detection  

### 6.3 Deterministic Template Substitution
Slurm scripts are generated from static templates.

### 6.4 No Side Effects
The GUI only:
- reads workflow files  
- writes Slurm scripts  
- updates GUI elements  

---

## 7. Theming System

Themes are defined in:

```
core/themes/darkblue3_theme.json
```

Applied via:

- **ThemeManager**

CSS variables in `styles.css` map directly to theme tokens.

This enables future migration to:
- Qt  
- Tkinter ttk  
- PyWebView  
- Electron/Tauri  

---

## 8. Validation System

GUI validation is handled by:

- **gui.utils.validators**

It validates:
- workflow path  
- credential fields  
- GUI input consistency  

Errors are returned as dictionaries for easy display.

---

## 9. Extensibility

The GUI is designed for future expansion:

### 9.1 Workflow Graph Visualization
Add:
```
gui/layout/workflow_graph_panel.py
```

### 9.2 Theme Switching Menu
Add:
```
gui/layout/theme_switcher.py
```

### 9.3 Export Settings Dialog
Add:
```
gui/layout/export_settings_dialog.py
```

### 9.4 Progress Bar + Status Bar
Add:
```
gui/components/progress_bar.py
gui/components/status_bar.py
```

---

## 10. Summary

The GUI is:
- modular  
- deterministic  
- safe  
- testable  
- extensible  

Its architecture cleanly separates:
- components  
- layout  
- utilities  
- theming  
- event loop  

This ensures long‑term maintainability and scientific reproducibility.

```

---
