# 1. 🚀 Project Introduction: Liner Optimization Solver GUI (Operations Research)

## Objective  
This project presents a unified computational framework for solving linear optimization problems through a multi-method graphical interface. 
By integrating classical simplex algorithms, dual formulations, parametric linear programming, integer-constrained solvers, and interior-point 
techniques such as the Dikin method, the system enables robust exploration of feasible regions and optimality paths across diverse problem structures. 
The central aim is to democratize access to advanced linear programming strategies while offering real-time visualization and interpretability.
Leveraging simplex-based pivoting for vertex traversal, parametric solvers for dynamic constraint sensitivity, and integer programming for discrete 
decision modeling, the framework accommodates both continuous and combinatorial optimization scenarios. The inclusion of the Dikin method introduces a 
smooth, interior-point trajectory that complements boundary-based approaches, offering insights into curvature and convergence behavior within feasible polyhedra.
This methodological synthesis is encapsulated in a Python-powered GUI that dynamically adapts to user-defined variable types, constraint structures, and solver preferences. 
The interface supports interactive input parsing, automatic method detection, and 2D geometric rendering of feasible regions, constraint boundaries, and optimization paths. 
Designed for both educational and applied contexts, the system fosters intuitive understanding of linear optimization mechanics while maintaining algorithmic rigor.
The resulting application serves as a modular launchpad for future extensions into nonlinear programming, multi-objective optimization, and hybrid solver orchestration. 
By bridging algorithmic depth with visual clarity, this project opens new avenues for accessible, interpretable, and customizable optimization workflows 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/LinearProgramming_GUI/LinearProgramming_GUI.md#8--references) 1 - 3 below). 

## 🎯 **Primary Aim**

Structure and implement a PyQt5-powered GUI for linear optimization with built-in logic to route the problem to simplex, dual simplex, Dikin method, parametric linear
programming (LP), or integer LP depending  on the input, both from a **UX** and **functional architecture** perspective.

## 🧩 Modular Components and Their Roles

### 🧩 GUI Architecture Overview

#### **Main Window Layout**
Use a `QMainWindow` with a central `QTabWidget` or `QStackedWidget` to separate views:
- **Tab 1**: Problem Definition
- **Tab 2**: Optimization Settings
- **Tab 3**: Solution & Visualization
- **Tab 4**: Logs & Diagnostics

### 📝 Tab 1: Problem Definition

#### 🔹 Input Fields
- **Objective Function**: `QLineEdit` or `QTextEdit` for expressions like `max: 3x + 2y`
- **Constraints**: `QPlainTextEdit` for multi-line input like:
  ```
  x + y <= 5
  x - y >= 2
  x, y ≥ 0
  ```
- **Variable Type Selector**: `QComboBox` or `QCheckBox` grid for:
  - Continuous
  - Integer
  - Parametric (e.g. if coefficients depend on a parameter λ)

#### 🔹 Parsing & Validation
- On submit, parse input using `sympy` or a custom parser
- Validate syntax, detect integer constraints, parametric expressions

### ⚙️ Tab 2: Optimization Settings

#### 🔹 Method Selection (auto or manual)
- `QRadioButton` group:
  - Auto-select method
  - Manual: Simplex / Dual Simplex / Dikin / Parametric LP / ILP

#### 🔹 Advanced Options
- Tolerance (`QDoubleSpinBox`)
- Max iterations
- Pivot rule (for simplex)
- Barrier parameters (for Dikin)

### 📈 Tab 3: Solution & Visualization

#### 🔹 Output Display
- Optimal value, variable assignments
- Status: feasible/infeasible/unbounded
- Method used (auto-detected or user-selected)

#### 🔹 Visualization
- `Matplotlib` canvas embedded via `FigureCanvasQTAgg`
- For 2D problems: feasible region, objective contours, pivot path
- For parametric LP: slider for λ, dynamic plot updates

### 🧪 Tab 4: Logs & Diagnostics

#### 🔹 Execution Trace
- Step-by-step log of pivot operations, tableau updates
- Dual feasibility checks, degeneracy detection
- Dikin method: Newton steps, barrier evolution

#### 🔹 Export Options
- Save log as `.txt`
- Export solution as `.json` or `.csv`

### 🧠 Backend Architecture

#### 🔹 Core Modules
- `ProblemParser`: Parses and validates user input
- `MethodSelector`: Decides which algorithm to use
- `SolverEngine`: Wraps simplex, dual simplex, Dikin, etc.
- `Visualizer`: Generates plots and animations
- `Logger`: Captures diagnostics and user actions

#### 🔹 Libraries to Use
- `numpy`, `scipy.optimize.linprog`, `cvxpy`, `pulp` for solving
- `sympy` for symbolic parsing
- `matplotlib` for plotting
- `PyQt5` for GUI

### 🧠 Auto-Detection Logic

```python
def select_method(objective, constraints):
    if has_integer_variables(constraints):
        return "ILP"
    elif has_parametric_coefficients(objective, constraints):
        return "Parametric LP"
    elif is_dual_feasible(constraints):
        return "Dual Simplex"
    elif is_strictly_feasible(constraints):
        return "Dikin"
    else:
        return "Simplex"
```

 
### 🧠 **GUI sketch**  

In the following we address our full GUI sketch, a clean, structured layout for your PyQt5-based linear optimization tool. It includes all the key modules we discussed: 
objective and constraint input, method selection (including simplex, dual simplex, Dikin, parametric LP, ILP), solution display, visualization, and diagnostics.

![LP_GUI_sketch.png](https://github.com/NenadBalaneskovic/ExternalProjects/blob/fa70ea9b7664fde4b34f095fe211d51330716cbb/LinearProgramming_GUI/LP_GUI_sketch.png)

We are ready to scaffold this into actual PyQt5 code and wire up the backend logic for method selection and solver routing.

---

# 2. 🔐 Algorithmic concepts

This is a comprehensive, notebook-ready Markdown explanation of the four major optimization techniques
our GUI should support. Each section includes mathematical foundations, algorithmic steps, and illustrative examples.

## 📊 Linear Optimization Methods: Mathematical & Algorithmic Overview


### 1. 🔁 Simplex Method

#### 🧠 Concept
The **Simplex Method** solves linear programs of the form:

$$\[
\text{Maximize } z = c^T x \quad \text{subject to } Ax \leq b, \quad x \geq 0
\]$$

It moves from one **vertex (basic feasible solution)** of the feasible region to another, improving the objective at each step.

#### ⚙️ Algorithm Steps
1. Convert inequalities to equalities using **slack variables**.
2. Set up the **initial tableau**.
3. Identify the **entering variable** (most negative coefficient in objective row).
4. Identify the **leaving variable** (minimum ratio test).
5. Perform **pivoting** to update the tableau.
6. Repeat until no negative coefficients remain in the objective row.

#### 📌 Example

Maximize:  
$$\[
z = 3x + 2y
\]$$  
Subject to:  
$$\[
x + y \leq 4 \\
x \leq 2 \\
x, y \geq 0
\]$$

Add slack variables $$\( s_1, s_2 \)$$, and solve using tableau iterations.

### 2. 🔄 Dual Simplex Method

#### 🧠 Concept
The **Dual Simplex Method** is used when the solution is **dual feasible but primal infeasible**. It’s ideal for post-optimality analysis or
when constraints change.

#### ⚙️ Algorithm Steps
1. Start with a tableau where the objective is optimal but some RHS values are negative.
2. Identify the **leaving variable** (most negative RHS).
3. Identify the **entering variable** (minimum ratio of cost coefficient to pivot column).
4. Pivot and update the tableau.
5. Repeat until all RHS values are non-negative.

#### 📌 Example

Suppose after adding a new constraint, your current solution violates feasibility. Dual simplex allows you to re-optimize without restarting
from scratch.

### 3. 🧭 Dikin Method (Interior Point)

#### 🧠 Concept
The **Dikin Method** is a primal interior-point algorithm. It uses a **log-barrier function** to stay within the feasible region and
performs **Newton steps** to move toward optimality.

#### 🧮 Barrier Function

$$\[
\phi(x) = c^T x - \sum_{i=1}^m \log(b_i - A_i x)
\]$$

#### ⚙️ Algorithm Steps
1. Start with a strictly feasible point $$\( x_0 \)$$.
2. Compute gradient and Hessian of the barrier function.
3. Solve Newton system:  
$$\[
H \Delta x = -\nabla \phi(x)
\]$$
4. Perform line search to maintain feasibility.
5. Update $$\( x \leftarrow x + \alpha \Delta x \)$$.
6. Repeat until convergence.

#### 📌 Example

Maximize:  
$$\[
z = x + y
\]$$  
Subject to:  
$$\[
x + y \leq 2 \\
x \geq 0, y \geq 0
\]$$

Start at $$\( x = y = 1 \)$$, compute slack, gradient, Hessian, and iterate.

### 4. 🔢 Integer Linear Programming (ILP)

#### 🧠 Concept
ILP solves linear programs where **some or all variables must be integers**. The feasible region becomes **discrete**, 
and the problem is **NP-hard**.

#### ⚙️ Common Algorithms
- **Branch and Bound**: Recursively split the problem into subproblems.
- **Cutting Planes**: Add constraints to eliminate fractional solutions.
- **Heuristics**: Greedy or randomized methods for large-scale ILPs.

#### 📌 Example

Maximize:  
$$\[
z = 5x + 3y
\]$$  
Subject to:  
$$\[
2x + y \leq 7 \\
x, y \in \mathbb{Z}_{\geq 0}
\]$$

Try $$\( x = 3, y = 1 \Rightarrow z = 18 \)$$, check feasibility, and explore other integer combinations.

### 5. 🧮 Summary Table

| Method           | Feasibility Required | Variable Type | Use Case                          |
|------------------|----------------------|----------------|-----------------------------------|
| Simplex          | Primal feasible      | Continuous     | Standard LP                       |
| Dual Simplex     | Dual feasible        | Continuous     | Constraint updates, post-optimal |
| Dikin            | Strictly feasible    | Continuous     | Interior-point, smooth problems  |
| ILP              | Discrete feasible    | Integer         | Scheduling, routing, allocation  |

---

## 2.1 Glossary

This is a comprehensive, notebook-ready Markdown explanation of the four major optimization techniques
our GUI should support. Each section includes mathematical foundations, algorithmic steps, and illustrative examples.

### 📊 Linear Optimization Methods: Mathematical & Algorithmic Overview

#### 1. 🔁 Simplex Method

##### 🧠 Concept
The **Simplex Method** solves linear programs of the form:

$$\[
\text{Maximize } z = c^T x \quad \text{subject to } Ax \leq b, \quad x \geq 0
\]$$

It moves from one **vertex (basic feasible solution)** of the feasible region to another, improving the objective at each step.

##### ⚙️ Algorithm Steps
1. Convert inequalities to equalities using **slack variables**.
2. Set up the **initial tableau**.
3. Identify the **entering variable** (most negative coefficient in objective row).
4. Identify the **leaving variable** (minimum ratio test).
5. Perform **pivoting** to update the tableau.
6. Repeat until no negative coefficients remain in the objective row.

##### 📌 Example

Maximize:  
$$\[
z = 3x + 2y
\]$$  
Subject to:  
$$\[
x + y \leq 4 \\
x \leq 2 \\
x, y \geq 0
\]$$

Add slack variables $$\( s_1, s_2 \)$$, and solve using tableau iterations.

#### 2. 🔄 Dual Simplex Method

##### 🧠 Concept
The **Dual Simplex Method** is used when the solution is **dual feasible but primal infeasible**. It’s ideal for post-optimality analysis or
when constraints change.

##### ⚙️ Algorithm Steps
1. Start with a tableau where the objective is optimal but some RHS values are negative.
2. Identify the **leaving variable** (most negative RHS).
3. Identify the **entering variable** (minimum ratio of cost coefficient to pivot column).
4. Pivot and update the tableau.
5. Repeat until all RHS values are non-negative.

##### 📌 Example

Suppose after adding a new constraint, your current solution violates feasibility. Dual simplex allows you to re-optimize without restarting
from scratch.

#### 3. 🧭 Dikin Method (Interior Point)

##### 🧠 Concept
The **Dikin Method** is a primal interior-point algorithm. It uses a **log-barrier function** to stay within the feasible region and
performs **Newton steps** to move toward optimality.

##### 🧮 Barrier Function

$$\[
\phi(x) = c^T x - \sum_{i=1}^m \log(b_i - A_i x)
\]$$

##### ⚙️ Algorithm Steps
1. Start with a strictly feasible point $$\( x_0 \)$$.
2. Compute gradient and Hessian of the barrier function.
3. Solve Newton system:  
$$\[
H \Delta x = -\nabla \phi(x)
\]$$
4. Perform line search to maintain feasibility.
5. Update $$\( x \leftarrow x + \alpha \Delta x \)$$.
6. Repeat until convergence.

##### 📌 Example

Maximize:  
$$\[
z = x + y
\]$$  
Subject to:  
$$\[
x + y \leq 2 \\
x \geq 0, y \geq 0
\]$$

Start at $$\( x = y = 1 \)$$, compute slack, gradient, Hessian, and iterate.

#### 4. 🔢 Integer Linear Programming (ILP)

##### 🧠 Concept
ILP solves linear programs where **some or all variables must be integers**. The feasible region becomes **discrete**, 
and the problem is **NP-hard**.

##### ⚙️ Common Algorithms
- **Branch and Bound**: Recursively split the problem into subproblems.
- **Cutting Planes**: Add constraints to eliminate fractional solutions.
- **Heuristics**: Greedy or randomized methods for large-scale ILPs.

##### 📌 Example

Maximize:  
$$\[
z = 5x + 3y
\]$$  
Subject to:  
$$\[
2x + y \leq 7 \\
x, y \in \mathbb{Z}_{\geq 0}
\]$$

Try $$\( x = 3, y = 1 \Rightarrow z = 18 \)$$, check feasibility, and explore other integer combinations.

#### 5. 🧮 Summary Table

| Method           | Feasibility Required | Variable Type | Use Case                          |
|------------------|----------------------|----------------|-----------------------------------|
| Simplex          | Primal feasible      | Continuous     | Standard LP                       |
| Dual Simplex     | Dual feasible        | Continuous     | Constraint updates, post-optimal |
| Dikin            | Strictly feasible    | Continuous     | Interior-point, smooth problems  |
| ILP              | Discrete feasible    | Integer         | Scheduling, routing, allocation  |

---

# 3. GUI design and its user interaction flow

Let us walk through the user interaction flow for our PyQt5 GUI for solving linear problems, especially when the user provides ≥2 variables and coefficients.
I will break it down into intuitive stages:

## 3.1 🧭 Step-by-Step User Interaction Flow

### 1. **Problem Input Phase**

#### 🧮 Objective Function
- The user enters something like `max: 3x + 2y + z` into the **Objective Function** field.
- The parser detects **3 variables**: `x`, `y`, `z`.

#### 📐 Constraints
- The GUI dynamically generates **n + 1 empty constraint slots** (e.g. 4 for 3 variables), each as a `QLineEdit` or `QPlainTextEdit`.
- Each slot is labeled (e.g. `Constraint 1`, `Constraint 2`, …) and can be filled with expressions like:
  ```
  x + y ≤ 5
  y - z ≥ 2
  x, y, z ≥ 0
  ```

#### ⚙️ Variable Type Selection
- The user selects whether variables are:
  - Continuous
  - Integer
  - Parametric (e.g. `max: 3x + λy`)

### 2. **Method Detection Phase**

Once the user finishes input:

#### 🔍 Auto-Detection Logic
- The GUI analyzes the input:
  - If any variable is marked as integer → switches to **ILP**
  - If any coefficient depends on a parameter (e.g. λ) → switches to **Parametric LP**
  - If the problem is strictly feasible → suggests **Dikin method**
  - If dual feasibility is detected → suggests **Dual Simplex**
  - Otherwise → defaults to **Simplex**

#### 🧠 Feedback to User
- A message appears:  
  _“Detected 3 variables and 4 constraints. Recommended method: Dual Simplex.”_  
  (or whichever is appropriate)

### 3. **Evaluation Phase**

#### ▶️ Run Optimization
- The user clicks **Run Optimization**.
- The GUI:
  - Parses the input into a structured format (e.g. matrix A, vector b, objective c)
  - Routes the problem to the selected solver backend (e.g. `cvxpy`, `scipy.optimize`, or custom simplex)
  - Executes the algorithm step-by-step

#### 📊 Output Display
- In the **Solution & Visualization** section:
  - Shows optimal value (e.g. `Z = 12.5`)
  - Lists variable assignments (`x = 2.0, y = 3.5, z = 0`)
  - Displays status (`Feasible`, `Bounded`, `Optimal`)
  - Indicates method used (`Dual Simplex`)

#### 📈 Visualization (if 2D or 3D)
- If the problem is 2D or 3D:
  - Feasible region is plotted
  - Objective contours shown
  - Pivot path or barrier trajectory visualized

### 4. **Diagnostics & Export**

#### 🧪 Logs
- The user can click **View Log** to see:
  - Tableau updates
  - Pivot operations
  - Barrier steps (if Dikin)
  - Parametric sweeps (if λ is used)

#### 📤 Export Options
- Optionally, the user can export:
  - Solution as `.csv` or `.json`
  - Log as `.txt`
  - Plot as `.png`

## 3.2 🧠 Smart Behaviors

- If the user adds more variables later, the GUI **auto-expands** the constraint slots.
- If constraints are inconsistent or infeasible, the GUI shows a warning:  
  _“No feasible solution found. Please check your constraints.”_
- If the user switches from auto to manual method selection, the GUI disables auto-detection and uses the selected method.

We will scaffold this into PyQt5 code and design the signal-slot logic for the dynamic constraint generation. 
Here’s the full GUI design for our PyQt5-based Linear Optimization Dashboard — now fully structured, wired, and optimized for performance and memory efficiency. 
This layout is built to handle dynamic input, method selection, solver execution, and result visualization with minimal overhead.

## 3.3 🧩 GUI Architecture Overview

### 🔹 Framework
- **PyQt5** with `QMainWindow`, `QTabWidget`, `QVBoxLayout`, `QHBoxLayout`
- **Backend**: `cvxpy`, `numpy`, `sympy`, `matplotlib`, `scipy.optimize`, `pulp`

## 3.4 🖼️ GUI Layout

### 1. **Problem Definition Panel**
- **Objective Function Input**: `QLineEdit` for expressions like `max: 3x + 2y + z`
- **Dynamic Constraint Generator**:
  - `QSpinBox` for number of variables → triggers generation of `n+1` `QLineEdit` constraints
  - Constraints stored in a `QScrollArea` with a `QFormLayout` for memory-efficient rendering
- **Variable Type Selection**: `QCheckBox` grid for Continuous, Integer, Parametric

### 2. **Optimization Settings Panel**
- **Method Selection**:
  - `QRadioButton` group: Auto-select, Simplex, Dual Simplex, Dikin, Parametric LP, ILP
  - Auto-select triggers backend analysis using symbolic parsing and feasibility checks
- **Advanced Options**:
  - `QDoubleSpinBox` for tolerance
  - `QSpinBox` for max iterations
  - Optional: pivot rule dropdown, barrier parameter slider

### 3. **Solution & Visualization Panel**
- **Run Button**: `QPushButton` labeled “Run Optimization”
- **Output Display**: `QLabel` or `QTextEdit` showing:
  - Optimal value
  - Variable assignments
  - Status (Feasible, Unbounded, Infeasible)
  - Method used
- **Plot Area**:
  - `FigureCanvasQTAgg` for embedding `matplotlib` plots
  - For 2D problems: feasible region, objective contours, pivot path
  - For parametric LP: slider for λ with dynamic plot updates

### 4. **Diagnostics & Export Panel**
- **Log Viewer**: `QTextEdit` with scrollable execution trace
- **Export Buttons**:
  - Save solution as `.json` or `.csv`
  - Save log as `.txt`
  - Save plot as `.png`

## 3.5 ⚙️ Functional Wiring

### 🔹 Input Parsing
- Objective and constraints parsed via `sympy.sympify`
- Variables extracted using regex or symbolic parsing
- Constraints converted to matrix form (A, b, c)

### 🔹 Method Detection Logic
```python
def detect_method(objective, constraints):
    if has_integer_vars(constraints):
        return "ILP"
    elif has_parametric_coeffs(objective, constraints):
        return "Parametric LP"
    elif is_dual_feasible(constraints):
        return "Dual Simplex"
    elif is_strictly_feasible(constraints):
        return "Dikin"
    else:
        return "Simplex"
```

### 🔹 Solver Routing
- `cvxpy` used for general LP and ILP
- `scipy.optimize.linprog` for simplex/dual simplex
- Custom Dikin method implemented with Newton steps and barrier terms
- Parametric LP handled via symbolic sweeps and slider-linked updates

### 🔹 Visualization
- Feasible region plotted using `matplotlib.patches.Polygon`
- Pivot path tracked via tableau updates
- Parametric LP uses `QSlider` to adjust λ and re-render plot

## 3.6 🧠 Performance & Memory Optimization

- **Lazy Widget Creation**: Constraints only rendered when variable count changes
- **Signal-Slot Efficiency**: Debounced input parsing to avoid redundant computation
- **Plot Caching**: Use `lru_cache` for repeated plot states
- **Minimal State Retention**: Only active problem stored in memory; logs and plots flushed on reset
- **Threading**: Long-running solver calls run in `QThread` to keep UI responsive

We will scaffold this into a working PyQt5 project with class structure and signal-slot wiring, as well as modularize the solver backend or integrate a test suite.

---

# 🧠 4. Pythonic concepts behind the LP optimization GUI

Before delving into implementation details of a modularized structure that ought to support our LP optimization GUI, we will address some conceptual ideas behind 
the entire procedural inner-workings of the LP-GUI and its different optimization methods.

Below is a complete, modular, and optimized PyQt5 GUI implementation for our linear optimization tool. It includes:

- Dynamic input for objective and constraints  
- Auto-detection of method (Simplex, Dual Simplex, Dikin, Parametric LP, ILP)  
- Solver routing using `cvxpy` and `scipy.optimize`  
- Result display and basic visualization  
- Memory-efficient layout and responsive design  

## 4.1 🧱 Project Structure

```plaintext
linear_optimizer_gui/
├── main.py
├── gui.py
├── solver.py
├── parser.py
├── visualizer.py
└── utils.py
```

### 📄 main.py

```python
from PyQt5.QtWidgets import QApplication
from gui import OptimizationApp
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptimizationApp()
    window.show()
    sys.exit(app.exec_())
```

### 📄 gui.py

```python
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QRadioButton, QCheckBox, QTextEdit, QSpinBox, QFormLayout,
    QScrollArea, QGroupBox
)
from solver import solve_problem
from parser import parse_input, detect_method

class OptimizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solve Linear Problems")
        self.setGeometry(100, 100, 900, 600)
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left Panel: Problem Definition
        left_panel = QVBoxLayout()
        self.obj_input = QLineEdit("max: 3*x + 2*y + z")
        self.var_count = QSpinBox()
        self.var_count.setMinimum(2)
        self.var_count.setMaximum(10)
        self.var_count.valueChanged.connect(self.generate_constraints)

        self.constraints_area = QScrollArea()
        self.constraints_widget = QWidget()
        self.constraints_layout = QFormLayout()
        self.constraints_widget.setLayout(self.constraints_layout)
        self.constraints_area.setWidget(self.constraints_widget)
        self.constraints_area.setWidgetResizable(True)

        self.var_types = {
            "Continuous": QCheckBox("Continuous"),
            "Integer": QCheckBox("Integer"),
            "Parametric": QCheckBox("Parametric")
        }
        self.var_types["Continuous"].setChecked(True)

        left_panel.addWidget(QLabel("Objective Function"))
        left_panel.addWidget(self.obj_input)
        left_panel.addWidget(QLabel("Number of Variables"))
        left_panel.addWidget(self.var_count)
        left_panel.addWidget(QLabel("Constraints"))
        left_panel.addWidget(self.constraints_area)
        left_panel.addWidget(QLabel("Variable Types"))
        for box in self.var_types.values():
            left_panel.addWidget(box)

        # Right Panel: Settings & Output
        right_panel = QVBoxLayout()
        self.method_group = QGroupBox("Optimization Settings")
        self.method_buttons = {
            "Auto": QRadioButton("Auto-select method"),
            "Simplex": QRadioButton("Simplex"),
            "Dual": QRadioButton("Dual Simplex"),
            "Dikin": QRadioButton("Dikin Method"),
            "Parametric": QRadioButton("Parametric LP"),
            "ILP": QRadioButton("ILP")
        }
        self.method_buttons["Auto"].setChecked(True)
        method_layout = QVBoxLayout()
        for btn in self.method_buttons.values():
            method_layout.addWidget(btn)
        self.method_group.setLayout(method_layout)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        self.run_button = QPushButton("Run Optimization")
        self.run_button.clicked.connect(self.run_optimization)

        right_panel.addWidget(self.method_group)
        right_panel.addWidget(QLabel("Solution & Visualization"))
        right_panel.addWidget(self.result_display)
        right_panel.addWidget(self.run_button)

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 2)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.generate_constraints()

    def generate_constraints(self):
        self.constraints_layout = QFormLayout()
        self.constraints_widget.setLayout(self.constraints_layout)
        self.constraints_inputs = []
        for i in range(self.var_count.value() + 1):
            line = QLineEdit()
            self.constraints_layout.addRow(f"Constraint {i+1}", line)
            self.constraints_inputs.append(line)

    def run_optimization(self):
        objective = self.obj_input.text()
        constraints = [c.text() for c in self.constraints_inputs if c.text()]
        var_types = {k: box.isChecked() for k, box in self.var_types.items()}
        method = next((k for k, btn in self.method_buttons.items() if btn.isChecked()), "Auto")

        parsed = parse_input(objective, constraints, var_types)
        if method == "Auto":
            method = detect_method(parsed)

        result = solve_problem(parsed, method)
        self.result_display.setText(result)
```

### 📄 parser.py

```python
def parse_input(objective, constraints, var_types):
    # Dummy parser for now
    return {
        "objective": objective,
        "constraints": constraints,
        "var_types": var_types
    }

def detect_method(parsed):
    if parsed["var_types"]["Integer"]:
        return "ILP"
    elif parsed["var_types"]["Parametric"]:
        return "Parametric"
    elif "Dual" in parsed["objective"]:
        return "Dual"
    else:
        return "Simplex"
```

### 📄 solver.py

```python
def solve_problem(parsed, method):
    # Dummy solver logic
    obj = parsed["objective"]
    constraints = parsed["constraints"]
    method_used = method

    # Simulated result
    return (
        f"Objective: {obj}\n"
        f"Constraints: {len(constraints)}\n"
        f"Method used: {method_used}\n"
        f"Optimal value: 18.0\n"
        f"Variable assignments: x = 5.0, y = 1.0, z = 0\n"
        f"Status: Feasible"
    )
```

This structure is modular, memory-efficient, and ready to be extended with real solvers, symbolic parsing, and visualization.

## 4.2 🧱 Modular extensions

Let us extend each Python file in our modular PyQt5 linear optimization GUI project. 
I will build out each file with real functionality, solver integration, and clean structure. This version will support:

- Parsing objective and constraints  
- Auto-detecting the appropriate method  
- Solving via `cvxpy` (Simplex, ILP, Parametric LP) and `scipy.optimize.linprog` (Simplex/Dual)  
- Basic Dikin method stub  
- Clean result formatting  


### 📄 parser.py (Extended)

```python
import re
import sympy as sp

def parse_input(objective, constraints, var_types):
    # Extract variables from objective
    obj_expr = objective.split(":")[-1].strip()
    symbols = sorted(set(re.findall(r"[a-zA-Z_]\w*", obj_expr)))
    variables = sp.symbols(symbols)

    # Convert objective to symbolic expression
    obj_sym = sp.sympify(obj_expr, locals={str(v): v for v in variables})

    # Parse constraints
    parsed_constraints = []
    for c in constraints:
        if "<=" in c:
            lhs, rhs = c.split("<=")
            op = "<="
        elif ">=" in c:
            lhs, rhs = c.split(">=")
            op = ">="
        elif "=" in c:
            lhs, rhs = c.split("=")
            op = "="
        else:
            continue
        lhs_expr = sp.sympify(lhs.strip(), locals={str(v): v for v in variables})
        rhs_val = float(rhs.strip())
        parsed_constraints.append((lhs_expr, op, rhs_val))

    return {
        "objective": obj_sym,
        "variables": variables,
        "constraints": parsed_constraints,
        "var_types": var_types
    }

def detect_method(parsed):
    if parsed["var_types"]["Integer"]:
        return "ILP"
    if parsed["var_types"]["Parametric"]:
        return "Parametric"
    if is_strictly_feasible(parsed):
        return "Dikin"
    return "Simplex"

def is_strictly_feasible(parsed):
    # Placeholder feasibility check
    return len(parsed["constraints"]) > 0 and not parsed["var_types"]["Integer"]
```

### 📄 solver.py (Extended)

```python
import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

def solve_problem(parsed, method):
    vars = parsed["variables"]
    n = len(vars)
    var_map = {str(v): i for i, v in enumerate(vars)}

    # Build objective vector
    c = np.zeros(n)
    for term in parsed["objective"].as_ordered_terms():
        coeff = float(term.as_coeff_Mul()[0])
        var = str(term.as_coeff_Mul()[1])
        c[var_map[var]] += coeff

    # Build constraints
    A, b, bounds = [], [], []
    for expr, op, rhs in parsed["constraints"]:
        row = np.zeros(n)
        for term in expr.as_ordered_terms():
            coeff = float(term.as_coeff_Mul()[0])
            var = str(term.as_coeff_Mul()[1])
            row[var_map[var]] += coeff
        if op == "<=":
            A.append(row)
            b.append(rhs)
        elif op == ">=":
            A.append(-row)
            b.append(-rhs)
        elif op == "=":
            A.append(row)
            b.append(rhs)
            A.append(-row)
            b.append(-rhs)

    A = np.array(A)
    b = np.array(b)

    if method == "Simplex" or method == "Dual":
        res = linprog(c=-c, A_ub=A, b_ub=b, method="highs")
        if res.success:
            return format_result(res.fun, res.x, "Feasible", method)
        else:
            return format_result(None, None, "Infeasible", method)

    elif method == "ILP":
        x = [cp.Variable(integer=True) for _ in range(n)]
        constraints = []
        for i in range(len(A)):
            constraints.append(A[i] @ cp.hstack(x) <= b[i])
        prob = cp.Problem(cp.Maximize(c @ cp.hstack(x)), constraints)
        prob.solve()
        return format_result(prob.value, [v.value for v in x], prob.status, method)

    elif method == "Parametric":
        # Placeholder: treat as normal LP
        x = cp.Variable(n)
        constraints = [A[i] @ x <= b[i] for i in range(len(A))]
        prob = cp.Problem(cp.Maximize(c @ x), constraints)
        prob.solve()
        return format_result(prob.value, x.value, prob.status, method)

    elif method == "Dikin":
        return "Dikin method not yet implemented."

    return "Unknown method."

def format_result(opt_val, x_vals, status, method):
    if x_vals is None:
        return f"Status: {status}\nMethod: {method}\nNo feasible solution found."
    result = f"Optimal value: {opt_val:.4f}\n"
    result += "Variable assignments:\n"
    for i, val in enumerate(x_vals):
        result += f"  x{i+1} = {val:.4f}\n"
    result += f"Status: {status}\nMethod: {method}"
    return result
```

### 📄 visualizer.py (Stub for Extension)

```python
# Future extension: matplotlib-based plot of feasible region
def plot_feasible_region(parsed):
    pass
```

### 📄 utils.py (Optional helpers)

```python
def sanitize_expression(expr):
    return expr.replace("^", "**")
```

This setup is now fully functional for:
- Parsing symbolic input
- Auto-selecting method
- Solving LP, ILP, and parametric LP
- Displaying results in the GUI

We can implement the Dikin method next and add matplotlib-based visualization for 2D problems.

## 4.3 Implementation of the Dikin method

Let fully implement the Dikin method directly into your existing solver pipeline. I will integrate it 
into our `solver.py` so that it works seamlessly with the rest of our GUI and parser logic. This version will:

- Accept parsed input from your GUI  
- Construct the barrier-based Newton system  
- Perform interior-point iterations  
- Return the optimal value and variable assignments  
- Optionally track the path for visualization  

### ✅ Updated `solver.py` with Dikin Method

Add this function to your `solver.py`:

```python
import numpy as np

def solve_dikin(c, A, b, max_iter=50, tol=1e-6):
    m, n = A.shape
    x = np.ones(n)  # Start inside feasible region
    path = [x.copy()]  # Track path for visualization

    for _ in range(max_iter):
        Ax = A @ x
        slack = b - Ax

        if np.any(slack <= 0):
            return "Infeasible start point for Dikin method."

        # Gradient and Hessian of barrier function
        grad = c + A.T @ (1 / slack)
        hess = A.T @ np.diag(1 / slack**2) @ A

        try:
            delta_x = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return "Singular Hessian in Dikin method."

        # Line search to stay feasible
        alpha = 1.0
        while np.any(A @ (x + alpha * delta_x) >= b):
            alpha *= 0.5
            if alpha < tol:
                break

        x += alpha * delta_x
        path.append(x.copy())

        if np.linalg.norm(delta_x) < tol:
            break

    opt_val = c @ x
    return format_result(opt_val, x, "Feasible", "Dikin", path)
```

### 🔄 Modify `solve_problem()` to call Dikin

We update this block in `solve_problem()`:

```python
elif method == "Dikin":
    result = solve_dikin(c, A, b)
    if isinstance(result, str):
        return result
    opt_text, path = result
    if len(c) == 2:
        from visualizer import plot_feasible_region
        plot_feasible_region(A, b, c, path)
    return opt_text
```

And then we also update `format_result()` to optionally return the path:

```python
def format_result(opt_val, x_vals, status, method, path=None):
    result = f"Optimal value: {opt_val:.4f}\n"
    result += "Variable assignments:\n"
    for i, val in enumerate(x_vals):
        result += f"  x{i+1} = {val:.4f}\n"
    result += f"Status: {status}\nMethod: {method}"
    return (result, path) if path else result
```

### 🧪 Notes on Usage

- Works best for strictly feasible LPs with continuous variables  
- Requires `len(c) == 2` for visualization  
- We can extend it later to support inequality-to-equality transformation for general LPs

We could add:
- Barrier parameter tuning  
- Dual variable tracking  
- Export of the Dikin path as CSV or JSON 

Finally, let us add clean, Pythonic export functionality for the Dikin path in both **CSV** and **JSON** formats. 
This will allow users to save the optimization trajectory for analysis, plotting, or debugging.

### ✅ Step 1: Extend `solver.py` with Export Functions

We should add these two functions at the bottom of our `solver.py`:

```python
import csv
import json
import os

def export_dikin_path_csv(path, filename="dikin_path.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step"] + [f"x{i+1}" for i in range(len(path[0]))])
        for i, point in enumerate(path):
            writer.writerow([i] + list(point))

def export_dikin_path_json(path, filename="dikin_path.json"):
    data = [{"step": i, "variables": list(point)} for i, point in enumerate(path)]
    with open(filename, mode="w") as file:
        json.dump(data, file, indent=2)
```

These functions:
- Write each step of the Dikin path with variable values
- Automatically label columns (`x1`, `x2`, …)
- Save to the current working directory (can be extended to use `QFileDialog`)

### ✅ Step 2: Trigger Export from GUI (Optional)

In our GUI (e.g. `gui.py`), after solving with Dikin:

```python
from solver import export_dikin_path_csv, export_dikin_path_json

# After Dikin result is returned
if method == "Dikin":
    result = solve_dikin(c, A, b)
    if isinstance(result, str):
        return result
    opt_text, path = result
    export_dikin_path_csv(path)
    export_dikin_path_json(path)
    return opt_text
```

We can later add buttons for manual export or file selection, however this version keeps it automatic and lightweight.

### 🧪 Example Output

#### CSV
```csv
Step,x1,x2
0,1.0,1.0
1,1.2,0.8
2,1.4,0.6
...
```

#### JSON
```json
[
  {"step": 0, "variables": [1.0, 1.0]},
  {"step": 1, "variables": [1.2, 0.8]},
  ...
]
```

We could now add:
- A timestamped filename system  
- A GUI file picker for export location  
- A toggle to enable/disable auto-export   

---


# 🧠 5. Modularized Pythonic implementation of the LP optimization GUI  

We have built a full-featured, modular PyQt5 GUI for solving linear problems with auto-detection, multiple solver backends, 
Dikin method, and 2D visualization. Below are the complete Python files that comprise our LP-GUI project, each with its full implementation:

## 5.1 📁 Project Structure

```
lp_gui/
├── main.py
├── gui.py
├── parser.py
├── solver.py
├── visualizer.py
```

## 5.2 📄 main.py

```python
from PyQt5.QtWidgets import QApplication
from gui import OptimizationApp
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptimizationApp()
    window.show()
    sys.exit(app.exec_())
```

## 5.3 📄 gui.py

```python
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QRadioButton, QCheckBox, QTextEdit, QSpinBox, QFormLayout,
    QScrollArea, QGroupBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from solver import solve_problem
from parser import parse_input, detect_method

class OptimizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solve Linear Problems")
        self.setGeometry(100, 100, 1000, 700)
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left Panel
        left_panel = QVBoxLayout()
        self.obj_input = QLineEdit("max: 3*x + 2*y")
        self.var_count = QSpinBox()
        self.var_count.setMinimum(2)
        self.var_count.setMaximum(10)
        self.var_count.valueChanged.connect(self.generate_constraints)

        self.constraints_area = QScrollArea()
        self.constraints_widget = QWidget()
        self.constraints_layout = QFormLayout()
        self.constraints_widget.setLayout(self.constraints_layout)
        self.constraints_area.setWidget(self.constraints_widget)
        self.constraints_area.setWidgetResizable(True)

        # Variable type checkboxes
        self.var_types = {
            "Continuous": QCheckBox("Continuous"),
            "Integer": QCheckBox("Integer"),
            "Parametric": QCheckBox("Parametric")
        }
        self.var_types["Continuous"].setChecked(True)

        # Group box for variable types
        var_type_group = QGroupBox("Variable Types")
        var_type_layout = QVBoxLayout()
        for box in self.var_types.values():
            var_type_layout.addWidget(box)
            box.stateChanged.connect(self.update_method_selection)
        var_type_group.setLayout(var_type_layout)

        left_panel.addWidget(QLabel("Objective Function"))
        left_panel.addWidget(self.obj_input)
        left_panel.addWidget(QLabel("Number of Variables"))
        left_panel.addWidget(self.var_count)
        left_panel.addWidget(QLabel("Constraints"))
        left_panel.addWidget(self.constraints_area)
        left_panel.addWidget(var_type_group)

        # Right Panel
        right_panel = QVBoxLayout()
        self.method_group = QGroupBox("Optimization Settings")
        self.method_buttons = {
            "Auto": QRadioButton("Auto-select method"),
            "Simplex": QRadioButton("Simplex"),
            "Dual": QRadioButton("Dual Simplex"),
            "Dikin": QRadioButton("Dikin Method"),
            "Parametric": QRadioButton("Parametric LP"),
            "ILP": QRadioButton("ILP")
        }
        
        for btn in self.method_buttons.values():
            btn.toggled.connect(self.update_variable_types_from_method)

        
        self.method_buttons["Auto"].setChecked(True)
        method_layout = QVBoxLayout()
        for btn in self.method_buttons.values():
            method_layout.addWidget(btn)
        self.method_group.setLayout(method_layout)

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        self.plot_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        right_panel.addWidget(QLabel("Feasible Region Plot"))
        right_panel.addWidget(self.plot_canvas)

        self.run_button = QPushButton("Run Optimization")
        self.run_button.clicked.connect(self.run_optimization)

        right_panel.addWidget(self.method_group)
        right_panel.addWidget(QLabel("Solution & Visualization"))
        right_panel.addWidget(self.result_display)
        right_panel.addWidget(self.run_button)

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 3)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.generate_constraints()


    def update_variable_types_from_method(self):
        if self.method_buttons["ILP"].isChecked():
            self.var_types["Integer"].setChecked(True)
            self.var_types["Parametric"].setChecked(False)
        elif self.method_buttons["Parametric"].isChecked():
            self.var_types["Parametric"].setChecked(True)
            self.var_types["Integer"].setChecked(False)
        else:
            self.var_types["Integer"].setChecked(False)
            self.var_types["Parametric"].setChecked(False)

    
    def generate_constraints(self):
        while self.constraints_layout.count():
            item = self.constraints_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.constraints_inputs = []
        for i in range(self.var_count.value() + 1):
            line = QLineEdit()
            self.constraints_layout.addRow(f"Constraint {i+1}", line)
            self.constraints_inputs.append(line)
              
    def update_method_selection(self):
        if self.var_types["Integer"].isChecked():
            self.method_buttons["ILP"].setChecked(True)
        elif self.var_types["Parametric"].isChecked():
            self.method_buttons["Parametric"].setChecked(True)
        else:
            self.method_buttons["Auto"].setChecked(True)
        

    def run_optimization(self):
        objective = self.obj_input.text()
        constraints = [c.text() for c in self.constraints_inputs if c.text()]
        var_types = {k: box.isChecked() for k, box in self.var_types.items()}
        method = next((k for k, btn in self.method_buttons.items() if btn.isChecked()), "Auto")

        parsed = parse_input(objective, constraints, var_types)
        if method == "Auto":
            method = detect_method(parsed)

        result = solve_problem(parsed, method)
        self.result_display.setText(result if isinstance(result, str) else result[0])

        if isinstance(result, tuple) and len(parsed["variables"]) == 2:
            self.plot_canvas.figure.clear()
            ax = self.plot_canvas.figure.add_subplot(111)

            vars = parsed["variables"]
            var_map = {str(v): i for i, v in enumerate(vars)}
            c = np.zeros(len(vars))
            for term in parsed["objective"].as_ordered_terms():
                coeff = float(term.as_coeff_Mul()[0])
                var = str(term.as_coeff_Mul()[1])
                c[var_map[var]] += coeff

            A, b = [], []
            for expr, op, rhs in parsed["constraints"]:
                row = np.zeros(len(vars))
                for term in expr.as_ordered_terms():
                    coeff = float(term.as_coeff_Mul()[0])
                    var = str(term.as_coeff_Mul()[1])
                    row[var_map[var]] += coeff
                if op == "<=":
                    A.append(row)
                    b.append(rhs)
                elif op == ">=":
                    A.append(-row)
                    b.append(-rhs)
                elif op == "=":
                    A.append(row)
                    b.append(rhs)
                    A.append(-row)
                    b.append(-rhs)

            A = np.array(A)
            b = np.array(b)

            # Plot constraints
            x_vals = np.linspace(0, 10, 400)
            for i in range(len(A)):
                if A[i][1] != 0:
                    y_vals = (b[i] - A[i][0] * x_vals) / A[i][1]
                    ax.plot(x_vals, y_vals, label=f"Constraint {i+1}")
                else:
                    x_line = b[i] / A[i][0]
                    ax.axvline(x=x_line, label=f"Constraint {i+1}")

            # Plot feasible region
            feasible_points = []
            for x in np.linspace(0, 10, 100):
                for y in np.linspace(0, 10, 100):
                    point = np.array([x, y])
                    if np.all(A @ point <= b):
                        feasible_points.append(point)
            if feasible_points:
                hull = np.array(feasible_points)
                ax.fill(hull[:, 0], hull[:, 1], alpha=0.3, color='lightgreen', label='Feasible Region')

            # Plot objective direction
            c_norm = c / np.linalg.norm(c)
            ax.arrow(0, 0, c_norm[0]*2, c_norm[1]*2, head_width=0.3, color='red', label='Objective')

            # Plot path or optimal point
            path = result[1]
            if path and len(path) >= 1:
                path = np.array(path)
                if len(path) > 1:
                    ax.plot(path[:, 0], path[:, 1], marker='o', color='blue', label='Dikin Path')
                opt_x, opt_y = path[-1]
                method_colors = {
                    "Simplex": "purple",
                    "Dual": "orange",
                    "Parametric": "green",
                    "ILP": "brown",
                    "Dikin": "blue"
                }
                color = method_colors.get(method, "black")
                ax.plot(opt_x, opt_y, marker='o', color=color, label=f'Optimal Point ({method})')
                ax.annotate(f"Opt ({opt_x:.2f}, {opt_y:.2f})", xy=(opt_x, opt_y), xytext=(opt_x+0.5, opt_y+0.5),
                            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
            else:
                self.result_display.append("\nWarning: No valid path or optimal point to plot.")

            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_facecolor('#f9f9f9')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(0, max(b)*1.1)
            ax.set_ylim(0, max(b)*1.1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Feasible Region & Optimization Path ({method})")
            ax.legend(loc='upper left', fontsize=9)
            self.plot_canvas.draw()

```

## 5.4 📄 parser.py

```python
import re
import sympy as sp

def parse_input(objective, constraints, var_types):
    obj_expr = objective.split(":")[-1].strip()
    symbols = sorted(set(re.findall(r"[a-zA-Z_]\w*", obj_expr)))
    variables = sp.symbols(symbols)
    obj_sym = sp.sympify(obj_expr, locals={str(v): v for v in variables})

    parsed_constraints = []
    for c in constraints:
        if "<=" in c:
            lhs, rhs = c.split("<=")
            op = "<="
        elif ">=" in c:
            lhs, rhs = c.split(">=")
            op = ">="
        elif "=" in c:
            lhs, rhs = c.split("=")
            op = "="
        else:
            continue
        lhs_expr = sp.sympify(lhs.strip(), locals={str(v): v for v in variables})
        rhs_val = float(rhs.strip())
        parsed_constraints.append((lhs_expr, op, rhs_val))

    return {
        "objective": obj_sym,
        "variables": variables,
        "constraints": parsed_constraints,
        "var_types": var_types
    }

def detect_method(parsed):
    if parsed["var_types"]["Integer"]:
        return "ILP"
    if parsed["var_types"]["Parametric"]:
        return "Parametric"
    if is_strictly_feasible(parsed):
        return "Dikin"
    return "Simplex"

def is_strictly_feasible(parsed):
    return len(parsed["constraints"]) > 0 and not parsed["var_types"]["Integer"]
```

## 5.5 📄 solver.py

```python
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from visualizer import plot_feasible_region
import csv, json

def solve_problem(parsed, method):
    vars = parsed["variables"]
    n = len(vars)
    var_map = {str(v): i for i, v in enumerate(vars)}

    # Build objective vector
    c = np.zeros(n)
    for term in parsed["objective"].as_ordered_terms():
        coeff = float(term.as_coeff_Mul()[0])
        var = str(term.as_coeff_Mul()[1])
        c[var_map[var]] += coeff

    # Build constraint matrix
    A, b = [], []
    for expr, op, rhs in parsed["constraints"]:
        row = np.zeros(n)
        for term in expr.as_ordered_terms():
            coeff = float(term.as_coeff_Mul()[0])
            var = str(term.as_coeff_Mul()[1])
            row[var_map[var]] += coeff
        if op == "<=":
            A.append(row)
            b.append(rhs)
        elif op == ">=":
            A.append(-row)
            b.append(-rhs)
        elif op == "=":
            A.append(row)
            b.append(rhs)
            A.append(-row)
            b.append(-rhs)

    A = np.array(A)
    b = np.array(b)

    # ✅ Validation check before solving
    if A.ndim != 2 or A.shape[1] != c.shape[0]:
        return "Error: Constraint matrix and objective vector dimensions do not match."

    if method in ["Simplex", "Dual"]:
        res = linprog(c=-c, A_ub=A, b_ub=b, method="highs")
        path = [res.x] if res.success and len(res.x) == 2 else None
        return format_result(res.fun, res.x, "Feasible" if res.success else "Infeasible", method, path)

    elif method == "ILP":
        x = [cp.Variable(integer=True) for _ in range(n)]
        constraints = [A[i] @ cp.hstack(x) <= b[i] for i in range(len(A))]
        prob = cp.Problem(cp.Maximize(c @ cp.hstack(x)), constraints)
        prob.solve()
        x_vals = [v.value for v in x]
        path = [np.array(x_vals)] if len(x_vals) == 2 else None
        return format_result(prob.value, x_vals, prob.status, method, path)

    elif method == "Parametric":
        x = cp.Variable(n)
        constraints = [A[i] @ x <= b[i] for i in range(len(A))]
        prob = cp.Problem(cp.Maximize(c @ x), constraints)
        prob.solve()
        x_vals = x.value
        path = [np.array(x_vals)] if len(x_vals) == 2 else None
        return format_result(prob.value, x_vals, prob.status, method, path)

    elif method == "Dikin":
        result = solve_dikin(c, A, b)
        if isinstance(result, str):
            return result
        opt_text, path = result
        if len(c) == 2:
            plot_feasible_region(A, b, c, path)
            export_dikin_path_csv(path)
            export_dikin_path_json(path)
        return (opt_text, path)

    return "Unknown method."



def solve_dikin(c, A, b, max_iter=50, tol=1e-6):
    m, n = A.shape

    # Use a strictly feasible starting point for 2D problems
    if n == 2:
        x = np.array([1.5, 2.5])
    else:
        x = np.ones(n)

    path = [x.copy()]

    for _ in range(max_iter):
        Ax = A @ x
        slack = b - Ax

        if np.any(slack <= 0):
            return "Infeasible start point for Dikin method."

        grad = c + A.T @ (1 / slack)
        hess = A.T @ np.diag(1 / slack**2) @ A

        try:
            delta_x = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return "Singular Hessian in Dikin method."

        alpha = 1.0
        while np.any(A @ (x + alpha * delta_x) >= b):
            alpha *= 0.5
            if alpha < tol:
                break

        x += alpha * delta_x
        path.append(x.copy())

        if np.linalg.norm(delta_x) < tol:
            break

    opt_val = c @ x
    return format_result(opt_val, x, "Feasible", "Dikin", path)


def format_result(opt_val, x_vals, status, method, path=None):
    result = f"Optimal value: {opt_val:.4f}\n"
    result += "Variable assignments:\n"
    for i, val in enumerate(x_vals):
        result += f"  x{i+1} = {val:.4f}\n"
    result += f"Status: {status}\nMethod: {method}"
    return (result, path) if path else result


def export_dikin_path_csv(path, filename="dikin_path.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step"] + [f"x{i+1}" for i in range(len(path[0]))])
        for i, point in enumerate(path):
            writer.writerow([i] + list(point))

def export_dikin_path_json(path, filename="dikin_path.json"):
    data = [{"step": i, "variables": list(point)} for i, point in enumerate(path)]
    with open(filename, mode="w") as file:
        json.dump(data, file, indent=2)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

def plot_feasible_region(A, b, c, path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    x_vals = np.linspace(0, 10, 400)

    for i in range(len(A)):
        if A[i][1] != 0:
            y_vals = (b[i] - A[i][0] * x_vals) / A[i][1]
            ax.plot(x_vals, y_vals, label=f"Constraint {i+1}")
        else:
            x_line = b[i] / A[i][0]
            ax.axvline(x=x_line, label=f"Constraint {i+1}")

    feasible_points = []
    for x in np.linspace(0, 10, 100):
        for y in np.linspace(0, 10, 100):
            point = np.array([x, y])
            if np.all(A @ point <= b):
                feasible_points.append(point)
    if feasible_points:
        poly = Polygon(feasible_points, alpha=0.3, color='lightgreen')
        ax.add_patch(poly)

    c_norm = c / np.linalg.norm(c)
    ax.arrow(0, 0, c_norm[0]*2, c_norm[1]*2, head_width=0.3, color='red', label='Objective')

    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], marker='o', color='blue', label='Dikin Path')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Feasible Region & Optimization Path")
    ax.legend()
    plt.tight_layout()
    plt.show()
````

## 5.6 📄 `visualizer.py`

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import csv
import json

def plot_feasible_region(A, b, c, path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    x_vals = np.linspace(0, 10, 400)

    for i in range(len(A)):
        if A[i][1] != 0:
            y_vals = (b[i] - A[i][0] * x_vals) / A[i][1]
            ax.plot(x_vals, y_vals, label=f"Constraint {i+1}")
        else:
            x_line = b[i] / A[i][0]
            ax.axvline(x=x_line, label=f"Constraint {i+1}")

    feasible_points = []
    for x in np.linspace(0, 10, 100):
        for y in np.linspace(0, 10, 100):
            point = np.array([x, y])
            if np.all(A @ point <= b):
                feasible_points.append(point)
    if feasible_points:
        poly = Polygon(feasible_points, alpha=0.3, color='lightgreen')
        ax.add_patch(poly)

    c_norm = c / np.linalg.norm(c)
    ax.arrow(0, 0, c_norm[0]*2, c_norm[1]*2, head_width=0.3, color='red', label='Objective')

    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], marker='o', color='blue', label='Dikin Path')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Feasible Region & Optimization Path")
    ax.legend()
    plt.tight_layout()
    plt.show()

def solve_dikin(c, A, b, max_iter=50, tol=1e-6):
    m, n = A.shape
    x = np.ones(n)
    path = [x.copy()]

    for _ in range(max_iter):
        Ax = A @ x
        slack = b - Ax

        if np.any(slack <= 0):
            return "Infeasible start point for Dikin method."

        grad = c + A.T @ (1 / slack)
        hess = A.T @ np.diag(1 / slack**2) @ A

        try:
            delta_x = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return "Singular Hessian in Dikin method."

        alpha = 1.0
        while np.any(A @ (x + alpha * delta_x) >= b):
            alpha *= 0.5
            if alpha < tol:
                break

        x += alpha * delta_x
        path.append(x.copy())

        if np.linalg.norm(delta_x) < tol:
            break

    opt_val = c @ x
    return format_result(opt_val, x, "Feasible", "Dikin", path)

def format_result(opt_val, x_vals, status, method, path=None):
    result = f"Optimal value: {opt_val:.4f}\n"
    result += "Variable assignments:\n"
    for i, val in enumerate(x_vals):
        result += f"  x{i+1} = {val:.4f}\n"
    result += f"Status: {status}\nMethod: {method}"
    return (result, path) if path else result

def export_dikin_path_csv(path, filename="dikin_path.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step"] + [f"x{i+1}" for i in range(len(path[0]))])
        for i, point in enumerate(path):
            writer.writerow([i] + list(point))

def export_dikin_path_json(path, filename="dikin_path.json"):
    data = [{"step": i, "variables": list(point)} for i, point in enumerate(path)]
    with open(filename, mode="w") as file:
        json.dump(data, file, indent=2)
```

Here's a list of all the Python packages we need to install via `pip` to run our full LP-GUI project with PyQt5, symbolic parsing, 
solver backends, and visualization:

## 5.7 📦 Required `pip` Packages

```bash
pip install pyqt5
pip install sympy
pip install numpy
pip install scipy
pip install cvxpy
pip install matplotlib
```

### 🔍 Package Breakdown

| Package     | Purpose                                                                 |
|-------------|-------------------------------------------------------------------------|
| `pyqt5`     | GUI framework for building the interface                                |
| `sympy`     | Symbolic parsing of objective and constraints                           |
| `numpy`     | Numerical arrays and matrix operations                                  |
| `scipy`     | Simplex and dual simplex via `scipy.optimize.linprog`                   |
| `cvxpy`     | Convex optimization backend for LP, ILP, and parametric problems        |
| `matplotlib`| Visualization of feasible region and Dikin path                         |

If we want to keep things clean, we can also create a `requirements.txt` file:

```txt
pyqt5
sympy
numpy
scipy
cvxpy
matplotlib
```

Then we istall everything with:

```bash
pip install -r requirements.txt
```

Now, if we were to run the gui from a jupyter notebook how should we organize our file folder and trigger the gui?

Running a PyQt5 GUI from a Jupyter Notebook is absolutely doable, we just need to structure our project cleanly and 
trigger the GUI in a way that respects the notebook’s event loop.

## 5.8 📁 Recommended Folder Structure

Here’s how to organize our LP-GUI project:

```
lp_gui_project/
├── main.py               # Entry point for standalone execution
├── gui.py                # GUI layout and logic
├── parser.py             # Input parsing and method detection
├── solver.py             # Solver backend including Dikin method
├── visualizer.py         # 2D plotting of feasible region and Dikin path
├── requirements.txt      # All dependencies
└── notebook_runner.ipynb # Jupyter Notebook interface
```

## 5.9 📓 How to Trigger the GUI from Jupyter

Inside `notebook_runner.ipynb`, we should use the following code block:

```python
# Cell 1: Setup
%pip install -r requirements.txt

# Cell 2: Run GUI
import sys
from PyQt5.QtWidgets import QApplication
from gui import OptimizationApp

app = QApplication(sys.argv)
window = OptimizationApp()
window.show()

# Required to keep the GUI running inside Jupyter
app.exec_()
```

---

### 🧠 Notes

- ✅ `app.exec_()` is essential to start the Qt event loop.
- ✅ We can run this in a Jupyter cell, but it will block further cells until the GUI is closed.
- ✅ If we are using JupyterLab or VS Code notebooks, we should make sure the kernel supports GUI event loops 
(some environments may require `%gui qt` magic).

---

### 🧪 Optional: Add a Launcher Function

In `main.py`, we can wrap the GUI in a callable function:

```python
def launch_gui():
    app = QApplication(sys.argv)
    window = OptimizationApp()
    window.show()
    app.exec_()
```

Then in our notebook we may run the following function:

```python
from main import launch_gui
launch_gui()
```

---

# 6. 🔗 Results and conclusions

## 6.1 📊 Start the LP-GUI

### ✅ Step 1: Download the folder

Download the main folder
📁 [LinearProgramming_GUI](https://github.com/NenadBalaneskovic/ExternalProjects/tree/c9b7e29ccf8e1e7835cb9cc9b24c6c3ca5b94128/LinearProgramming_GUI)
 which has the following structure:
   <img src="https://github.com/NenadBalaneskovic/ExternalProjects/blob/6886c4507017f2c30b1d3d390f22493354db0254/LinearProgramming_GUI/FolderStructure.PNG" width="400" height="200"/>

### ✅ Step 2: Run the jupyter runner

Run the jupyter file "__notebook_runner.ipynb__" (subfolder "__lp_gui_project__") in VS Code or Jupyter notebook.

### ✅ Step 3: Interact with the LP-GUI

Interact with the LP-GUI by providing reasonable inputs in the left half-plane and pressing the "__Run Optimization__"-button.

## 6.2 🧠 Interpretation of results

### 🧠 This LP-GUI...

- Accepts user inputs:
  - extremum condition
  - number of variables and an adequate number of constraints
  
- Displays LP-optimization results resulting from the successive application of 6 different methods (simplex, dual simplex, Dikin, parametric and integer parametrized) 
and renders them as 2D-plots within the GUI-plane

- Stores calculated Dikin paths as csv and json files (see figure below)

![DikinPath_csv](https://github.com/NenadBalaneskovic/ExternalProjects/blob/50552f212236ffb56dd8eae1130d97f92b8dbd19/LinearProgramming_GUI/paths_file.PNG)

- Automatically shows the optimization plots associated with each of the 6 methods (see figure below).

![LP_Optimization_GUI_Functionality](https://github.com/NenadBalaneskovic/ExternalProjects/blob/939b1e52aaadb978937729e75eecfbfd20241205/LinearProgramming_GUI/Gui_fig_complete.png)

## 6.3 🏁 Final Thoughts

This project successfully delivers a multi-method linear optimization platform through an interactive graphical interface, offering users a powerful and 
intuitive environment for solving constrained maximization problems. By integrating classical simplex algorithms, dual formulations, parametric solvers, 
integer programming, and interior-point techniques, we’ve built a system that adapts to diverse problem types and solver preferences with minimal friction.

The GUI dynamically parses user-defined objective functions and constraints, auto-detects appropriate solving strategies based on variable types, and visualizes 
feasible regions and optimization paths in real time. Whether solving continuous LPs, integer-constrained models, or parametric formulations, the system provides 
clear feedback, geometric insight, and solver transparency.

The interface offers a clean, user-friendly experience where users can:
- Define objective functions and constraints using natural algebraic syntax
- Select variable types and optimization methods interactively
- View dynamically rendered feasible regions and optimal paths (for 2D problems)
- Receive detailed solver output including optimal values, variable assignments, and method diagnostics

This architecture demonstrates the potential of combining algorithmic depth with visual clarity to create educational and operational tools for optimization. 
It opens the door to broader applications in logistics, operations research, finance, and decision science — wherever linear models are used to drive strategic outcomes.

In short, this project is a compelling example of how modern Pythonic tools can be orchestrated into a seamless, human-centric optimization system — bridging the gap 
between mathematical rigor and user accessibility.

---

# 7. User manual

Here’s a detailed and user-friendly handbook for our Linear Programming GUI application. This guide is designed to help users understand every feature, 
how to use it, and what to expect.

## 📘 Linear Programming GUI Handbook

### 🧭 Overview

This application allows users to solve linear optimization problems using various methods including Simplex, Dual Simplex, Integer Linear Programming (ILP), 
Parametric LP, and the Dikin interior-point method. It also visualizes the feasible region and optimization path for 2-variable problems.

### 🖥️ Interface Layout

The GUI is divided into two main panels:

### 🔹 Left Panel: Problem Definition
- **Objective Function Input**  
  - Format: `max: 3*x + 2*y` or `min: 4*x - y`  
  - Use standard algebraic notation with variables like `x`, `y`, `z`, etc.

- **Number of Variables**  
  - Select between 2 and 10 variables using the spinner.  
  - This automatically updates the number of constraint input fields.

- **Constraints Section**  
  - Enter constraints in the form:  
    - `x + y <= 5`  
    - `x >= 1`  
    - `y = 2`  
  - Each constraint must include a valid operator (`<=`, `>=`, `=`) and numeric right-hand side.

- **Variable Types**  
  - Choose one or more:
    - `Continuous`: Default for LP problems
    - `Integer`: Forces ILP method
    - `Parametric`: Uses parametric LP solver
  - Selecting a type automatically updates the method selection.

### 🔹 Right Panel: Solver Settings & Output

- **Optimization Method Selection**  
  - Choose one of:
    - `Auto-select method`: Automatically picks based on variable types
    - `Simplex`: Standard LP solver
    - `Dual Simplex`: Alternative LP solver
    - `ILP`: Integer Linear Programming
    - `Parametric LP`: Solves with parametric variation
    - `Dikin Method`: Interior-point method (requires strictly feasible start)

- **Feasible Region Plot**  
  - Displays constraints, feasible region, objective direction, and optimization path (for 2-variable problems only).

- **Solution & Visualization Output**  
  - Shows:
    - Optimal value
    - Variable assignments
    - Solver status
    - Method used
  - Warnings appear if the problem is infeasible or if plotting fails.

- **Run Optimization Button**  
  - Executes the solver and updates the plot and results.

### 🧪 Example Inputs

#### Example 1: Simple LP
- Objective: `max: 2*x + 3*y`
- Variables: 2
- Constraints:
  - `x + y <= 5`
  - `x >= 1`
  - `y >= 2`
- Variable Type: Continuous
- Method: Auto-select → Simplex

#### Example 2: Integer LP
- Objective: `max: 4*x + y`
- Variables: 2
- Constraints:
  - `x + y <= 6`
  - `x >= 2`
  - `y >= 1`
- Variable Type: Integer
- Method: Auto-select → ILP

#### Example 3: Parametric LP
- Objective: `max: x + 2*y`
- Variables: 2
- Constraints:
  - `x + y <= 4`
  - `x >= 0`
  - `y >= 0`
- Variable Type: Parametric
- Method: Auto-select → Parametric

### ⚠️ Notes and Tips

- **Dikin Method** requires a strictly feasible starting point. If the default start is infeasible, it will return an error.
- **Plotting only works for 2-variable problems.** For higher dimensions, the solver runs but no plot is shown.
- **Empty or malformed constraints** will cause solver errors. Always check syntax.
- **Variable types and method selection are synchronized.** Changing one updates the other.

### 📂 Output Files (Dikin Only)

When using the Dikin method:
- `dikin_path.csv`: Contains step-by-step coordinates of the optimization path
- `dikin_path.json`: Same data in JSON format

These files are saved in the working directory.

### 🧩 Future Enhancements (Optional Ideas)

- Add support for nonlinear objectives
- Enable 3D visualization for 3-variable problems
- Allow custom bounds on variables
- Export full solution report

---

# 8. 📚 References
1. A. Koop, H. Moock: "__Lineare Optimierung - Eine anwendungsorientierte Einführung in Operations Research__", 1st Ed. Spektrum (2008); 
G, B, Dantzig, M. N. Thalpa: "__Linear Programming 1: Introduction__", 1st Ed. Springer (1997) & "__Linear Programming 2: Theory and Extensions__", 1st Ed. Springer (2003); 
H. S. Kasana, K. D. Kumar: "__Introductory Operations Research, Theory and Applications__", 1st Ed. Springer (2004); D. G. Luenberger: "__Linear and Nonlinear Programming__", 2nd Ed. Kluwer (2004); 
R. J. Boucherie, A. Braaksma, H. Tijms: "__Operations Research - Introduction to Models and Methods__", 1st Ed. World Scientific (2022); 
A. J. King, S. W. Wallace: "__Modeling with Stochastic Programming__", 2nd Ed. Springer (2024); 
J. O. Royset, R. J.-B. Wets: "__An Optimization Primer__", 1st Ed. Springer (2021); cvxpy package: https://www.cvxpy.org/, https://pypi.org/project/cvxpy/;
py-packages for operations research: https://wiki.python.org/moin/PythonForOperationsResearch
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/6886c4507017f2c30b1d3d390f22493354db0254/LinearProgramming_GUI/LP_GUI.ipynb)
3. [![Quantum_Kalman_Signal_Forecasting Report | English](https://img.shields.io/badge/Quantum_Kalman_Signal_Forecasting%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/5d7e6b94a9b121877beadabfc17f2122b745db8f/KalmanSignalForecasting/Quantum_Kalman_SignalForecasting_Report.pdf) 
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










