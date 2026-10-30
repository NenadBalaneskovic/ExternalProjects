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
