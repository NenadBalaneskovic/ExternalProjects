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