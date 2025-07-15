import numpy as np

def hatch(x):
    """
    
    Args:
        x: A numpy array of two elements, where x[0] is x1 (flange thickness) and x[1] is x2 (beam height).
    Returns:
        A list of two objective function values [f1, f2].
    """
    x1 = x[0]
    x2 = x[1]
    E = 700000  # kg/cm²

   
    sigma_b = 4500 / (x1 * x2)
    tau = 1800 / x2
    delta = (56.2 * 10**4) / (E * x1 * x2**2)
    sigma_k = (E * x1**2) / 100

    g1 = 1.0 - (sigma_b / 700)
    g2 = 1.0 - (tau / 450)
    g3 = 1.0 - (delta / 1.5)
    g4 = 1.0 - (sigma_b / sigma_k)

 
    violations = [
        max(g1, 0),
        max(g2, 0),
        max(g3, 0),
        max(g4, 0)
    ]
    f2 = sum(violations)

   
    f1 = x1 + 120 * x2

    return [f1, f2]