import sympy as sp

# Define symbolic variables
kx, ky = sp.symbols('kx ky')
a = 0.142e-9  # Lattice constant in meters
t = 3  # Energy scale factor in electron volts

# Define the energy expression for the conduction band (+t) and valence band (-t)
E = t * sp.sqrt(1 + 4 * sp.cos(sp.sqrt(3) * ky * a / 2) * sp.cos(3 * kx * a / 2) +
                 4 * sp.cos(sp.sqrt(3) * ky * a / 2)**2)

# Compute the second partial derivatives with respect to kx and ky
d2Edkx2 = E.diff(kx, 2)
d2Edky2 = E.diff(ky, 2)

# Laplacian of the energy expression
laplacian_E = d2Edkx2 + d2Edky2

# Output the laplacian
print("Laplacian of the energy expression:")
print(laplacian_E)
