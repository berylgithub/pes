import numpy as np
import psi4
import time

h2_geometry = """
H
--
H 1 {0}
"""

Rvals = [2.5, 3.0]

psi4.set_options({'freeze_core': 'true'})

# Initialize a blank dictionary of counterpoise corrected energies
# (Need this for the syntax below to work)

data = np.zeros((len(Rvals), 2))
start = time.time()

for i, R in enumerate(Rvals):
    mol = psi4.geometry(h2_geometry.format(R))
    data[i][0] = R
    data[i][1] = psi4.energy('ccsd(t)/aug-cc-pvdz', bsse_type='cp', molecule=mol)

print(data)
np.save("data/abinitio_test", data)

'''
# Prints to screen
print("CP-corrected CCSD(T)/aug-cc-pVDZ Interaction Energies\n\n")
print("          R [Ang]                 E_int [kcal/mol]       ")
print("---------------------------------------------------------")
for R in Rvals:
    e = ecp[R] * psi4.constants.hartree2kcalmol
    print("            {:3.1f}                        {:1.6f}".format(R, e))

# Prints to output.dat
psi4.core.print_out("CP-corrected CCSD(T)/aug-cc-pVDZ Interaction Energies\n\n")
psi4.core.print_out("          R [Ang]                 E_int [kcal/mol]       \n")
psi4.core.print_out("---------------------------------------------------------\n")
for R in Rvals:
    e = ecp[R] * psi4.constants.hartree2kcalmol
    psi4.core.print_out("            {:3.1f}                        {:1.6f}\n".format(R, e))


'''
end  = time.time()

print("runtime = ",end-start)