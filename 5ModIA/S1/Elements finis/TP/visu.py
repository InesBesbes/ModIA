import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

with open('errors.txt', 'r') as f:
    lines = f.readlines()

np_values = []
errorL2_values = []
errorH1_values = []

for line in lines:
    values = line.strip().split()
    np_values.append(int(values[0]))
    errorL2_values.append(float(values[1]))
    errorH1_values.append(float(values[2]))


###################################################

log_np = np.log(np_values)
log_L2 = np.log(errorL2_values)
log_H1 = np.log(errorH1_values)

slope_L2, intercept_L2, r_value_L2, pvalue_L2, std_err_L2 = linregress(log_np, log_L2)
slope_H1, intercept_H1, r_value_H1, pvalue_H1, std_err_H1 = linregress(log_np, log_H1)

plt.figure(figsize=(10, 6))

print("Pente pour L2 :" , slope_L2)
print("Pente pour H1 :" , slope_H1)

plt.loglog(np_values, errorL2_values, label='Erreur L2', marker='o')
plt.loglog(np_values, errorH1_values, label='Erreur H1', marker='o')

plt.xlabel('np')
plt.ylabel('Erreur')
plt.title('Erreurs en fonction de np')
plt.legend()
plt.grid(True, which='both', linestyle='--')

plt.show()