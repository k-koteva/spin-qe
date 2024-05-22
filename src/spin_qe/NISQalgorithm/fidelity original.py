import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import load

# Load the .npz file
pwd = os.getcwd()
print(pwd)
data = load('data/sq_fid_data.npz')
print(data)
print(data['fid_99'])
print(data['fid_100'])
print(data['f_1qb'])
print(data['f_2qb'])
# Access arrays stored inside


f_1 = data['f_1qb']
f_2 = data['f_2qb']
fid_arr_99 = data['fid_99']
fid_arr_100 = data['fid_100']

plt.figure(figsize=(len(f_2), len(f_1)), dpi = 600) 
#labels = b
#ax = sns.heatmap(fid_arr, cmap="rocket", norm = colors.LogNorm(), annot=labels, annot_kws={'fontsize': 10}, fmt='.3f', cbar_kws={'label': 'Energy'})
#ax = sns.heatmap(fid_arr, cmap="rocket", cbar_kws={'label': 'Circ_ fid'}, vmin = 0.30, vmax = 1)
# ax = sns.heatmap(fid_arr_100-fid_arr_99, cmap="rocket", cbar_kws={'label': 'Circ_ fid'})
ax = sns.heatmap(fid_arr_100, cmap="rocket", cbar_kws={'label': 'Circ_ fid'}, annot=True, fmt=".2f", annot_kws={"color": "white"})

ax.set(xticklabels=f_2)
ax.set(yticklabels=f_1)
ax.invert_yaxis()
#plt.xticks(ticks = p_2)
#ax.yticks(p_1)
plt.title(r'$Circuit~fidelity: H_2, F_{meas} = 1$')
plt.xlabel(r'$ F_{2qb} $')
plt.ylabel(r'$ F_{1qb} $')
# plt.savefig("SQ_circ_fid_H2_f100.pdf")




def analytic_fid(fid_1q: float, fid_2q: float, fid_meas: float) -> float:
    num_1q_gates = 94
    num_2q_gates = 64
    num_meas = 4
    total_fid = (fid_1q ** num_1q_gates) * (fid_2q **
                                                num_2q_gates) * (fid_meas ** num_meas)
    return total_fid

def analytic_circuit_fid(fid_meas: float) -> np.ndarray:
    all_fids = []
    for one_qb in f_1:
        fid_list = []
        for two_qb in f_2:
            fid_list.append(analytic_fid(one_qb, two_qb, fid_meas))
        all_fids.append(fid_list)
    my_fids = np.array(all_fids)
    return my_fids

analytical_fid_99 = analytic_circuit_fid(0.99)
analytical_fid_100 = analytic_circuit_fid(1.0)

# plt.figure(figsize=(len(f_2), len(f_1)), dpi = 600) 
# ax = sns.heatmap(my_fids, cmap="rocket", cbar_kws={'label': 'Circ_ fid'}, annot=True, fmt=".2f", annot_kws={"color": "white"})

# ax.set(xticklabels=f_2)
# ax.set(yticklabels=f_1)
# ax.invert_yaxis()

# plt.title(r'$Circuit~fidelity~ analytical: H_2, F_{meas} = 1$')
# plt.xlabel(r'$ F_{2qb} $')
# plt.ylabel(r'$ F_{1qb} $')
# plt.savefig("SQ_circ_analytical_fid_H2_f100.pdf")


plt.figure(figsize=(len(f_2), len(f_1)), dpi = 600) 
ax = sns.heatmap(fid_arr_100-analytical_fid_100, cmap="rocket", cbar_kws={'label': 'Circ_ fid'}, annot=True, fmt=".2f", annot_kws={"color": "white"})

ax.set(xticklabels=f_2)
ax.set(yticklabels=f_1)
ax.invert_yaxis()

plt.title(r'$Circuit~fidelity~difference: H_2, F_{meas} = 1$')
plt.xlabel(r'$ F_{2qb} $')
plt.ylabel(r'$ F_{1qb} $')
# plt.savefig("SQ_circ_diff_fid_H2_f100.pdf")

plt.figure(figsize=(len(f_2), len(f_1)), dpi = 600) 
ax = sns.heatmap(fid_arr_99-analytical_fid_99, cmap="rocket", cbar_kws={'label': 'Circ_ fid'}, annot=True, fmt=".2f", annot_kws={"color": "white"})

ax.set(xticklabels=f_2)
ax.set(yticklabels=f_1)
ax.invert_yaxis()

plt.title(r'$Circuit~fidelity~difference: H_2, F_{meas} = 1$')
plt.xlabel(r'$ F_{2qb} $')
plt.ylabel(r'$ F_{1qb} $')
# plt.savefig("SQ_circ_diff_fid_H2_f99.pdf")

plt.figure(figsize=(len(f_2), len(f_1)), dpi = 600) 
ax = sns.heatmap(analytical_fid_99, cmap="rocket", cbar_kws={'label': 'Circ_ fid'}, annot=True, fmt=".2f", annot_kws={"color": "white"})

ax.set(xticklabels=f_2)
ax.set(yticklabels=f_1)
ax.invert_yaxis()

plt.title(r'$Circuit~fidelity~difference: H_2, F_{meas} = 0.99$')
plt.xlabel(r'$ F_{2qb} $')
plt.ylabel(r'$ F_{1qb} $')
# plt.savefig("SQ_circ_analytical_fid_H2_f99.pdf")
print('analytical_fid_100')
print(analytical_fid_100)

print('fid_arr_100')
print(fid_arr_100)

#########

import matplotlib.pyplot as plt
import numpy as np

# analytical_fid_100 = np.array([
#     [0.12349944, 0.17133198, 0.23729487, 0.32811195, 0.45294634, 0.62426587],
#     [0.13572918, 0.18829841, 0.26079338, 0.36060377, 0.4978001,  0.68608482],
#     [0.14915585, 0.20692536, 0.28659172, 0.3962756,  0.54704374, 0.75395405],
#     [0.16389522, 0.22737344, 0.31491231, 0.435435,   0.60110183, 0.82845873],
#     [0.18007412, 0.24981858, 0.34599884, 0.47841892, 0.66043953, 0.91023995],
#     [0.19783148, 0.27445354, 0.38011828, 0.52559649, 0.72556641, 1.        ]
# ])

# fid_arr_100 = np.array([
#     [0.40752424, 0.45212469, 0.51123645, 0.56055532, 0.65879455, 0.77944972],
#     [0.40166793, 0.4528639,  0.51550724, 0.58818516, 0.68620659, 0.81417714],
#     [0.40110784, 0.45391736, 0.5245664,  0.63416811, 0.73675929, 0.86724989],
#     [0.41737924, 0.47582659, 0.55014333, 0.64624639, 0.75994307, 0.90313768],
#     [0.42640839, 0.48904126, 0.56975229, 0.68547322, 0.80560998, 0.95320627],
#     [0.4577119,  0.52104298, 0.60517927, 0.70533254, 0.83612066, 1.        ]
# ])

# Ensure both arrays have the same shape
assert analytical_fid_100.shape == fid_arr_100.shape, "Arrays must have the same shape."





# Create dataset
x_coords = analytical_fid_100.flatten()
y_coords = fid_arr_100.flatten()
plt.clf()


from numpy.polynomial.polynomial import Polynomial

# Fit a polynomial of degree 3
coefs = Polynomial.fit(x_coords, y_coords, 4)


# # Scatter plot of original data
plt.scatter(x_coords, y_coords, label='Data')

# Plot the polynomial fit
x_line = np.linspace(min(x_coords), max(x_coords), num=500)
y_line = coefs(x_line)
plt.plot(x_line, y_line, color='red', label='Polynomial Fit')

plt.xlabel('Analytical FID')
plt.ylabel('FID Arr')
plt.title('Fidelity scatter with Polynomial Fit')
plt.legend()
# plt.show()


# Plotting
# plt.scatter(x_coords, y_coords)
# plt.xlabel('Analytical FID')
# plt.ylabel('FID Arr')
# plt.title('Data Points from Arrays')
plt.savefig("Fidelity scatter.pdf")
# plt.show()

# Check number of data points
num_data_points = x_coords.size
expected_data_points = analytical_fid_100.size
assert num_data_points == expected_data_points, f"Data points ({num_data_points}) do not match expected ({expected_data_points})."
