import os

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import load

# Load the .npz file
pwd = os.getcwd()
print(pwd)
data = load('data/sq_fid_data.npz')
# print(data)
# print(data['fid_99'])
# print(data['fid_100'])
# print(data['f_1qb'])
# print(data['f_2qb'])
# # Access arrays stored inside
# # Assuming 'array1' is one of the arrays stored in the file
# array1 = data['array1']
f_1 = data['f_1qb']
f_2 = data['f_2qb']
fid_arr_99 = data['fid_99']
fid_arr_100 = data['fid_100']
# dpi = 600, I 
plt.figure(figsize=(len(f_2), len(f_1)), dpi = 600) 
#labels = b
#ax = sns.heatmap(fid_arr, cmap="rocket", norm = colors.LogNorm(), annot=labels, annot_kws={'fontsize': 10}, fmt='.3f', cbar_kws={'label': 'Energy'})
#ax = sns.heatmap(fid_arr, cmap="rocket", cbar_kws={'label': 'Circ_ fid'}, vmin = 0.30, vmax = 1)
# ax = sns.heatmap(fid_arr_100-fid_arr_99, cmap="rocket", cbar_kws={'label': 'Circ_ fid'})
ax = sns.heatmap(fid_arr_100-fid_arr_99, cmap="rocket", cbar_kws={'label': 'Circ_ fid'}, annot=True, fmt=".2f", annot_kws={"color": "white"})

ax.set(xticklabels=f_2)
ax.set(yticklabels=f_1)
ax.invert_yaxis()
#plt.xticks(ticks = p_2)
#ax.yticks(p_1)
plt.title(r'$Circuit~fidelity~difference: H_2, \Delta F_{meas}$')
plt.xlabel(r'$ F_{2qb} $')
plt.ylabel(r'$ F_{1qb} $')
plt.savefig("SQ_circ_fid_H2_f100-f99.pdf")
# plt.show()