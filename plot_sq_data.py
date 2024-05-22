# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:32:09 2024

@author: cqtv201
"""

#plot sns heatmap

# figsize=(6, 6) control width and height
# dpi = 600, I 
plt.figure(figsize=(len(f_2), len(f_1)), dpi = 600) 
#labels = b
#ax = sns.heatmap(fid_arr, cmap="rocket", norm = colors.LogNorm(), annot=labels, annot_kws={'fontsize': 10}, fmt='.3f', cbar_kws={'label': 'Energy'})
#ax = sns.heatmap(fid_arr, cmap="rocket", cbar_kws={'label': 'Circ_ fid'}, vmin = 0.30, vmax = 1)
ax = sns.heatmap(fid_arr_100-fid_arr_99, cmap="rocket", cbar_kws={'label': 'Circ_ fid'})
ax.set(xticklabels=f_2)
ax.set(yticklabels=f_1)
ax.invert_yaxis()
#plt.xticks(ticks = p_2)
#ax.yticks(p_1)
plt.title(r'$Circuit~fidelity~difference: H_2, \Delta F_{meas}$')
plt.xlabel(r'$ F_{2qb} $')
plt.ylabel(r'$ F_{1qb} $')
#plt.savefig("SQ_circ_fid_H2_f100-f99.pdf")
plt.show()