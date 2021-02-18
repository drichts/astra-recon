import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

folder = '/home/knoll/LDAData/CT_02_03_21_v3/CT'

data_c = np.load(os.path.join(folder, 'CGLSCT.npy'))
data_s = np.load(os.path.join(folder, 'SIRTCT.npy'))
data_f = np.load(os.path.join(folder, 'FDK_CT.npy'))

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].imshow(data_f[11], cmap='gray', vmin=0, vmax=0.08)
ax[0].set_title('FDK')
ax[1].imshow(data_s[11], cmap='gray', vmin=0, vmax=0.08)
ax[1].set_title('SIRT')
ax[2].imshow(data_c[11], cmap='gray', vmin=0, vmax=0.08)
ax[2].set_title('CGLS')
plt.show()

## Iteration check
# data2 = np.load(os.path.join(folder, 'CGLS_iteration_check.npy'))
# # data2 = np.load(os.path.join(folder, 'recon_SIRT_v2.npy'))
#
# for i in range(0, len(data2), 5):
#
#     fig = plt.figure(figsize=(10, 10))
#     plt.imshow(data2[i], cmap='gray', vmin=0, vmax=0.08)
#     plt.title(f'{(i+1)*5} iterations')
#     plt.show()
#     plt.savefig(os.path.join('/home/knoll/LDAData/CT_02_03_21_v3/fig', f'CGLS_{(i+1)*5}.png'), dpi=fig.dpi)
#     # plt.pause(1)
#     plt.close()
