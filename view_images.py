import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

folder1 = '/home/knoll/LDAData/21-02-18_CT_water_only/phantom_scan/CT'
folder2 = '/home/knoll/LDAData/21-02-19_CT_min_vol/phantom_scan/CT'
folder3 = '/home/knoll/LDAData/21-02-19_CT_min_Gd/phantom_scan/CT'

# data_c = loadmat(os.path.join(folder1, 'CT.mat'))['ct_img']
# data_s = np.load(os.path.join(folder1, 'FDK_CT_no_filt.npy'))
# data_f = np.load(os.path.join(folder1, 'FDK_CT.npy'))

bin_num = 2

# for i in range(7, 15):
#
#     fig, ax = plt.subplots(1, 3, figsize=(12, 6))
#     ax[0].imshow(data_f[bin_num, i], cmap='gray', vmin=0, vmax=0.08)
#     # ax[0].set_title('FDK')
#     ax[1].imshow(data_s[bin_num, i], cmap='gray', vmin=0, vmax=0.008)
#     # ax[1].set_title('SIRT')
#     ax[2].imshow(data_c[bin_num, :, :, i], cmap='gray', vmin=0, vmax=0.08)
#     # ax[2].set_title('CGLS')
#     plt.show()
#     plt.pause(10)
#     plt.close()

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
