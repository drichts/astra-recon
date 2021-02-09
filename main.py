import astra
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

pixel_pitch = 1  # [mm]
source_origin = 322.0 * 3  # [mm]
origin_det = 256.0 * 3  # [mm]

raw_data = np.load(r'/home/knoll/LDAData/data_corr.npy')[:, :, :, 6]

num_bins = 7
det_col_count = 576
det_row_count = 24
num_angles = 720

raw_data = np.transpose(raw_data, axes=(1, 0, 2))  # Transpose to (rows, angles, columns)

angles = np.linspace(0, 2*np.pi, num_angles, False)

# Create a 3D projection geometry with our cone-beam data
# Parameters: 'acquisition type', number of detector rows, number of detector columns, data ndarray
proj_geom = astra.create_proj_geom('cone', pixel_pitch, pixel_pitch, det_row_count, det_col_count, angles,
                                   source_origin, origin_det)
proj_id = astra.data3d.create('-proj3d', proj_geom, raw_data)

# Create a 3D volume geometry.
# Parameter order: rows, columns, slices (y, x, z)
vol_geom = astra.create_vol_geom(det_col_count, det_col_count, det_row_count)

# Create a data object for the reconstruction
recon_id = astra.data3d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ReconstructionDataId'] = recon_id
cfg['ProjectionDataId'] = proj_id
# cfg['option'] = {'FilterType': 'Hamming'}

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run 150 iterations of the algorithm
# Note that this requires about 750MB of GPU memory, and has a runtime
# in the order of 10 seconds.
astra.algorithm.run(alg_id, 150)

# Get the result
rec = astra.data3d.get(recon_id)
print(np.shape(rec))
plt.figure(figsize=(8, 8))
plt.imshow(rec[15, :, :], vmin=0, vmax=0.006)
plt.show(block=True)

np.save(r'/home/knoll/LDAData/recon_SIRT.npy', rec)

# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data3d.delete(recon_id)
astra.data3d.delete(proj_id)

