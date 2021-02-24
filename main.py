import astra
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime


pixel_pitch = 1  # [mm]
source_origin = 322.0 * 3  # [mm]
origin_det = 256.0 * 3  # [mm]

det_col_count = 576
det_row_count = 24
num_angles = 720
angles = np.linspace(0, 2 * np.pi, num_angles, False)

bins = [0, 1, 2, 3, 4, 5, 6]


def run_full_recon(folder, alg, iterations, bins=bins):

    # Array for the reconstructed data
    ct_img = np.zeros([len(bins), det_row_count, det_col_count, det_col_count])

    # Reconstruct all the bins
    start = datetime.now().timestamp()

    if alg != 'FDK_CUDA':
        ct_data = np.load(os.path.join(directory, folder, 'CT', 'FDK_CT.npy'))

    raw_data_full = np.load(os.path.join(directory, folder, 'Data', 'data_corr_no_filt.npy'))
    raw_data_full = np.transpose(raw_data_full, axes=(3, 1, 0, 2))  # Transpose to (bins, rows, angles, columns)

    # Change if isocentre is not directly in the center of the detector
    # raw_data_full = np.roll(raw_data_full, -2, axis=3)
    
    for bin_num in bins:
        
        # Get the right bin number
        raw_data = raw_data_full[bin_num]
        
        # Create a 3D projection geometry with our cone-beam data
        # Parameters: 'acquisition type', number of detector rows, number of detector columns, data ndarray
        proj_geom = astra.create_proj_geom('cone', pixel_pitch, pixel_pitch, det_row_count, det_col_count, angles,
                                           source_origin, origin_det)
        proj_id = astra.data3d.create('-proj3d', proj_geom, raw_data)

        # Create a 3D volume geometry.
        # Parameter order: rows, columns, slices (y, x, z)
        vol_geom = astra.create_vol_geom(det_col_count, det_col_count, det_row_count)

        # Create a data object for the reconstruction
        if alg == 'FDK_CUDA':
            recon_id = astra.data3d.create('-vol', vol_geom)
        else:
            recon_id = astra.data3d.create('-vol', vol_geom, data=ct_data[bin_num])

        # Set up the parameters for a reconstruction algorithm using the GPU
        cfg = astra.astra_dict(alg)
        cfg['ReconstructionDataId'] = recon_id
        cfg['ProjectionDataId'] = proj_id

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)

        # Run the desired algorithm
        if alg == 'FDK_CUDA':
            astra.algorithm.run(alg_id)
        else:
            astra.algorithm.run(alg_id, iterations)

        # Get the result
        rec = astra.data3d.get(recon_id)
        ct_img[bin_num] = rec

        # Clean up. Note that GPU memory is tied up in the algorithm object,
        # and main RAM in the data objects.
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(recon_id)
        astra.data3d.delete(proj_id)

        # Show the resulting image
        plt.figure(figsize=(8, 8))
        plt.imshow(rec[14], vmin=0, vmax=0.08)
        plt.show()
        plt.savefig(os.path.join(directory, folder, 'fig', f'{alg[0:4]}_bin{bin_num}_nofilt.png'))
        plt.close()
    stop = datetime.now().timestamp()
    print(f'Recon time: {stop-start:.2f} s')
    np.save(os.path.join(directory, folder, 'CT', alg[0:4] + 'CT_no_filt.npy'), ct_img)


def check_recon(folder, alg, iterations, bin_num):

    raw_data = np.load(os.path.join(directory, folder, 'Data', 'data_corr.npy'))[:, :, :, 6]
    raw_data = np.transpose(raw_data, axes=(1, 0, 2))  # Transpose to (rows, angles, columns)

    # Change if isocentre is not directly in the center of the detector
    # raw_data = np.roll(raw_data, -2, axis=2)

    # Create a 3D projection geometry with our cone-beam data
    # Parameters: 'acquisition type', number of detector rows, number of detector columns, data ndarray
    proj_geom = astra.create_proj_geom('cone', pixel_pitch, pixel_pitch, det_row_count, det_col_count, angles,
                                       source_origin, origin_det)
    proj_id = astra.data3d.create('-proj3d', proj_geom, raw_data)

    # Create a 3D volume geometry.
    # Parameter order: rows, columns, slices (y, x, z)
    vol_geom = astra.create_vol_geom(det_col_count, det_col_count, det_row_count)

    # Create a data object for the reconstruction
    if alg == 'FDK_CUDA':
        recon_id = astra.data3d.create('-vol', vol_geom)
    else:
        ct_data = np.load(os.path.join(directory, folder, 'CT', 'FDK_CT.npy'))
        recon_id = astra.data3d.create('-vol', vol_geom, data=ct_data)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict(alg)
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = proj_id

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run the specified number of iterations of the algorithm
    rec_data = np.zeros((iterations // 5, 576, 576))
    for i in range(0, iterations, 5):
        astra.algorithm.run(alg_id, 5)

        # Get the result
        rec = astra.data3d.get(recon_id)[11]
        print(alg[0:4] + f': {i + 5} Iterations')
        rec_data[i // 5] = rec

    np.save(os.path.join(directory, folder, 'CT', alg[0:4] + '_iteration_check.npy'), rec_data)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(recon_id)
    astra.data3d.delete(proj_id)


if __name__ == '__main__':
    directory = '/home/knoll/LDAData/'
    folder = '21-02-18_CT_water_only/phantom_scan'
    alg = 'FDK_CUDA'  # Algorithms: SIRT3D_CUDA, CGLS3D_CUDA, FDK_CUDA
    iterations = 100

    # Create save folder if necessary
    save_folder = os.path.join(directory, folder, 'CT')
    fig_folder = os.path.join(directory, folder, 'fig')
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(fig_folder, exist_ok=True)

    # Run the full recon
    run_full_recon(folder, alg, iterations)

    # Run a recon to check the right number of iterations
    # check_recon(folder, alg, iterations)
