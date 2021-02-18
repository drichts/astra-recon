import astra
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def run_recon(folder, alg, iterations, check):

    pixel_pitch = 1  # [mm]
    source_origin = 322.0 * 3  # [mm]
    origin_det = 256.0 * 3  # [mm]

    raw_data = np.load(os.path.join(directory, folder, 'Data', 'data_corr.npy'))
    ct_data = np.load(os.path.join(directory, folder, 'CT', 'FDK_CT.npy'))

    num_bins = 1
    det_col_count = 576
    det_row_count = 24
    num_angles = 720

    raw_data = np.transpose(raw_data, axes=(1, 0, 2))  # Transpose to (rows, angles, columns)
    raw_data = np.roll(raw_data, -2, axis=2)

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
    recon_id = astra.data3d.create('-vol', vol_geom, data=ct_data)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict(alg)
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = proj_id

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run 150 iterations of the algorithm
    # Note that this requires about 750MB of GPU memory, and has a runtime
    # in the order of 10 seconds.

    if check:
        rec_data = np.zeros((iterations//5, 576, 576))
        for i in range(0, iterations, 5):

            astra.algorithm.run(alg_id, 5)

            # Get the result
            rec = astra.data3d.get(recon_id)[11]
            print(alg[0:4] + f': {i+5} Iterations')
            rec_data[i//5] = rec
            # plt.figure(figsize=(8, 8))
            # plt.imshow(rec[15, :, :], vmin=0, vmax=0.006)
            # plt.show(block=True)
            # plt.pause(1)
            # plt.close()

        np.save(os.path.join(directory, folder, 'CT', alg[0:4] + '_iteration_check.npy'), rec_data)

    else:
        if alg == 'FDK_CUDA':
            astra.algorithm.run(alg_id)
        else:
            astra.algorithm.run(alg_id, iterations)

        # Get the result
        rec = astra.data3d.get(recon_id)
        np.save(os.path.join(directory, folder, 'CT', alg[0:4] + 'CT.npy'), rec)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(recon_id)
    astra.data3d.delete(proj_id)
    #
    # plt.figure(figsize=(8, 8))
    # plt.imshow(rec[11], vmin=0, vmax=0.08)
    # plt.show(block=True)


if __name__ == '__main__':
    directory = '/home/knoll/LDAData/'
    folder = 'CT_02_03_21_v3'
    alg = 'SIRT3D_CUDA'  # Algorithms: SIRT3D_CUDA, CGLS3D_CUDA, FDK_CUDA
    iterations = 100

    # Check how many iterations to do?
    check = False

    run_recon(folder, alg, iterations, check)

    alg = 'CGLS3D_CUDA'
    run_recon(folder, alg, iterations, check)
