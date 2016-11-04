import matplotlib as mpl
import sys
import numpy as np
import ca_source_extraction as cse
from scipy.sparse import coo_matrix
from time import time
import pylab as pl
import psutil
from glob import glob
import os
import scipy
from ipyparallel import Client
import json
from neurofinder import load, centers, shapes,match

mpl.use('TKAgg')

# The path of all datasets and parameters
params = [
    ['/Users/songyang/Documents/CSCI8360/Project4/testData/neurofinder.00.00.test', 7, False, False, False, 7, 4],
    ['/Users/songyang/Documents/CSCI8360/Project4/testData/neurofinder.00.01.test', 7, False, False, False, 7, 4],
    ['/Users/songyang/Documents/CSCI8360/Project4/testData/neurofinder.01.00.test', 7.5, False, False, False, 7, 4],
    ['/Users/songyang/Documents/CSCI8360/Project4/testData/neurofinder.01.01.test', 7.5, False, False, False, 7, 4],
    ['/Users/songyang/Documents/CSCI8360/Project4/testData/neurofinder.02.01.test', 8, False, False, False, 6, 4],
    ['/Users/songyang/Documents/CSCI8360/Project4/testData/neurofinder.02.00.test', 8, False, False, False, 6, 4],
    ['/Users/songyang/Documents/CSCI8360/Project4/testData/neurofinder.03.00.test', 7.5, False, False, False, 7, 4],
    ['/Users/songyang/Documents/CSCI8360/Project4/testData/neurofinder.04.00.test', 6.75, False, False, False, 6, 4],
    ['/Users/songyang/Documents/CSCI8360/Project4/testData/neurofinder.04.01.test', 3, False, False, False, 6, 4],
]

f_rates = np.array([el[1] for el in params])
base_folders = [el[0] for el in params] # folders
# all false
do_rotate_template = np.array([el[2] for el in params])
do_self_motion_correct = np.array([el[3] for el in params])
do_motion_correct = np.array([el[4] for el in params])

gsigs = np.array([el[5] for el in params])
Ks = np.array([el[6] for el in params])

backend = 'local'
if backend == 'SLURM':
    n_processes = np.int(os.environ.get('SLURM_NPROCS'))
else:
    n_processes = np.maximum(np.int(psutil.cpu_count()), 1)  # roughly number of cores on your machine minus 1
print 'using ' + str(n_processes) + ' processes'
single_thread = False

if single_thread:
    dview = None
else:
    try:
        c.close()
    except:
        print 'C was not existing, creating one'
    print "Stopping  cluster to avoid unnencessary use of memory...."
    sys.stdout.flush()
    if backend == 'SLURM':
        try:
            cse.utilities.stop_server(is_slurm=True)
        except:
            print 'Nothing to stop'
        slurm_script = '/mnt/xfs1/home/agiovann/SOFTWARE/Constrained_NMF/SLURM/slurmStart.sh'
        cse.utilities.start_server(slurm_script=slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        c = Client(ipython_dir=pdir, profile=profile)
    else:
        cse.utilities.stop_server()
        cse.utilities.start_server()
        c = Client()

    print 'Using ' + str(len(c)) + ' processes'
    dview = c[:len(c)]

final_frate = 3
# ----- Options ----- #
load_results = False
save_results = False
save_mmap = True

for testIdx in xrange(len(params)):
    print("!!!!!!!!!!!!!!!START!!!!!!!!!!!!")
    for folder_in, f_r, gsig, K in zip(base_folders[testIdx:(testIdx+1)], f_rates[testIdx:(testIdx+1)], gsigs[testIdx:(testIdx+1)], Ks[testIdx:(testIdx+1)]):
        # LOAD MOVIE HERE USE YOUR METHOD, Movie is frames x dim2 x dim2
        movie_name = os.path.join(folder_in, 'images', 'images_all.tif')
        if save_mmap:
            downsample_factor = final_frate / f_r
            base_name = 'Yr'
            name_new = cse.utilities.save_memmap_each([movie_name], dview=None, base_name=base_name,
                                                      resize_fact=(1, 1, downsample_factor), remove_init=0, idx_xy=None)
            print name_new
            fname_new = cse.utilities.save_memmap_join(name_new, base_name='Yr', n_chunks=6, dview=dview)
        else:
            fname_new = glob(os.path.join(folder_in, 'images', 'Yr_*.mmap'))[0]

        Yr, dims, T = cse.utilities.load_memmap(fname_new)
        d1, d2 = dims
        Y = np.reshape(Yr, dims + (T,), order='F')
        if load_results:
            with np.load(os.path.join(folder_in, 'images', 'results_analysis_patch_3.npz')) as ld:
                locals().update(ld)
            gSig = [gsig, gsig]
            merge_thresh = 0.8  # merging threshold, max correlation allowed
            p = 2  # order of the autoregressive system
            memory_fact = 1;  # unitless number a

        else:
            # %
            Cn = cse.utilities.local_correlations(Y[:, :, :3000])
            # %%
            rf = 15  # half-size of the patches in pixels. rf=25, patches are 50x50
            stride = 2  # amounpl.it of overlap between the patches in pixels
            #        K=K # number of neurons expected per patch
            gSig = [gsig, gsig]  # expected half size of neurons
            merge_thresh = 0.8  # merging threshold, max correlation allowed
            p = 2  # order of the autoregressive system
            memory_fact = 0.4  
            # RUN ALGORITHM ON PATCHES
            options_patch = cse.utilities.CNMFSetParms(Y, n_processes, p=0, gSig=gSig, K=K, ssub=1, tsub=4,
                                                       thr=merge_thresh)
            A_tot, C_tot, YrA_tot, b, f, sn_tot, optional_outputs = cse.map_reduce.run_CNMF_patches(fname_new, (d1, d2, T),
                                                                                                    options_patch, rf=rf,
                                                                                                    stride=stride,
                                                                                                    dview=dview,
                                                                                                    memory_fact=memory_fact)
            print 'Number of components:' + str(A_tot.shape[-1])

            if save_results:
                np.savez(os.path.join(folder_in, 'images', 'results_analysis_patch_3.npz'), A_tot=A_tot.todense(),
                         C_tot=C_tot, sn_tot=sn_tot, d1=d1, d2=d2, b=b, f=f, Cn=Cn)

        options = cse.utilities.CNMFSetParms(Y, n_processes, p=0, gSig=gSig, K=A_tot.shape[-1], thr=merge_thresh)
        pix_proc = np.minimum(np.int((d1 * d2) / n_processes / (T / 2000.)),
                              np.int((d1 * d2) / n_processes))  # regulates the amount of memory used
        options['spatial_params']['n_pixels_per_process'] = pix_proc
        options['temporal_params']['n_pixels_per_process'] = pix_proc

        merged_ROIs = [0]
        A_m = scipy.sparse.coo_matrix(A_tot)
        C_m = C_tot
        while len(merged_ROIs) > 0:
            A_m, C_m, nr_m, merged_ROIs, S_m, bl_m, c1_m, sn_m, g_m = cse.merge_components(Yr, A_m, [], np.array(C_m), [],
                                                                                           np.array(C_m), [],
                                                                                           options['temporal_params'],
                                                                                           options['spatial_params'],
                                                                                           dview=dview,
                                                                                           thr=options['merging']['thr'],
                                                                                           mx=np.Inf)
        options['temporal_params']['p'] = 0
        options['temporal_params']['fudge_factor'] = 0.96  # change ifdenoised traces time constant is wrong
        options['temporal_params']['backend'] = 'ipyparallel'
        C_m, f_m, S_m, bl_m, c1_m, neurons_sn_m, g2_m, YrA_m = cse.temporal.update_temporal_components(Yr, A_m,
                                                                                                       np.atleast_2d(b).T,
                                                                                                       C_m, f, dview=dview,
                                                                                                       bl=None, c1=None,
                                                                                                       sn=None, g=None,
                                                                                                       **options[
                                                                                                           'temporal_params'])

        tB = np.minimum(-2, np.floor(-5. / 30 * final_frate))
        tA = np.maximum(5, np.ceil(25. / 30 * final_frate))
        Npeaks = 10
        traces = C_m + YrA_m

        fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = cse.utilities.evaluate_components(
            Y, traces, A_m, C_m, b, f, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB,
            tA=tA, thresh_C=0.3)

        idx_components_r = np.where(r_values >= .4)[0]
        idx_components_raw = np.where(fitness_raw < -20)[0]
        idx_components_delta = np.where(fitness_delta < -10)[0]

        idx_components = np.union1d(idx_components_r, idx_components_raw)
        idx_components = np.union1d(idx_components, idx_components_delta)
        idx_components_bad = np.setdiff1d(range(len(traces)), idx_components)

        A_m = A_m[:, idx_components]
        C_m = C_m[idx_components, :]

        print 'Number of components:' + str(A_m.shape[-1])

        t1 = time()
        A2, b2, C2 = cse.spatial.update_spatial_components(Yr, C_m, f, A_m, sn=sn_tot, dview=dview,
                                                           **options['spatial_params'])
        print time() - t1

        # UPDATE TEMPORAL COMPONENTS
        options['temporal_params']['p'] = 0
        options['temporal_params']['fudge_factor'] = 0.96  # change ifdenoised traces time constant is wrong
        C2, f2, S2, bl2, c12, neurons_sn2, g21, YrA = cse.temporal.update_temporal_components(Yr, A2, b2, C2, f,
                                                                                              dview=dview, bl=None, c1=None,
                                                                                              sn=None, g=None,
                                                                                              **options['temporal_params'])
        # MERGE AGAIN
        merged_ROIs2 = [0]
        A_m = A2
        C_m = C2
        while len(merged_ROIs2) > 0:
            A2, C2, nr2, merged_ROIs2, S2, bl_2, c1_2, sn_2, g_2 = cse.merge_components(Yr, A2, b2, np.array(C2), f2,
                                                                                        np.array(S2), [],
                                                                                        options['temporal_params'],
                                                                                        options['spatial_params'],
                                                                                        dview=dview,
                                                                                        thr=options['merging']['thr'],
                                                                                        mx=np.Inf)
        options['temporal_params']['p'] = p
        options['temporal_params']['fudge_factor'] = 0.96  # change if denoised traces time constant is wrong
        C2, f2, S2, bl2, c12, neurons_sn2, g21, YrA = cse.temporal.update_temporal_components(Yr, A2, b2, C2, f2,
                                                                                              dview=dview, bl=None, c1=None,
                                                                                              sn=None, g=None, **options[
                'temporal_params'])  # Order components

        log_files = glob('Yr*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

        traces = C2 + YrA

        fitness_raw, fitness_delta, erfc_raw, erfc_delta, r_values, significant_samples = cse.utilities.evaluate_components(
            Y, traces, A2, C2, b2, f2, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=Npeaks, tB=tB,
            tA=tA, thresh_C=0.3)

        idx_components_r = np.where(r_values >= .5)[0]
        idx_components_raw = np.where(fitness_raw < -30)[0]
        idx_components_delta = np.where(fitness_delta < -15)[0]

        idx_components = np.union1d(idx_components_r, idx_components_raw)
        idx_components = np.union1d(idx_components, idx_components_delta)
        idx_components_bad = np.setdiff1d(range(len(traces)), idx_components)

        print(len(idx_components))
        print len(traces)
        print(idx_components.size * 1. / traces.shape[0])

        if save_results:
            np.savez(os.path.join(folder_in, 'results_analysis_3.npz'), Cn=Cn, A_tot=A_tot, C_tot=C_tot, sn_tot=sn_tot,
                     A2=A2, C2=C2, b2=b2, S2=S2, f2=f2, bl2=bl2, c12=c12, neurons_sn2=neurons_sn2, g21=g21, YrA=YrA, d1=d1,
                     d2=d2,
                     fitness_raw=fitness_raw, fitness_delta=fitness_delta, erfc_raw=erfc_raw, erfc_delta=erfc_delta,
                     r_values=r_values, significant_samples=significant_samples)

        min_radius = gSig[0] - 2
        masks_ws, pos_examples, neg_examples = cse.utilities.extract_binary_masks_blob(
            A2.tocsc()[:, :], min_radius, dims, num_std_threshold=1,
            minCircularity=0.5, minInertiaRatio=0.2, minConvexity=.8)

        np.savez(os.path.join(os.path.split(fname_new)[0], 'regions_CNMF_3.npz'), masks_ws=masks_ws,
                 pos_examples=pos_examples, neg_examples=neg_examples, idx_components=idx_components)
        final_masks = np.array(masks_ws)[np.intersect1d(idx_components, pos_examples)]

        regions_CNMF = cse.utilities.nf_masks_to_json(final_masks,
                                                      os.path.join(folder_in, 'regions_CNMF_3.json'))
    print("!!!!!!!!!!!!!!!END!!!!!!!!!!!!")

