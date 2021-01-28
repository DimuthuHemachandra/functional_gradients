import nibabel as nib
import numpy as np
import load_confounds
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import signal
from load_confounds import Params36
from brainspace.utils.parcellation import reduce_by_labels
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from brainspace.gradient import GradientMaps
from sklearn.metrics.pairwise import cosine_similarity
from mapalign.embed import DiffusionMapEmbedding
import os
import glob

in_cii = '/scratch/dimuthu1/PPMI_project2/derivatives/fmriprep_20_2_1_test_syn_sdc/fmriprep/sub-3116/ses-Month12/func/sub-3116_ses-Month12_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii'
bold_file = "/scratch/dimuthu1/PPMI_project2/derivatives/fmriprep_20_2_1_test_syn_sdc/fmriprep/sub-3116/ses-Month12/func/sub-3116_ses-Month12_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii"


def make_out_dir(out_path):

	#Make subdirectories to save files
	filename = out_path
	if not os.path.exists(os.path.dirname(filename)):
	    try:
	        os.makedirs(os.path.dirname(filename))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	          raise


def grad_calculater(bold_file):

    cii = nib.load(bold_file)
    timeseries = cii.get_fdata()

    #cleaning data
    confounds = Params36().load(bold_file)
    clean_ts = signal.clean(timeseries, confounds=confounds)

    atlas_L = nib.load('/scratch/dimuthu1/PPMI_project2/PPMI_gradients/cfg/L.atlasroi.32k_fs_LR.shape.gii')
    atlas_L = atlas_L.darrays[0].data
    atlas_R = nib.load('/scratch/dimuthu1/PPMI_project2/PPMI_gradients/cfg/R.atlasroi.32k_fs_LR.shape.gii')
    atlas_R = atlas_R.darrays[0].data
    atlas = np.hstack((atlas_L,atlas_R))

    schaefer_left = nib.load('/scratch/dimuthu1/PPMI_project2/PPMI_gradients/cfg/schaefer-1000.L.32k_fs_LR.label.gii').darrays[0].data
    schaefer_right = nib.load('/scratch/dimuthu1/PPMI_project2/PPMI_gradients/cfg/schaefer-1000.R.32k_fs_LR.label.gii').darrays[0].data
    schaefer_LR = np.hstack((schaefer_left,schaefer_right))
    labels_LR_no_medialwall = schaefer_LR[atlas==1]

    atlas = nib.load('/scratch/dimuthu1/PPMI_project2/PPMI_gradients/cfg/91282_Greyordinates.dscalar.nii').get_fdata()
    new_1 = clean_ts[:,(atlas[0]==1)]

    #This method was not used because it is returning only 999 instead of 1000. Reason was a NaN value
    #cortex_vals = reduce_by_labels(new_1, labels_LR_no_medialwall, axis=0, red_op='sum')

    #To fix the above error, I did it manually and add a zero to the NaN value. One region is giving a NaN value. Should check it later.
    mean_ts = np.zeros((len(new_1), 1000))

    for i in range(0,1000):
        mean_ts[:,i] = np.mean(new_1[:,(labels_LR_no_medialwall==i+1)], axis =1)


    mean_ts = np.nan_to_num(mean_ts)

    #subctx_dict = {0: (12, 51, 'putamen'),
    #               1: (11, 50, 'caudate'),
    #               2: (26, 58, 'accumbens')}


    #For now, only selecting the Left side

    put_L = clean_ts[:,(atlas[0]==12)]
    #put_R = clean_ts[:,(atlas[0]==51)]
    cau_L = clean_ts[:,(atlas[0]==11)]
    #cau_R = clean_ts[:,(atlas[0]==50)]
    acc_L = clean_ts[:,(atlas[0]==26)]
    #acc_R = clean_ts[:,(atlas[0]==58)]

    #Uncomment this if you want both L and R
    #subctx_vals = np.concatenate((put_L, put_R, cau_L, cau_R, acc_L, acc_R),axis=1)
    subctx_vals = np.concatenate((put_L, cau_L, acc_L),axis=1)


    #combining cortex and sub cortex
    #combined_ts = np.concatenate((cortex_vals,subctx_vals), axis=1)
    combined_ts = np.concatenate((mean_ts,subctx_vals), axis=1)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([combined_ts])[0]

    #slicing out Left hemisphere 
    sliced_matrix = correlation_matrix[:500,1000:]

    #scaling up to make it psotive
    #sliced_matrix = (sliced_matrix+1)/2

    corr_plot = plotting.plot_matrix(sliced_matrix, figure=(15, 15))

    sim_matrix = cosine_similarity(sliced_matrix, sliced_matrix)

    #uncomment these if you want to use braispace
    #gm = GradientMaps(n_components=4, random_state=0)
    #gm.fit(sim_matrix)
    #gradient_df = pd.DataFrame({'grad_1': grad_1, 'grad_2': grad_2, 'grad_3': grad_3, 'grad_4': grad_4})
    #grad_1 = gm.gradients_.T[0]
    #grad_2 = gm.gradients_.T[1]
    #grad_3 = gm.gradients_.T[2]
    #grad_4 = gm.gradients_.T[3]
    
    
    gm = DiffusionMapEmbedding(alpha=0.5, diffusion_time=1, affinity='markov', n_components=5).fit_transform(sim_matrix.copy())
    
    gradient_df = pd.DataFrame({'L_grad_1': gm[:,0],
			 'L_grad_2': gm[:,1], 'L_grad_3': gm[:,2], 'L_grad_4': gm[:,3]})


    


    

    return gradient_df,gm
    
grad_df,L_gm = grad_calculater(snakemake.input.bold)

make_out_dir(snakemake.input.gradient_path)

grad_df.to_csv(snakemake.output.gradient, index=False)

#np.save(snakemake.output.R_emb, R_gradient)
np.save(snakemake.output.L_emb, L_gm)




