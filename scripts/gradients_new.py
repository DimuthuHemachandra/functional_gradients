import pandas as pd
import os
import glob
import io 
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as sns
from scipy.stats import zscore
from brainspace.utils.parcellation import reduce_by_labels, map_to_mask
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_parcellation
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_table('../participants.tsv')
subjects = df.participant_id.to_list() 
subj = [ s.strip('sub-') for s in subjects ]

labels_gii = nib.load('../cfg/Schaefer2018_1000Parcels_7Networks_order.dlabel.nii').get_fdata()
mask = ~np.isin(labels_gii[0],0)


def get_gradients(matrix,region):

    #if region!='ctx':
    matrix= cosine_similarity(matrix.T, matrix.T)

    gm = GradientMaps(n_components=4, random_state=0)
    gm.fit(matrix)

    grad_1 = gm.gradients_.T[0]
    grad_2 = gm.gradients_.T[1]
    grad_3 = gm.gradients_.T[2]
    #grad_4 = gm.gradients_.T[3]

    gradient_df = pd.DataFrame({region+'_grad_1': grad_1, region+'_grad_2': grad_2, region+'_grad_3': grad_3})

    return gradient_df


import itertools
def reject_outliers_2(matrix, m=2.):

    mean_matrix = np.zeros((np.shape(matrix)[0],np.shape(matrix)[1]))
    for i, j in itertools.product(range(np.shape(matrix)[0]),range(np.shape(matrix)[1])):

        data = matrix[i,j,:]
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / (mdev if mdev else 1.)
        mean_matrix[i,j] = np.mean(data[s < m])
    return mean_matrix

def get_aligned_gradients(correlation_mean, correlation_matrix):

    #getting the gradinet for the mean
    ngradients = 4
    gm = GradientMaps(n_components=ngradients, kernel='normalized_angle', approach='dm', random_state=0).fit(correlation_mean)
    #gm_emb = gm.fit(correlation_mean)

    #getting the gradient for individual subject and aligning with the average
    gp = GradientMaps(n_components=ngradients, kernel='normalized_angle',approach='dm', random_state=0,alignment='procrustes')
        
    # With procrustes alignment
    nsubjects = len(subj)

    no_rois = np.shape(correlation_matrix)[0]
    print(no_rois)
 
    grad_aligned = np.zeros((no_rois,ngradients,nsubjects))


    for s in range(0,len(subj)):
        gp.fit(correlation_matrix[:,:,s], reference=gm.gradients_) # i.e., 3D matrix, with third dimension = subjects

        grad_aligned[:,:,s] = gp.aligned_

        #for i, g in enumerate(gp.aligned_.T):
        #    grad_aligned[:,i,s] = map_to_labels(g, labels_gii[0],mask=mask, fill=np.nan)


    return gm,grad_aligned

def get_mean_matrix(session,region):
    #subj = ['3119','3120']
    matrix_dir = '/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/corr_matrix/'+session

    matrix = np.load(matrix_dir+"/sub-"+subj[0]+"_"+session+"_corr-matrix.npy")

    if region=='ctx':
        sliced_matrix = matrix[:1000,:1000]
    if region=='sbctx_L':
        sliced_matrix = matrix[:500,1000:2914]
        sliced_matrix= cosine_similarity(sliced_matrix.T, sliced_matrix.T)
    if region=='sbctx_R':
        sliced_matrix = matrix[500:1000,2914:]
        sliced_matrix= cosine_similarity(sliced_matrix.T, sliced_matrix.T)

    cube_matrix = np.zeros((np.shape(sliced_matrix)[0],np.shape(sliced_matrix)[1],len(subj)))

    
    for i,subjects in enumerate(subj):
        
        matrix = np.load(matrix_dir+"/sub-"+subjects+"_"+session+"_corr-matrix.npy")

        if region=='ctx':
            sliced_matrix = matrix[:1000,:1000]

        if region=='sbctx_L':
            sliced_matrix = matrix[:500,1000:2914]
            sliced_matrix= cosine_similarity(sliced_matrix.T, sliced_matrix.T)

        if region=='sbctx_R':
            sliced_matrix = matrix[500:1000,2914:]
            sliced_matrix= cosine_similarity(sliced_matrix.T, sliced_matrix.T)

        cube_matrix[:,:,i]=sliced_matrix

    mean_data = np.mean(cube_matrix, axis=2)
    #mean_data = reject_outliers_2(cube_matrix, m=2.)

    return mean_data, cube_matrix





mean_ctx_12, cube_matrix_ctx_12 = get_mean_matrix('month12','ctx')
mean_ctx_24, cube_matrix_ctx_24 = get_mean_matrix('month24','ctx')

mean_sbctx_L_12, cube_matrix_sbctx_L_12= get_mean_matrix('month12','sbctx_L')
mean_sbctx_L_24, cube_matrix_sbctx_L_24 = get_mean_matrix('month24','sbctx_L')

mean_sbctx_R_12, cube_matrix_sbctx_R_12= get_mean_matrix('month12','sbctx_R')
mean_sbctx_R_24, cube_matrix_sbctx_R_24 = get_mean_matrix('month24','sbctx_R')



grad_ctx_12, aligned_grad_ctx_12 = get_aligned_gradients(mean_ctx_12, cube_matrix_ctx_12)
grad_ctx_24, aligned_grad_ctx_24 = get_aligned_gradients(mean_ctx_24, cube_matrix_ctx_24)

grad_sbctx_L_12, aligned_grad_sbctx_L_12 = get_aligned_gradients(mean_sbctx_L_12, cube_matrix_sbctx_L_12)
grad_sbctx_L_24, aligned_grad_sbctx_L_24 = get_aligned_gradients(mean_sbctx_L_24, cube_matrix_sbctx_L_24)

grad_sbctx_R_12, aligned_grad_sbctx_R_12 = get_aligned_gradients(mean_sbctx_R_12, cube_matrix_sbctx_R_12)
grad_sbctx_R_24, aligned_grad_sbctx_R_24 = get_aligned_gradients(mean_sbctx_R_24, cube_matrix_sbctx_R_24)


import dill

# Save the file
dill.dump(grad_ctx_12, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/emb_ctx_12.pickle", "wb"))
dill.dump(grad_ctx_24, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/emb_ctx_24.pickle", "wb"))

dill.dump(grad_sbctx_L_12, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/emb_sbctx_L_12.pickle", "wb"))
dill.dump(grad_sbctx_L_24, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/emb_sbctx_L_24.pickle", "wb"))

dill.dump(grad_sbctx_R_12, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/emb_sbctx_R_12.pickle", "wb"))
dill.dump(grad_sbctx_R_24, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/emb_sbctx_R_24.pickle", "wb"))

dill.dump(aligned_grad_ctx_12, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/aligned_emb_ctx_12.pickle", "wb"))
dill.dump(aligned_grad_ctx_24, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/aligned_emb_ctx_24.pickle", "wb"))

dill.dump(aligned_grad_sbctx_L_12, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/aligned_emb_sbctx_L_12.pickle", "wb"))
dill.dump(aligned_grad_sbctx_L_24, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/aligned_emb_sbctx_L_24.pickle", "wb"))

dill.dump(aligned_grad_sbctx_R_12, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/aligned_emb_sbctx_R_12.pickle", "wb"))
dill.dump(aligned_grad_sbctx_R_24, file = open("/home/dimuthu1/temporay_scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/aligned_emb_sbctx_R_24.pickle", "wb"))



#np.save('/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/emb_ctx_12.npy', grad_ctx_12, allow_pickle=True)
#np.save('/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/emb_ctx_24.npy', grad_ctx_24, allow_pickle=True)
#np.save('/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/emb_sbctx_12.npy', grad_sbctx_12, allow_pickle=True)
#np.save('/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/emb_sbctx_24.npy', grad_sbctx_24, allow_pickle=True)

#np.save('/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/aligned_emb_ctx_12.npy', aligned_grad_ctx_12)
#np.save('/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/aligned_emb_ctx_24.npy', aligned_grad_ctx_24)
#np.save('/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/aligned_emb_sbctx_12.npy', aligned_grad_sbctx_12)
#np.save('/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb/aligned_emb_sbctx_24.npy', aligned_grad_sbctx_24)