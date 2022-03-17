import pandas as pd
import os
import dill
import numpy as np
from scipy.stats import zscore
from brainspace.utils.parcellation import reduce_by_labels, map_to_mask
from brainspace.gradient import GradientMaps
from sklearn.metrics.pairwise import cosine_similarity
import itertools


"""df = pd.read_table('../participants.tsv')
subjects = df.participant_id.to_list() 
subj = [ s.strip('sub-') for s in subjects ]"""

subj = snakemake.params.subjects


def make_out_dir(out_path):

	#Make subdirectories to save files
	filename = out_path
	if not os.path.exists(os.path.dirname(filename)):
	    try:
	        os.makedirs(os.path.dirname(filename))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	          raise



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
    
    gm = GradientMaps(n_components=ngradients, kernel='gaussian', approach='dm', random_state=0).fit(correlation_mean)

    #getting the gradient for individual subject and aligning with the average
    gp = GradientMaps(n_components=ngradients, kernel='gaussian',approach='dm', random_state=0,alignment='procrustes')
        
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

def get_mean_matrix(region,matrix_files):


    matrix = np.load(matrix_files[0]) #np.load(matrix_dir+"/sub-"+subj[0]+"_"+session+"_corr-matrix.npy")

    #Slicing the connectivity matrix of the first subject to get dimentions of each matrix
    #The hardcoded numbers are unique to the atlas used
    if region=='ctx':
        sliced_matrix = matrix[:1000,:1000]
    if region=='sbctx_L':
        sliced_matrix = matrix[:500,1000:2923]
        sliced_matrix= cosine_similarity(sliced_matrix.T, sliced_matrix.T) #Getting the transpose because we want striatal voxels x striatal voxels
    if region=='sbctx_R':
        sliced_matrix = matrix[500:1000,2923:]
        sliced_matrix= cosine_similarity(sliced_matrix.T, sliced_matrix.T)

    cube_matrix = np.zeros((np.shape(sliced_matrix)[0],np.shape(sliced_matrix)[1],len(subj)))

    #Loop to combine each matrix into a 3D matrix called cube_matrix
    for i,matrix_file in enumerate(matrix_files):
        
        matrix = np.load(matrix_file) #np.load(matrix_dir+"/sub-"+subjects+"_"+session+"_corr-matrix.npy")

        if region=='ctx':
            sliced_matrix = matrix[:1000,:1000]

        if region=='sbctx_L':
            sliced_matrix = matrix[:500,1000:2923]
            sliced_matrix= cosine_similarity(sliced_matrix.T, sliced_matrix.T)

        if region=='sbctx_R':
            sliced_matrix = matrix[500:1000,2923:]
            sliced_matrix= cosine_similarity(sliced_matrix.T, sliced_matrix.T)

        cube_matrix[:,:,i]=sliced_matrix

    mean_data = np.mean(cube_matrix, axis=2) #Getting the mean across subjects

    mean_data[np.isnan(mean_data)] = 0 #Replace nan values with 0
    cube_matrix[np.isnan(cube_matrix)] = 0


    return mean_data, cube_matrix


matrix_files_12 = snakemake.input.matrix_files_12
matrix_files_24 = snakemake.input.matrix_files_24

make_out_dir(snakemake.params.grad_path)

#Getting mean connectivity matrices for ctx and sbctx for both time points seperately
mean_ctx_12, cube_matrix_ctx_12 = get_mean_matrix('ctx',matrix_files_12)
mean_sbctx_L_12, cube_matrix_sbctx_L_12= get_mean_matrix('sbctx_L',matrix_files_12)
mean_sbctx_R_12, cube_matrix_sbctx_R_12= get_mean_matrix('sbctx_R',matrix_files_12)

mean_ctx_24, cube_matrix_ctx_24 = get_mean_matrix('ctx',matrix_files_24)
mean_sbctx_L_24, cube_matrix_sbctx_L_24= get_mean_matrix('sbctx_L',matrix_files_24)
mean_sbctx_R_24, cube_matrix_sbctx_R_24= get_mean_matrix('sbctx_R',matrix_files_24)

#combining m12 and m24 together and calculating a new mean
mean_ctx = np.mean( np.array([mean_ctx_12, mean_ctx_24]), axis=0 )
mean_sbctx_L = np.mean( np.array([mean_sbctx_L_12, mean_sbctx_L_24]), axis=0 )
mean_sbctx_R = np.mean( np.array([mean_sbctx_R_12, mean_sbctx_R_24]), axis=0 )

#Using that new mean to align each subject. LH and RH are aligned seperately
grad_ctx_12, aligned_grad_ctx_12 = get_aligned_gradients(mean_ctx, cube_matrix_ctx_12)
grad_sbctx_L_12, aligned_grad_sbctx_L_12 = get_aligned_gradients(mean_sbctx_L, cube_matrix_sbctx_L_12)
grad_sbctx_R_12, aligned_grad_sbctx_R_12 = get_aligned_gradients(mean_sbctx_R, cube_matrix_sbctx_R_12)

grad_ctx_24, aligned_grad_ctx_24 = get_aligned_gradients(mean_ctx, cube_matrix_ctx_24)
grad_sbctx_L_24, aligned_grad_sbctx_L_24 = get_aligned_gradients(mean_sbctx_L, cube_matrix_sbctx_L_24)
grad_sbctx_R_24, aligned_grad_sbctx_R_24= get_aligned_gradients(mean_sbctx_R, cube_matrix_sbctx_R_24)



#Saving the files
dill.dump(grad_ctx_12, file = open(snakemake.output.grad_ctx_12, "wb"))
dill.dump(grad_sbctx_L_12, file = open(snakemake.output.grad_sbctx_L_12, "wb"))
dill.dump(grad_sbctx_R_12, file = open(snakemake.output.grad_sbctx_R_12, "wb"))
dill.dump(aligned_grad_ctx_12, file = open(snakemake.output.aligned_grad_ctx_12, "wb"))
dill.dump(aligned_grad_sbctx_L_12, file = open(snakemake.output.aligned_grad_sbctx_L_12, "wb"))
dill.dump(aligned_grad_sbctx_R_12, file = open(snakemake.output.aligned_grad_sbctx_R_12, "wb"))

dill.dump(grad_ctx_24, file = open(snakemake.output.grad_ctx_24, "wb"))
dill.dump(grad_sbctx_L_24, file = open(snakemake.output.grad_sbctx_L_24, "wb"))
dill.dump(grad_sbctx_R_24, file = open(snakemake.output.grad_sbctx_R_24, "wb"))
dill.dump(aligned_grad_ctx_24, file = open(snakemake.output.aligned_grad_ctx_24, "wb"))
dill.dump(aligned_grad_sbctx_L_24, file = open(snakemake.output.aligned_grad_sbctx_L_24, "wb"))
dill.dump(aligned_grad_sbctx_R_24, file = open(snakemake.output.aligned_grad_sbctx_R_24, "wb"))





