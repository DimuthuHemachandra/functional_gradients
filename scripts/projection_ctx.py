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
import dill



labeling = nib.load(snakemake.params.labels).get_fdata()
#print(np.shape(labeling[1]))
mask = ~np.isin(labeling[0],0)

labeling = labeling.astype(int)

#labeling = np.squeeze(labeling).shape
#print(np.shape(labeling))

#df = pd.read_table('../participants.tsv')
#subjects = df.participant_id.to_list() 
#subj = [ s.strip('sub-') for s in subjects ]

subj = snakemake.params.subjects

nsubjects = len(subj)

def make_out_dir(out_path):

	#Make subdirectories to save files
	filename = out_path
	if not os.path.exists(os.path.dirname(filename)):
	    try:
	        os.makedirs(os.path.dirname(filename))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	          raise

#projecting to cortex. Left and Right seperately 

def project_to_cortex(emb_ctx,out_L,out_R):
    grad = [None] * 4  #Number of gradients to project

    #Left Hemisphere
    gii_L = nib.gifti.GiftiImage()
    for i, g in enumerate(emb_ctx.T):
        grad[i] = map_to_labels(g, labeling[0], mask=mask, fill=np.nan)

    for g in range(0,len(grad)):
        gii_L.add_gifti_data_array(nib.gifti.GiftiDataArray(data=grad[g][:32492].astype(np.float32)))# For left hemisphere, right hemisphere change to [32492:]
    
    nib.save(gii_L, out_L)

    #Right Hemisphere
    gii_R = nib.gifti.GiftiImage()
    for i, g in enumerate(emb_ctx.T):
        grad[i] = map_to_labels(g, labeling[0], mask=mask, fill=np.nan)

    for g in range(0,len(grad)):
        gii_R.add_gifti_data_array(nib.gifti.GiftiDataArray(data=grad[g][32492:].astype(np.float32)))# For left hemisphere, right hemisphere change to [32492:]
    
    nib.save(gii_R, out_R)

#emb_dir ='/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/smoothed/gradients/bs_emb'

def get_cortex():

    aligned_emb_ctx = dill.load(open(snakemake.input.aligned_grad_ctx, "rb"))
    mean_emb_ctx= dill.load(open(snakemake.input.grad_ctx, "rb"))


    #Projecting individual aligned gradients to the cortex
    for i, s in enumerate(subj):
        emb_ctx = aligned_emb_ctx[:,:,i]
        project_to_cortex(emb_ctx,snakemake.output.ctx_L[i], snakemake.output.ctx_R[i])

    #Projecting the gradient based on the mean
    #grad_num=3 #Number of gradients interested in projecting
    #for i in range(grad_num):
    mean_emb_ctx= mean_emb_ctx.gradients_ #[:,i]  #Mean gradient is saved as an object. To get gradients, we have to use .gradients_
    #print(np.shape(emb_sbctx_R_12.gradients_))
    project_to_cortex(mean_emb_ctx,snakemake.output.mean_ctx_L, snakemake.output.mean_ctx_R)



make_out_dir(snakemake.params.out_path+'/cortex/')
get_cortex()



