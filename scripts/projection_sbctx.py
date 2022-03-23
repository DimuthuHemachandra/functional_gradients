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

def fill_array(mask_data,grad_array,low,up,roi):

    sliced_array = grad_array[low:up]
    mask_shape=np.shape(mask_data)
    #print(np.shape(sliced_array))
    q=0
    for i in range(0,mask_shape[0]):
        for j in range(0,mask_shape[1]):
            for k in range(0,mask_shape[2]):
                if mask_data[i,j,k] == roi:
                    mask_data[i,j,k]= sliced_array[q]
                    q=q+1   

    return mask_data




def get_sbctx_projections(grad_array,side,output_file):
    
    
    if side =='R':
        mask_data = nib.load(snakemake.params.str_rh).get_fdata()
        img = nib.load(snakemake.params.str_rh)
        mask_data = fill_array(mask_data,grad_array,0,1010,51)
        mask_data = fill_array(mask_data,grad_array,1010,1765,50)
        mask_data = fill_array(mask_data,grad_array,1765,1905,58)
        final_img = nib.Nifti1Image(mask_data, img.affine, img.header)
        nib.save(final_img,output_file)

    if side =='L':
        mask_data = nib.load(snakemake.params.str_lh).get_fdata()
        img = nib.load(snakemake.params.str_lh)
        mask_data = fill_array(mask_data,grad_array,0,1060,12)
        mask_data = fill_array(mask_data,grad_array,1060,1788,11)
        mask_data = fill_array(mask_data,grad_array,1788,1923,26)
        final_img = nib.Nifti1Image(mask_data, img.affine, img.header)
        nib.save(final_img,output_file)





#emb_dir ='/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/smoothed/gradients/bs_emb'



def get_sbctx(sbctx_L,sbctx_R,sbctx_mean_L,sbctx_mean_R,grad):

    aligned_emb_sbctx_R = dill.load(open(snakemake.input.aligned_grad_sbctx_R, "rb"))
    aligned_emb_sbctx_L = dill.load(open(snakemake.input.aligned_grad_sbctx_L, "rb"))
    mean_emb_sbctx_L= dill.load(open(snakemake.input.mean_grad_sbctx_L, "rb"))
    mean_emb_sbctx_R= dill.load(open(snakemake.input.mean_grad_sbctx_R, "rb"))


    for i, s in enumerate(subj):
        emb_sbctx_L = aligned_emb_sbctx_L[:,grad,i] #projecting gradient 3
        emb_sbctx_R = aligned_emb_sbctx_R[:,grad,i] #projecting gradient 3
        
        get_sbctx_projections(emb_sbctx_L,'L',sbctx_L[i])
        get_sbctx_projections(emb_sbctx_R,'R',sbctx_R[i])



    emb_sbctx_L_mean = mean_emb_sbctx_L.gradients_[:,grad]  #Mean gradient is passed as an object
    emb_sbctx_R_mean = mean_emb_sbctx_R.gradients_[:,grad]  #Mean gradient is passed as an object
    #print(np.shape(emb_sbctx_R_12.gradients_))
    get_sbctx_projections(emb_sbctx_L_mean,'L',sbctx_mean_L)
    get_sbctx_projections(emb_sbctx_R_mean,'R',sbctx_mean_R)



make_out_dir(snakemake.params.out_path+'/sbctx/')
#get_sbctx()



#sorting file names for aligned gradients according to gradient number
grad1_L_list= [x for x in snakemake.output.sbctx_L if 'grad1' in x ]
grad2_L_list= [x for x in snakemake.output.sbctx_L if 'grad2' in x ]
grad3_L_list= [x for x in snakemake.output.sbctx_L if 'grad3' in x ]

grad1_R_list= [x for x in snakemake.output.sbctx_R if 'grad1' in x ]
grad2_R_list= [x for x in snakemake.output.sbctx_R if 'grad2' in x ]
grad3_R_list= [x for x in snakemake.output.sbctx_R if 'grad3' in x ]

#sorting file names for mean gradients according to gradient number
mean_grad1_L_list= [x for x in snakemake.output.sbctx_mean_L if 'grad1' in x ]
mean_grad2_L_list= [x for x in snakemake.output.sbctx_mean_L if 'grad2' in x ]
mean_grad3_L_list= [x for x in snakemake.output.sbctx_mean_L if 'grad3' in x ]

mean_grad1_R_list= [x for x in snakemake.output.sbctx_mean_R if 'grad1' in x ]
mean_grad2_R_list= [x for x in snakemake.output.sbctx_mean_R if 'grad2' in x ]
mean_grad3_R_list= [x for x in snakemake.output.sbctx_mean_R if 'grad3' in x ]

get_sbctx(grad1_L_list, grad1_R_list, mean_grad1_L_list[0], mean_grad1_R_list[0],0)
get_sbctx(grad2_L_list, grad2_R_list, mean_grad2_L_list[0], mean_grad2_R_list[0],1)
get_sbctx(grad3_L_list, grad3_R_list, mean_grad3_L_list[0], mean_grad3_R_list[0],2)


