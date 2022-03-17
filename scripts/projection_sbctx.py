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




def get_sbctx_projections(grad_array,side,output_file):
    atlas = nib.load(snakemake.params.greyordinates).get_fdata()
    img = nib.load(snakemake.params.greyordinates)

    true_vals = atlas[0]
    
    if side =='R':
        put_R = np.where(true_vals==51)
        cau_R = np.where(true_vals==50)
        acc_R = np.where(true_vals==58)

        subctx_vals = np.concatenate((put_R, cau_R, acc_R),axis=1)

    if side =='L':
        put_L = np.where(true_vals==12)
        cau_L = np.where(true_vals==11)
        acc_L = np.where(true_vals==26)

        subctx_vals = np.concatenate((put_L, cau_L, acc_L),axis=1)

    print(np.shape(true_vals))
    print(np.shape(subctx_vals[0,:]))
    sbctx_ind = subctx_vals[0,:]
    for i, val  in enumerate(sbctx_ind):
        true_vals[val]= grad_array[i] #*1000

    for i, val in enumerate(true_vals):
        #print(i)
        if i not in sbctx_ind.tolist():
            true_vals[i]= 0 #float("nan")

    true_vals = true_vals.reshape(1,-1)
    print(output_file)
    new_img = nib.Cifti2Image(true_vals, img.header)
    nib.save(new_img, output_file)


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


