import pandas as pd
import os
import glob
import io 
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics.pairwise import cosine_similarity
from nilearn.image import concat_imgs
from nilearn.image import concat_imgs

def make_out_dir(out_path):

	#Make subdirectories to save files
	filename = out_path
	if not os.path.exists(os.path.dirname(filename)):
	    try:
	        os.makedirs(os.path.dirname(filename))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	          raise

def get_tstat(hemi, n):
    out_path = snakemake.params.out_path+'grad'+n+'/'+hemi+'h'
    make_out_dir(out_path)
    files = snakemake.input.nii_files
    diff_files = snakemake.output.diff_grad
    m12_R_g1= [x for x in files if 'month12_sbctx_'+hemi+'_aligned_grad'+n in x]
    m24_R_g1= [x for x in files if 'month24_sbctx_'+hemi+'_aligned_grad'+n in x]
    diff_R_g1= [x for x in diff_files if '_'+hemi+'_grad'+n+'_' in x]

    for i, s in enumerate(m12_R_g1):

        os.system('fslmaths '+m24_R_g1[i]+' -sub '+m12_R_g1[i]+' '+diff_R_g1[i])

    image_diff_4D = concat_imgs(diff_R_g1)
    nib.save(image_diff_4D, out_path+'/concatednated_diff_4D.nii')
    os.system('randomise -i '+out_path+'/concatednated_diff_4D.nii -o '+out_path+'/OneSampT -1 -T')

get_tstat('R', '1')
get_tstat('R', '2')
get_tstat('R', '3')

get_tstat('L', '1')
get_tstat('L', '2')
get_tstat('L', '3')
"""
print(m12_R_g1)
print(m24_R_g1)

all_nii = m12_R_g1 + m24_R_g1
print(np.shape(all_nii))

image_4D = concat_imgs(all_nii)

nib.save(image_4D, out_path+'/concatednated_4D.nii')

#result_img = smooth_img(out_path+'/*3d.nii') 


diff_files = []
for i, s in enumerate(subj):

    diff_files.append(glob.glob(out_path+'/'+s+'_diff.nii.gz')[0])

#print(diff_files)

image_diff_4D = concat_imgs(diff_files)
nib.save(image_diff_4D, out_path+'/concatednated_diff_4D.nii')"""

