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


print(snakemake.input.nii_files)
"""
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



m12 = []
m24 = []

for i, s in enumerate(subj):

    m12.append(glob.glob(out_path+'/'+s+'-m12_sbctx_R_grad_3_3d.nii')[0])
    m24.append(glob.glob(out_path+'/'+s+'-m24_sbctx_R_grad_3_3d.nii')[0])


all_nii = m12 + m24
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

