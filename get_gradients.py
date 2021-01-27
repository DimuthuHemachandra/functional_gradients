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

in_cii = '/scratch/dimuthu1/PPMI_project2/derivatives/fmriprep_20_2_1_test_syn_sdc/fmriprep/sub-3116/ses-Month12/func/sub-3116_ses-Month12_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii'
cii = nib.load(in_cii)
timeseries = cii.get_fdata()


file = "/scratch/dimuthu1/PPMI_project2/derivatives/fmriprep_20_2_1_test_syn_sdc/fmriprep/sub-3116/ses-Month12/func/sub-3116_ses-Month12_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii"
confounds = Params36().load(file)

clean_ts = signal.clean(timeseries, confounds=confounds)


#clean_ts = clean_ts.T

atlas_L = nib.load('../L.atlasroi.32k_fs_LR.shape.gii')
atlas_L = atlas_L.darrays[0].data
atlas_R = nib.load('../R.atlasroi.32k_fs_LR.shape.gii')
atlas_R = atlas_R.darrays[0].data
atlas = np.hstack((atlas_L,atlas_R))


schaefer_left = nib.load('../schaefer-1000.L.32k_fs_LR.label.gii').darrays[0].data
schaefer_right = nib.load('../schaefer-1000.R.32k_fs_LR.label.gii').darrays[0].data
schaefer_LR = np.hstack((schaefer_left,schaefer_right))
labels_LR_no_medialwall = schaefer_LR[atlas==1]

atlas = nib.load('../91282_Greyordinates.dscalar.nii').get_fdata()
new_1 = clean_ts[:,(atlas[0]==1)]



cortex_vals = reduce_by_labels(new_1, labels_LR_no_medialwall, axis=0, red_op='sum')


mean_ts = np.zeros((len(new_1), 1000))
#print(len(clean_ts))

for i in range(0,1000):
    mean_ts[:,i] = np.mean(new_1[:,(labels_LR_no_medialwall==i+1)], axis =1)


mean_ts = np.nan_to_num(mean_ts)

#fmri_data_ctx = fmri_fdata[:,0:len(schaefer_LR)]

#all = np.zeros((len(fmri_data_ctx), 1))

subctx_dict = {0: (12, 51, 'putamen'),
               1: (11, 50, 'caudate'),
               2: (26, 58, 'accumbens')}

put_L = clean_ts[:,(atlas[0]==12)]
#put_R = clean_ts[:,(atlas[0]==51)]
cau_L = clean_ts[:,(atlas[0]==11)]
#cau_R = clean_ts[:,(atlas[0]==50)]
acc_L = clean_ts[:,(atlas[0]==26)]
#acc_R = clean_ts[:,(atlas[0]==58)]

print(np.shape(put_L))
print(np.shape(acc_L))
#subctx_vals = np.concatenate((put_L, put_R, cau_L, cau_R, acc_L, acc_R),axis=1)
subctx_vals = np.concatenate((put_L, cau_L, acc_L),axis=1)


#combining cortex and sub cortex
print(np.unique(mean_ts))
#combined_ts = np.concatenate((cortex_vals,subctx_vals), axis=1)
combined_ts = np.concatenate((mean_ts,subctx_vals), axis=1)




correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([combined_ts])[0]




# Reduce matrix size, only for visualization purposes
#mat_mask = np.where(np.std(correlation_matrix, axis=1) > 0.2)[0]
#c = correlation_matrix[mat_mask][:, mat_mask]

# Create corresponding region names
#regions_list = ['%s_%s' % (h, r.decode()) for h in ['L', 'R'] for r in regions]
#masked_regions = [regions_list[i] for i in mat_mask]


#corr_plot = plotting.plot_matrix(correlation_matrix, figure=(15, 15), vmax=0.8, vmin=-0.8)

#plt.show()

sliced_matrix = correlation_matrix[:500,1000:]
#slice = arr[0:2,0:2]



sliced_matrix = (sliced_matrix+1)/2

print(np.unique(sliced_matrix))

corr_plot = plotting.plot_matrix(sliced_matrix, figure=(15, 15))



sim_matrix = cosine_similarity(sliced_matrix, sliced_matrix)



#corr_plot = plotting.plot_matrix(sim_matrix, figure=(15, 15))

#plt.show()

gm = GradientMaps(n_components=3, random_state=0)
gm.fit(sim_matrix)

grad_1 = gm.gradients_.T[0]
grad_2 = gm.gradients_.T[1]
grad_3 = gm.gradients_.T[2]





dataset = pd.DataFrame({'grad_1': grad_1, 'grad_2': grad_2, 'grad_3': grad_3})

print(dataset)

#sns.scatterplot(x="grad_1", y="grad_2",data = dataset)

#plt.show()




