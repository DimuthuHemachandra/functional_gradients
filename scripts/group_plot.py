import pandas as pd
import os
import glob
import io 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

sns.set()

def make_out_dir(out_path):

	#Make subdirectories to save files
	filename = out_path
	if not os.path.exists(os.path.dirname(filename)):
	    try:
	        os.makedirs(os.path.dirname(filename))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	          raise

def remove_outliers(df):
	z_scores = zscore(df)

	abs_z_scores = np.abs(z_scores)
	filtered_entries = (abs_z_scores < 3).all(axis=1)
	new_df = df[filtered_entries]

	return new_df

from scipy import stats

def drop_numerical_outliers(df, z_thresh=2):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh, result_type='reduce') \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)

#make_out_dir(snakemake.params.stat_out_path)

#hcp_dir = snakemake.input.hcp_path
grad_dir = snakemake.input.grad_csv_path
subj = snakemake.params.subj
out_path = snakemake.params.out_path



#myelin_lh = pd.read_csv("test_materials/hcp_360/sub-CT01/CT01_mean_myelin_R.txt", names = ['myelin'])

#print(myelin_lh)


#subj = ['3118','3119','3120']
m12_L= []

#grad_dir='/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/cortex/aligned_gradients/month12'
#out_path='/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/cortex'


for subjects in subj:

	
	grads = pd.read_csv(grad_dir+"/sub-"+subjects+"_L_gradients.csv")
	gradient_lh = grads[['L_grad_1','L_grad_2','L_grad_3']]
	#gradient_rh = grads[['R_grad_1','R_grad_2','R_grad_3']]
	gradient_lh = gradient_lh.drop([0]).reset_index(drop=True)
	#gradient_rh = gradient_rh.drop([0]).reset_index(drop=True)
	
	#gradient_lh = remove_outliers(gradient_lh)
	m12_L.append(gradient_lh)



df_L = pd.concat(m12_L)
#drop_numerical_outliers(df_L)
#df_L = remove_outliers(df_L)

#df_R = pd.concat(PDs_R)
#drop_numerical_outliers(df_R)
#df_R = remove_outliers(df_R)

df_L.to_csv(out_path+'/all_stat_L.csv', index=False)
#df_R.to_csv(out_path+'/all_stat_R.csv', index=False)

print(df_L)
#print(df_R)
#sns.color_palette("viridis", as_cmap=True)
sns_plot_L = sns.pairplot(df_L,plot_kws={"s": 5})
#sns_plot_R = sns.pairplot(df_R, plot_kws={"s": 8}, hue="group")
#plt.show()
sns_plot_L.savefig(out_path+"/group_stat_L.png")
#sns_plot_R.savefig(out_path+"/group_stat_R.png")
#plt.show()

"""
plt.clf()

#sns.color_palette("viridis", as_cmap=True)
ct_df = df_L.loc[df_L['group'] == 'CT']
ct_df = ct_df.drop(['group'], axis = 1)
sns_plot_ct = sns.heatmap(ct_df.corr(), annot=True)
sns_plot_ct.figure.savefig(out_path+"/ct_L_heatmap.png")

plt.clf()


PD_df = df_L.loc[df_L['group'] == 'PD']
PD_df = PD_df.drop(['group'], axis = 1)
sns_plot_PD = sns.heatmap(PD_df.corr(), annot=True)
sns_plot_PD.figure.savefig(out_path+"/PD_L_heatmap.png")
"""










