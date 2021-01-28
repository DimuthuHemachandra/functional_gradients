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

hcp_dir = snakemake.input.hcp_path
grad_dir = snakemake.input.grad_csv_path
subj = snakemake.params.subj
out_path = snakemake.params.out_path



#myelin_lh = pd.read_csv("test_materials/hcp_360/sub-CT01/CT01_mean_myelin_R.txt", names = ['myelin'])

#print(myelin_lh)


#['CT01','CT02','CT03','PD01','PD02','PD03']
PDs_L= []
PDs_R= []



for subjects in subj:

	stats_lh = pd.read_csv(hcp_dir+"/sub-"+subjects+"/"+subjects+"_hcp_stat_lh.csv")
	stats_rh = pd.read_csv(hcp_dir+"/sub-"+subjects+"/"+subjects+"_hcp_stat_rh.csv")

	stats_lh = stats_lh.drop(['Num_of_vertices','thickness_error', 'ROI_name'], axis = 1) 
	stats_rh = stats_rh.drop(['Num_of_vertices','thickness_error', 'ROI_name'], axis = 1) 
	
	grads = pd.read_csv(grad_dir+"/sub-"+subjects+"_gradients.csv")
	gradient_lh = grads[['L_grad_1','L_grad_2','L_grad_3']]
	gradient_rh = grads[['R_grad_1','R_grad_2','R_grad_3']]
	gradient_lh = gradient_lh.drop([0]).reset_index(drop=True)
	gradient_rh = gradient_rh.drop([0]).reset_index(drop=True)
	
	
	myelin_lh = pd.read_csv(hcp_dir+"/sub-"+subjects+"/sub-"+subjects+"_mean_myelin_L.txt", names = ['myelin'])
	myelin_rh = pd.read_csv(hcp_dir+"/sub-"+subjects+"/sub-"+subjects+"_mean_myelin_R.txt", names = ['myelin'])

	if 'CT' in subjects:
		CT_list = ['CT']*179
		CT_df = pd.DataFrame(CT_list, columns=['group']) 
		combined_lh = pd.concat([stats_lh, myelin_lh, gradient_lh, CT_df ], axis=1, sort=False)
		PDs_L.append(combined_lh)

		combined_rh = pd.concat([stats_rh, myelin_rh, gradient_rh, CT_df ], axis=1, sort=False)
		PDs_R.append(combined_rh)

	if 'PD' in subjects:
		PD_list = ['PD']*179
		PD_df = pd.DataFrame(PD_list, columns=['group']) 
		combined_lh = pd.concat([stats_lh, myelin_lh, gradient_lh, PD_df], axis=1, sort=False)
		PDs_L.append(combined_lh)

		combined_rh = pd.concat([stats_rh, myelin_rh, gradient_rh, PD_df], axis=1, sort=False)
		PDs_R.append(combined_rh)



df_L = pd.concat(PDs_L)
#drop_numerical_outliers(df_L)
#df_L = remove_outliers(df_L)

df_R = pd.concat(PDs_R)
#drop_numerical_outliers(df_R)
#df_R = remove_outliers(df_R)

df_L.to_csv(out_path+'/all_stat_L.csv', index=False)
df_R.to_csv(out_path+'/all_stat_R.csv', index=False)

print(df_L)
#print(df_R)
#sns.color_palette("viridis", as_cmap=True)
sns_plot_L = sns.pairplot(df_L, plot_kws={"s": 8}, hue="group")
sns_plot_R = sns.pairplot(df_R, plot_kws={"s": 8}, hue="group")
#plt.show()
sns_plot_L.savefig(out_path+"/group_stat_L.png")
sns_plot_R.savefig(out_path+"/group_stat_R.png")

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











