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
	filtered_entries = (abs_z_scores < 2).all(axis=1)
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
grad_12_dir = snakemake.input.grad_12_csv_path
grad_24_dir = snakemake.input.grad_24_csv_path
subj = snakemake.params.subj
out_path = snakemake.params.out_path

make_out_dir(out_path)

#myelin_lh = pd.read_csv("test_materials/hcp_360/sub-CT01/CT01_mean_myelin_R.txt", names = ['myelin'])

#print(myelin_lh)

def get_plots(region, grad_12_dir, grad_24_dir):

	#subj = ['3118','3119','3120']
	m12= []
	m24= []


	#grad_12_dir='/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/cortex/aligned_gradients/month12'
	#grad_24_dir='/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/cortex/aligned_gradients/month24'
	#out_path='/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/cortex'


	for subjects in subj:

		
		grads12 = pd.read_csv(grad_12_dir+"/sub-"+subjects+"_gradients.csv")
		gradient12 = grads12[[region+'_grad_1',region+'_grad_2',region+'_grad_3']]

		gradient12 = gradient12.drop([0]).reset_index(drop=True)

		#gradient_lh = remove_outliers(gradient_lh)
		m12.append(gradient12)
		gradient12['month'] = 'm12'

		
		grads24 = pd.read_csv(grad_24_dir+"/sub-"+subjects+"_gradients.csv")
		gradient24 = grads24[[region+'_grad_1',region+'_grad_2',region+'_grad_3']]

		gradient24 = gradient24.drop([0]).reset_index(drop=True)
	
		
		#gradient_lh = remove_outliers(gradient_lh)
		m24.append(gradient24)
		gradient24['month'] = 'm24'

		
		df_subj = gradient24.append(gradient12)
		sns_plot = sns.scatterplot(data = df_subj, x=region+'_grad_1',y=region+'_grad_2', hue="month")
		plt.savefig(out_path+"/group_"+subjects+"_stat_"+region+".png")
		plt.close()



get_plots('ctx', grad_12_dir, grad_24_dir)
get_plots('sbctx', grad_12_dir, grad_24_dir)














