import pandas as pd
import os
import glob
import io 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

#make_out_dir(snakemake.params.stat_out_path)


#path_lh ='/home/dimuthu1/scratch/project2/derivatives/hcp_360/work/MyWorkflow/_sub_id_*/parce/out_put/*/tables/table_lh.txt'

#path_rh ='/home/dimuthu1/scratch/project2/derivatives/hcp_360/work/MyWorkflow/_sub_id_*/parce/out_put/*/tables/table_rh.txt'

hcp_lh = snakemake.input.hcp_lh   #'../../derivatives/analysis/hcp_stat/sub-CT01/CT01_hcp_stat_lh.csv'
hcp_rh = snakemake.input.hcp_rh  #'../../derivatives/analysis/hcp_stat/sub-CT01/CT01_hcp_stat_rh.csv'
gradient = snakemake.input.gradient #'../../derivatives/analysis/gradients/sub-CT01/gradients.csv'

grads = pd.read_csv(gradient)
gradient_lh = grads[['L_grad_1','L_grad_2','L_grad_3','L_grad_4']]
gradient_rh = grads[['R_grad_1','R_grad_2','R_grad_3','R_grad_4']]

stats_lh = pd.read_csv(hcp_lh) 
stats_rh = pd.read_csv(hcp_rh) 

stats_lh = stats_lh.drop(['Num_of_vertices','thickness_error', 'ROI_name'], axis = 1) 
stats_rh = stats_rh.drop(['Num_of_vertices','thickness_error', 'ROI_name'], axis = 1) 


gradient_lh = gradient_lh.drop([0]).reset_index(drop=True)
gradient_rh = gradient_rh.drop([0]).reset_index(drop=True)


combined_lh = pd.concat([stats_lh, gradient_lh], axis=1, sort=False)
combined_rh = pd.concat([stats_rh, gradient_rh], axis=1, sort=False)



sns_plot = sns.pairplot(combined_lh)
plt.show()
sns_plot.savefig(snakemake.output.lh_stat_plots)

sns_plot = sns.pairplot(combined_rh)
plt.show()
sns_plot.savefig(snakemake.output.rh_stat_plots)

#sns_plot_1 = sns.jointplot("L_grad_2", "thickness", data=combined_lh, kind="reg", truncate=False)
#plt.show()
#sns_plot_1.savefig("thick_vs_g2_PD.png")

#plt.show()














