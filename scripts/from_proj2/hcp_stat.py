import pandas as pd
import os
import glob
import io 
import numpy as np


#path_lh ='/home/dimuthu1/scratch/project2/derivatives/hcp_360/work/MyWorkflow/_sub_id_*/parce/out_put/*/tables/table_lh.txt'

#path_rh ='/home/dimuthu1/scratch/project2/derivatives/hcp_360/work/MyWorkflow/_sub_id_*/parce/out_put/*/tables/table_rh.txt'


hcp_lh_stat = glob.glob(snakemake.input.stat_lh)
hcp_rh_stat = glob.glob(snakemake.input.stat_rh)


def make_out_dir(out_path):

	#Make subdirectories to save files
	filename = out_path
	if not os.path.exists(os.path.dirname(filename)):
	    try:
	        os.makedirs(os.path.dirname(filename))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	          raise

make_out_dir(snakemake.params.stat_out_path)


def get_dfs(txt_file):


	data = np.loadtxt(txt_file,dtype='str',delimiter='   ', skiprows=16)

	#The txt files have different delimiters. The following lines are just trying to fix messed up colums
	col3 = [i.split('  ') for i in data[:,2]]
	volume = np.array(col3)[:,0]
	thickness = np.array(col3)[:,1]

	thickness_n_error = [i.split(' ') for i in thickness]
	thickness = np.array(thickness_n_error)[:,0]
	error = np.array(thickness_n_error)[:,1]

	col7 = [i.split('  ') for i in data[:,7]]
	index = np.array(col7)[:,1]
	roi = np.array(col7)[:,2]



	dataset = pd.DataFrame({'Num_of_vertices': data[:, 0].astype(np.float), 'area_mm^2': data[:, 1].astype(np.float), 
		'volume_mm^3': volume.astype(np.float),'thickness_mm': thickness.astype(np.float),'thickness_error': error.astype(np.float),
		'mean_curv': data[:, 3].astype(np.float),'Gauss_curv': data[:, 4].astype(np.float),
		'folding index': data[:, 6].astype(np.float), 'curvature_index': index, 'ROI_name': roi})


	return dataset



df = get_dfs(snakemake.input.stat_lh)
df.to_csv(snakemake.output.stat_csv_lh, index=False)

df = get_dfs(snakemake.input.stat_rh)
df.to_csv(snakemake.output.stat_csv_rh, index=False)









