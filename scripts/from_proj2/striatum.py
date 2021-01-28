#from brainspace.datasets import load_conte69
#from brainspace.plotting import plot_hemispheres
#from brainspace.datasets import load_group_fc, load_parcellation
import matplotlib.pyplot as plt
from brainspace.gradient import GradientMaps
import numpy as np
#from brainspace.utils.parcellation import map_to_labels
import scipy.io as sio
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nilearn.image import load_img
import nibabel as nib
import os
import glob

from mapalign.embed import DiffusionMapEmbedding

from scipy.io import loadmat

#subject = 'CT01'

def make_out_dir(out_path):

	#Make subdirectories to save files
	filename = out_path
	if not os.path.exists(os.path.dirname(filename)):
	    try:
	        os.makedirs(os.path.dirname(filename))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	          raise


def get_df(mat_path):
	#Reads a .mat file obtained from diffparc and conver it to a panda df
	#mat_path = path to the .mat file

	file_mat = loadmat(mat_path)
	conn = file_mat['connmap_feats'] #Extracting the table named connmap_feat from the .mat file
	df = pd.DataFrame(conn.T)  #Transposing and converting to a df

	return df


def split_df(df,headsize):
	#Splits a df into two parts. This is to seperate left and right from the df.

	left = df.head(headsize)
	right = df.tail(len(df)-headsize)

	return left, right


def get_gradients(matrix):
	#Calculating the diffusion gradients
	#matrix: nxm matrix
	#gradient_n: gradient componanat (int 1 to n). I have set n = 5 here.

	sim_matrix = cosine_similarity(matrix)  #Calculating the cosine similarity matrix
	#sim_matrix = matrix.to_numpy()

	#plt.imshow(sim_matrix)
	#plt.show()

	de = DiffusionMapEmbedding(alpha=0.5, diffusion_time=1, affinity='markov', n_components=5).fit_transform(sim_matrix.copy())

	#de = GradientMaps(alignment=None, approach='dm', kernel = None , n_components=5, random_state=0)
	#de.fit(sim_matrix.copy())
	#grad=de[:,gradient_n-1] #-1 to fix for the index
	#print(de.gradients_())

	#return de.gradients_	
	return de



def get_diffusion_maps(path_to_mat_R,path_to_mat_L, componanat):
	#Calculate the diffusion gradients for LEft and Right sides.
	#subject: string of the subject ID
	#componanat: int (1-5). Can change the upper limit (5) by editing get_gradinets().

	Right_matrix = glob.glob(path_to_mat_R) #finding all the mat files
	Left_matrix = glob.glob(path_to_mat_L)
	#Left_matrix = glob.glob("../data/matrices/Left/"+'*'+subject+'*'+'.mat') #finding all the mat files

	R_matrix = get_df(Right_matrix[0])
	L_matrix= get_df(Left_matrix[0])

	left,true_right = split_df(R_matrix, 180)
	true_left,right = split_df(L_matrix, 180)

	#plt.imshow(true_right*1000,aspect="auto")
	#plt.show()

	#Transposing the matrix to get the striatum gradients.
	true_right = true_right.T
	true_left = true_left.T

	R_gradient = get_gradients(true_right)
	L_gradient = get_gradients(true_left)

	np.save(snakemake.output.R_emb, R_gradient)
	np.save(snakemake.output.L_emb, L_gradient)

	print("Right gradient is processed for shape:",np.shape(R_gradient))
	#print("Left gradient is processed for "+subject+" with shape:",np.shape(L_gradient))

	grad_df = pd.DataFrame({'R_grad_1': R_gradient[:,0], 'R_grad_2': R_gradient[:,1],
			'R_grad_3': R_gradient[:,2], 'R_grad_4': R_gradient[:,3], 'L_grad_1': L_gradient[:,0],
			 'L_grad_2': L_gradient[:,1], 'L_grad_3': L_gradient[:,2], 'L_grad_4': L_gradient[:,3]})

	return grad_df



grad_df = get_diffusion_maps(snakemake.input.Right,snakemake.input.Left,1)

make_out_dir(snakemake.input.gradient_path)

grad_df.to_csv(snakemake.output.gradient, index=False)

#np.savetxt(snakemake.output.gradient,R_gradient)





"""
def get_projections():

  img = load_img("HCP-MMP1.nii.gz")
  data = (img.get_data())
  x = np.shape(data)[0]
  y = np.shape(data)[1]
  z = np.shape(data)[2]
  R_rois = np.loadtxt("R_coords.txt")
  L_rois = np.loadtxt("L_coords.txt")
  #rois = coords.values
  R_vals = np.column_stack((R_gradient, R_rois))
  L_vals = np.column_stack((L_gradient, L_rois))


  new_img = data
  for i in range(x):
     for j in range (y):
             for k in range (z):
                 if data[i,j,k] in R_vals[:,1].tolist():
                     index = np.where(R_vals[:,1] == data[i,j,k])[0][0]
                     #print(val.index(data[i,j,k]))
                     new_img[i,j,k]=R_vals[index,0]*1000

                 elif data[i,j,k] in L_vals[:,1].tolist():
                     index = np.where(L_vals[:,1] == data[i,j,k])[0][0]
                     #print(val.index(data[i,j,k]))
                     new_img[i,j,k]=L_vals[index,0]*1000
                 else:
                     new_img[i,j,k]=3000

  final_img = nib.Nifti1Image(new_img, img.affine, img.header)
  nib.save(final_img,"final_image.nii")


get_projections()"""


