from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_group_fc, load_parcellation
import matplotlib.pyplot as plt
from brainspace.gradient import GradientMaps
import numpy as np
from brainspace.utils.parcellation import map_to_labels
import scipy.io as sio
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nilearn.image import load_img
import nibabel as nib
from nilearn import input_data
from nilearn.image import resample_to_img
from nilearn.image import math_img

from mapalign.embed import DiffusionMapEmbedding


data_path = "../BrainSpace/test_materials/"

#matrix = pd.read_csv('test.csv', header=None)
R_matrix = pd.read_csv('right_conn.csv', header=None)
L_matrix = pd.read_csv('left_conn.csv', header=None)




def split_df(df,headsize):

	left = df.head(headsize)
	right = df.tail(len(df)-headsize)

	return left, right

left,true_right = split_df(R_matrix, 180)
true_left,right = split_df(L_matrix, 180)

plt.imshow(true_right*1000,aspect="auto")
plt.show()

def get_gradients(matrix, gradient_n):

	sim_matrix = cosine_similarity(matrix)

	print(np.shape(sim_matrix))


	plt.imshow(sim_matrix)
	plt.show()

	de = DiffusionMapEmbedding(alpha=0.5, diffusion_time=10, affinity='markov', n_components=4).fit_transform(sim_matrix.copy())

	grad=de[:,gradient_n-1] #-1 to fix for the index

	return grad

#Transposing the matrix to get gradients for the striatum.
true_right = true_right.T
true_left = true_left.T

R_gradient = get_gradients(true_right, 1)
L_gradient = get_gradients(true_left, 1)

#print(np.shape(R_gradient))
#print(np.shape(L_gradient))


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



def get_vals(side):

	img = nib.load(data_path+'striatum_'+side+'.nii.gz')
	img_data = img.get_fdata()

	"""mask = nib.load(data_path+'/diffusion_parcellation.masked.normMax.nii.gz')

	mask_data = mask.get_fdata()

	masker = input_data.NiftiMasker()
	masker.fit(mask)
	#plot_roi(masker.mask_img_, title='EPI automatic mask')
	#show()
	masked_img = masker.mask_img_

	resampled_mask = resample_to_img(masked_img,img,interpolation= "nearest")


	result_img = math_img("img1 + img2",img1=resampled_mask, img2=img)

	#plot_roi(result_img)
	#show()

	img_data = result_img.get_fdata()"""
	data=img_data.flatten()

	#plt.plot(data)
	#plt.show()

	res = [val for idx, val in enumerate(data) if val > 1.5] 
	
	print(np.size(res))

	return img,img_data,res


def grad_to_image(data_cube,mask_image,grad):

	#This method gets an 3D data cube and a Data cube of a mask and also a gradient
	#and fills the Data cube with gradinet values using the mask

	dimentions = np.shape(mask_image)
	print(dimentions)
	l=0

	for i in range(dimentions[0]):
		for j in range(dimentions[1]):
			for k in range(dimentions[2]):
				if mask_image[i,j,k] >1.5:
					data_cube[i,j,k]=grad[l]
					print(l)
					l = l+1

	return data_cube


img,R_data_cube,val = get_vals('right')

fixed_data_cube = R_data_cube  #This fix image is defined from the right side, but will be also used for the left side to add both gradients

data_cube = grad_to_image(fixed_data_cube,R_data_cube,R_gradient)

img,L_data_cube,val = get_vals('left')

data_cube = grad_to_image(fixed_data_cube,L_data_cube,L_gradient)

plt.show()

new_img = nib.Nifti1Image(data_cube, img.affine, img.header)
nib.save(new_img, 'new_image.nii')








