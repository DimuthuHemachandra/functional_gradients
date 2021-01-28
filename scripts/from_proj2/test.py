from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_group_fc, load_hcp_parcellation, hcp360, load_parcellation
import matplotlib.pyplot as plt
from brainspace.gradient import GradientMaps
import numpy as np
from brainspace.utils.parcellation import map_to_labels
import nibabel as nib
from nilearn import surface
import pandas as pd




surf_lh, surf_rh = hcp360()

labeling = load_hcp_parcellation('hcp', scale=360, join=True)

print(np.shape(labeling))

P = plot_hemispheres(surf_lh, surf_rh, array_name=labeling, size=(1200, 200), cmap='tab20', zoom=1.85)

P.screenshot('test.png')

"""hcp_grad = pd.read_csv("gradients.csv")

mask = labeling != 0
grad = [None] * 4
for i in range(4):
	
    L_grad = hcp_grad['L_grad_'+str(i+1)].tolist()
    R_grad = hcp_grad['R_grad_'+str(i+1)].tolist()
    grad_LR = L_grad + R_grad
    # map the gradient to the parcels
    grad[i] = map_to_labels(np.array(grad_LR), labeling, mask=mask, fill=np.nan)


plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 600), label_text=['Grad1', 'Grad2', 'Grad3', 'Grad4'])"""





#grad = map_to_labels(, labeling, mask=mask, fill=np.nan)
#plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 600))

#print(np.shape(grad_1))

#surf_lh, surf_rh = load_conte69()

#print(dir(surf_lh))
"""

# Load left and right hemispheres
surf_lh, surf_rh = load_conte69()
#print(dir(surf_lh))

#plot_hemispheres(surf_lh, surf_rh, size=(800, 200))

labeling = load_parcellation('schaefer', scale=400, join=True)
print(labeling)
m = load_group_fc('schaefer', scale=400)

print(np.shape(labeling))


plt.imshow(m)
plt.show()

"""
labeling = load_parcellation('schaefer', scale=400, join=True)
print(labeling)
m = load_group_fc('schaefer', scale=400)

# Build gradients using diffusion maps and normalized angle
gm = GradientMaps(n_components=10, approach='dm', kernel='normalized_angle')

# and fit to the data
gm.fit(m)
GradientMaps(alignment=None, approach='dm', kernel='normalized_angle', n_components=10, random_state=None)

# The gradients are in
g1 = gm.gradients_[:, 0]
print(g1.shape)
#plt.imshow(g1)
#plt.show()

#print(g1)

print(np.shape(gm.gradients_[:, 1]))

mask = labeling != 0

grad = [None] * 4
for i in range(4):
    # map the gradient to the parcels
    grad[i] = map_to_labels(gm.gradients_[:, i], labeling, mask=mask, fill=np.nan)
# Plot first gradient on the cortical surface.
#plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 600), label_text=['Grad1', 'Grad2', 'Grad3', 'Grad4'])

#plt.scatter(range(gm.lambdas_.size), gm.lambdas_)
#plt.show()

"""

img=nib.load('brainspace/datasets/surfaces/mmpL.func.gii')

img_data_L = [x.data for x in img.darrays]

img=nib.load('brainspace/datasets/surfaces/mmpR.func.gii')

img_data_R = [x.data for x in img.darrays]

#img_data = img_data_L + img_data_R

for x in img_data_R:
  img_data_L.append(x)

print(np.shape(img_data_L))

#random.shuffle(img_data)

#vertices = np.size(img_data)
#img_data = img_data.reshape(1,-1)
img_data = np.array(img_data_L).flatten()

np.savetxt("hcp_360_hcp360.csv", img_data.T, delimiter=",")

#new_list = np.ones(vertices)
#new_list = new_list.reshape(1,-1)



#new_list = new_list.tolist()
#print(np.shape(new_list))
#print(new_list)

#print(img_data[0])
print(np.shape(img.darrays[0].data))


img.darrays[0].data=new_list
print(np.shape(img.darrays[0].data))

nib.save(img,"lh.final.label.gii")

#print(dir(curv))

#curv=nib.load('mmpL.func.gii')

#print(curv.agg_data())

#header = curv.header

#print(header)

#print(curv.get_fdata())

#print(curv.labeltable.labels.count())
#print(', '.join("%s: %s" % item for item in attrs.items()))

#print(curv.labeltable.labels)

#print(surface.load_surf_mesh('mmpL.func.gii'))"""
