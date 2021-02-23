import pandas as pd
import os
import glob
import io 
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as sns
from scipy.stats import zscore
from brainspace.utils.parcellation import reduce_by_labels, map_to_mask
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_parcellation
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels
from sklearn.metrics.pairwise import cosine_similarity

emb_dir ='/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/gradients/bs_emb'

emb = np.load(emb_dir+"/emb_ctx_12.npy", allow_pickle=True)

print(emb)