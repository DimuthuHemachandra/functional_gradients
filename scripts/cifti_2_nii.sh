#!/bin/bash

nii_dir=$1
out_file=$2

for cii in `ls ~/scratch/PPMI_project2/derivatives/analysis/smoothed/plots/sbctx_nii/*image.dscalar.nii`

	do 
	prefix="image.dscalar.nii"
	new_name=${cii/%$prefix}

    wb_command -cifti-separate $cii COLUMN -volume-all ${new_name}3d.nii.gz

	#img=/home/dimuthu1/scratch/project2/derivatives/diffparc_3T_nodistcorr/work/${labels}/bedpost.CIT168_striatum_cortical/seed_dtires.nii.gz
	#compute ROIs for MNI left/right hemi
	#dimx=`fslval ${img} dim1`
	#halfx=$(echo "scale=0; $dimx/2" | bc)


	#roi_left="0 ${halfx} 0 -1 0 -1 0 -1"
	#roi_right="$(($halfx+1)) -1 0 -1 0 -1 0 -1"

	#echo "roi_left $roi_left"
	#echo "roi_right $roi_right"

	#mask with left/right  ROI:
	#fslmaths $img -roi $roi_left /home/dimuthu1/scratch/project2/derivatives/diffparc_3T_nodistcorr/work/${labels}/bedpost.CIT168_striatum_cortical/seed_dtires_striatum_lh.nii.gz
	#fslmaths $img -roi $roi_right /home/dimuthu1/scratch/project2/derivatives/diffparc_3T_nodistcorr/work/${labels}/bedpost.CIT168_striatum_cortical/seed_dtires_striatum_rh.nii.gz


	done