#!/bin/bash

sub=sub-${1}
myelin_img=$2
hcp_img=$3
hcp_labels=$4

out_path=$5

flirt -in $myelin_img -ref $hcp_img -interp nearestneighbour -out $out_path/lin_registered.nii.gz

#Removing the existing files.
file="$out_path/sub-${1}_mean_myelin_L.txt"

if [ -f $file ] ; then
    rm $file
fi

file="$out_path/sub-${1}_mean_myelin_R.txt"

if [ -f $file ] ; then
    rm $file
fi

#Looping through all the HCP labels

for labels in `cut -f1 <  $hcp_labels`
	do 
		echo $labelsL
		
		
		#fnirt --in=lin_registered.nii.gz --ref=HCP-MMP1.nii.gz --iout=$out_path/nonlin_registered.nii.gz

		fslmaths $hcp_img -thr $labels -uthr $labels $out_path/thersh_image.nii.gz
		fslmaths $out_path/lin_registered.nii.gz -mas $out_path/thersh_image.nii.gz $out_path/final_1_image.nii.gz
		#fslmaths nonlin_registered.nii.gz -mas thersh_image.nii.gz final_2_image.nii.gz
		#fslstats ${out_path}/final_1_image.nii.gz -M >> $out_path/${1}_mean_myelin.txt
		#echo $val

		if [ $labels -gt 1000 -a $labels -lt 1181 ]
		then
		fslstats ${out_path}/final_1_image.nii.gz -M >> $out_path/sub-${1}_mean_myelin_L.txt
		fi
		if [ $labels -gt 2001 -a $labels -lt 2182 ]
		then
		fslstats ${out_path}/final_1_image.nii.gz -M >> $out_path/sub-${1}_mean_myelin_R.txt
		fi
		

done
