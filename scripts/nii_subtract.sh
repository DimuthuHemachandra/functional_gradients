#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=24:00:00
#SBATCH --job-name=transform
#SBATCH --account=ctb-akhanf
#SBATCH --output=/scratch/dimuthu1/PPMI_project2/jobs/fmriprep.%A.out

nii_path='/home/dimuthu1/scratch/PPMI_project2/derivatives/analysis/smoothed/plots/sbctx_nii'

for labels in `sed '1d'  participants.tsv`
	do 
	
	
	ID=${labels##*-}
	echo $ID
	
	file_A=${nii_path}/${ID}-m12_sbctx_R_grad_3_3d.nii.gz
	file_B=${nii_path}/${ID}-m24_sbctx_R_grad_3_3d.nii.gz
	
	fslmaths ${file_B} -sub ${file_A} ${nii_path}/${ID}_diff
	#hcp_nii_file=~/scratch/project2/derivatives/hcp_360/work/MyWorkflow/_sub_id_${ID}/parce/out_put/${ID}/HCP-MMP1.nii.gz
	#destination=~/scratch/project2/derivatives/diffparc_3T_nodistcorr/work/${labels}/labels/t1/HCP_native
	#mkdir -p $destination
	#cp $hcp_nii_file $destination
	done

