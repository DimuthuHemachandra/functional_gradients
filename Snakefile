#!/usr/bin/env python
from os.path import join
from glob import glob
import pandas as pd

configfile: 'cfg/config.yml'


#load participants.tsv file, and strip off sub- from participant_id column
df = pd.read_table(config['participants_tsv'])
subjects = df.participant_id.to_list() 
subjects = [ s.strip('sub-') for s in subjects ]

print(subjects)

componants=['1','2','3']
hemis=['LH','RH']
sessions=['Month12','Month24']

wildcard_constraints:
    subject="[a-zA-Z0-9]+",
    componant="[a-zA-Z0-9]+",
    hemi="[a-zA-Z0-9]+",
    ses="[a-zA-Z0-9]+"




diffparc_dir = config['diffparc_dir']

rule all:
    input: 
        cleaned = expand(config['cleaned_dir']+'sub-{subject}_ses-{ses}_cleaned.dtseries.nii',subject=subjects,ses=sessions),
        smoothed = expand(config['analysis_dir']+'processed_dtseries/sub-{subject}_ses-{ses}_cleaned_smoothed.dtseries.nii',subject=subjects,ses=sessions),
        cor_matrix = expand(config['analysis_dir']+'corr_matrix/sub-{subject}_ses-{ses}_corr-matrix.npy',subject=subjects,ses=sessions),
    	grad = expand([config['analysis_dir']+'gradients/bs_emb/emb_ctx_{ses}.pickle', config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_{ses}.pickle', 
        config['analysis_dir']+ 'gradients/bs_emb/emb_sbctx_{hemi}_{ses}.pickle', config['analysis_dir']+'gradients/bs_emb/emb_sbctx_{hemi}_{ses}.pickle',
        config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_{hemi}_{ses}.pickle', config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_{hemi}_{ses}.pickle'],ses=sessions,hemi=hemis),
        ctx = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_ses-{ses}_ctx_{hemi}_aligned_grad.func.gii',subject=subjects,ses=sessions,hemi=hemis),
        mean_ctx = expand(config['analysis_dir']+'gradients/projections/cortex/mean_{ses}_ctx_{hemi}_grad.func.gii',ses=sessions,hemi=hemis),
        sbctx = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_ses-{ses}_sbctx_{hemi}_aligned_grad{componant}_image.dscalar.nii',subject=subjects,ses=sessions,hemi=hemis,componant=componants),
        mean_sbctx = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/mean_{ses}_sbctx_{hemi}_grad{componant}_image.dscalar.nii',ses=sessions,hemi=hemis,componant=componants),
        nii_all_files = expand(config['analysis_dir']+'gradients/projections/sbctx/nii/sub-{subject}_ses-{ses}_sbctx_{hemi}_aligned_grad{componant}_image.nii',subject=subjects, componant=componants, hemi=hemis, ses=sessions),
        diff_grads = expand(config['analysis_dir']+'stats/grad1/LH/sub-{subject}_sbctx_LH_grad1_diff.nii.gz',subject=subjects),
        concated_4d = config['analysis_dir']+'stats/grad1/LH/concatenated_diff_4D.nii'


    resources: 
            mem_mb = 8000, 
            time = 400, #30 mins
        


#Cleaning fmri data
####################################################################################
rule clean_dtseries: 
    input: bold = config['fmriprep_dir'],
    	   tsv = config['fmriprep_tsv']
           
    output: cleaned = join(config['cleaned_dir'],'sub-{subject}_ses-{ses}_cleaned.dtseries.nii')

    params: out_path = config['cleaned_dir'],
            ciftify_container = config['ciftify_container']

    group: 'preprocess'
    shell: 'mkdir -p {params.out_path} && singularity exec {params.ciftify_container} ciftify_clean_img --output-file={output.cleaned} \
            --confounds-tsv={input.tsv} --clean-config=cfg/cleaning_ciftify.json {input.bold}'



#smoothing 
####################################################################################

rule smoothed_dtseries: 
    input: cleaned_dts = join(config['cleaned_dir'], 'sub-{subject}_ses-{ses}_cleaned.dtseries.nii'),
           left_surface = config['left-surface'],
	       right_surface = config['right-surface']

    params: out_path = config['analysis_dir']+'processed_dtseries/'
           
    output: smoothed_dts = join(config['analysis_dir'],'processed_dtseries/sub-{subject}_ses-{ses}_cleaned_smoothed.dtseries.nii')


    group: 'preprocess'
    shell: 'mkdir -p {params.out_path} && wb_command -cifti-smoothing {input.cleaned_dts} 2.55 2.55 COLUMN {output.smoothed_dts} -left-surface {input.left_surface} -right-surface  {input.right_surface} -fix-zeros-volume -fix-zeros-surface -merged-volume'




#Calculating connectivity matrices
####################################################################################
rule get_corr_matrix_12: 
    input: cleaned_bold = join(config['analysis_dir'],'processed_dtseries/sub-{subject}_ses-{ses}_cleaned_smoothed.dtseries.nii')

    params: out_path = join(config['analysis_dir'],'corr_matrix/'),
            atlas_labels = config['atlas_label'],
            atlas_labels_txt = config['atlas_label_txt'],
            greyordinates = config['greyordinates']
           
    output: corr_matrix = join(config['analysis_dir'],'corr_matrix/sub-{subject}_ses-{ses}_corr-matrix.npy')

    group: 'participant'

    script: 'scripts/get_correlation.py'



#Calculating gradients
####################################################################################
rule get_gradients: 
    input: matrix_files_12 = expand(config['analysis_dir']+'corr_matrix/sub-{subject}_ses-Month12_corr-matrix.npy',subject=subjects),
           matrix_files_24 = expand(config['analysis_dir']+'corr_matrix/sub-{subject}_ses-Month24_corr-matrix.npy',subject=subjects)

    params: subjects = subjects,
            grad_path = config['analysis_dir']+'gradients/bs_emb/'
           
    output: grad_ctx_12 = config['analysis_dir']+'gradients/bs_emb/emb_ctx_Month12.pickle',
            grad_sbctx_L_12 = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_LH_Month12.pickle',
            grad_sbctx_R_12 = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_RH_Month12.pickle',
            aligned_grad_ctx_12 = config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_Month12.pickle',
            aligned_grad_sbctx_L_12 = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_LH_Month12.pickle',
            aligned_grad_sbctx_R_12 = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_RH_Month12.pickle',
            grad_ctx_24 = config['analysis_dir']+'gradients/bs_emb/emb_ctx_Month24.pickle',
            grad_sbctx_L_24 = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_LH_Month24.pickle',
            grad_sbctx_R_24 = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_RH_Month24.pickle',
            aligned_grad_ctx_24 = config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_Month24.pickle',
            aligned_grad_sbctx_L_24 = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_LH_Month24.pickle',
            aligned_grad_sbctx_R_24 = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_RH_Month24.pickle'
            
    resources: 
            mem_mb = 8000, 
            time = 400, #30 mins

    group: 'group'
    script: 'scripts/gradients_new.py'



#Projections
####################################################################################
rule get_projection_month12: 
    input: grad_ctx = config['analysis_dir']+'gradients/bs_emb/emb_ctx_Month12.pickle',
           aligned_grad_ctx = config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_Month12.pickle',

    params: month = 'Month12',
            subjects = subjects,
            out_path = config['analysis_dir']+'gradients/projections/',
            labels = config['schaefer_1000_labels']
          
    output: ctx_L = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_ses-Month12_ctx_LH_aligned_grad.func.gii',subject=subjects),
            ctx_R = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_ses-Month12_ctx_RH_aligned_grad.func.gii',subject=subjects), 
            mean_ctx_L = config['analysis_dir']+'gradients/projections/cortex/mean_Month12_ctx_LH_grad.func.gii',
            mean_ctx_R = config['analysis_dir']+'gradients/projections/cortex/mean_Month12_ctx_RH_grad.func.gii'

    group: 'group'
               
    script: 'scripts/projection_ctx.py'


rule get_projection_month24: 
    input: grad_ctx = config['analysis_dir']+'gradients/bs_emb/emb_ctx_Month24.pickle',
           aligned_grad_ctx = config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_Month24.pickle',

    params: month = 'Month24',
            subjects = subjects,
            out_path = config['analysis_dir']+'gradients/projections/',
            labels = config['schaefer_1000_labels']
          
    output: ctx_L = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_ses-Month24_ctx_LH_aligned_grad.func.gii',subject=subjects),
            ctx_R = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_ses-Month24_ctx_RH_aligned_grad.func.gii',subject=subjects), 
            mean_ctx_L = config['analysis_dir']+'gradients/projections/cortex/mean_Month24_ctx_LH_grad.func.gii',
            mean_ctx_R = config['analysis_dir']+'gradients/projections/cortex/mean_Month24_ctx_RH_grad.func.gii'

    group: 'group'
               
    script: 'scripts/projection_ctx.py'


rule get_sbctx_projection_month12: 
    input: mean_grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_LH_Month12.pickle',
           mean_grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_RH_Month12.pickle',
           aligned_grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_LH_Month12.pickle',
           aligned_grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_RH_Month12.pickle'

    params: month = 'Month12',
            subjects = subjects,
            out_path = config['analysis_dir']+'gradients/projections/',
            labels = config['schaefer_1000_labels'],
            greyordinates = config['greyordinates']
          
    output: sbctx_L = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_ses-Month12_sbctx_LH_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants),
            sbctx_R = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_ses-Month12_sbctx_RH_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants),
            sbctx_mean_L = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/mean_Month12_sbctx_LH_grad{componant}_image.dscalar.nii',componant=componants),
            sbctx_mean_R = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/mean_Month12_sbctx_RH_grad{componant}_image.dscalar.nii',componant=componants)
            

    group: 'group'

    resources: time = 60, #30 mins
               
    script: 'scripts/projection_sbctx.py'

rule get_sbctx_projection_month24: 
    input: mean_grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_LH_Month24.pickle',
           mean_grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_RH_Month24.pickle',
           aligned_grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_LH_Month24.pickle',
           aligned_grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_RH_Month24.pickle'

    params: month = 'Month24',
            subjects = subjects,
            out_path = config['analysis_dir']+'gradients/projections/',
            labels = config['schaefer_1000_labels'],
            greyordinates = config['greyordinates']
          
    output: sbctx_L = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_ses-Month24_sbctx_LH_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants),
            sbctx_R = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_ses-Month24_sbctx_RH_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants),
            sbctx_mean_L = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/mean_Month24_sbctx_LH_grad{componant}_image.dscalar.nii',componant=componants),
            sbctx_mean_R = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/mean_Month24_sbctx_RH_grad{componant}_image.dscalar.nii',componant=componants)
            

    group: 'group'

    resources: time = 60, #30 mins
               
    script: 'scripts/projection_sbctx.py'


rule cii_to_nii: 
    input: nii_files = config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_ses-{ses}_sbctx_{hemi}_aligned_grad{componant}_image.dscalar.nii'

    params: out_path = config['analysis_dir']+'gradients/projections/sbctx/nii/',
            
            
    output: nii_out = config['analysis_dir']+'gradients/projections/sbctx/nii/sub-{subject}_ses-{ses}_sbctx_{hemi}_aligned_grad{componant}_image.nii'

    group: 'group_2'

    resources: time = 60, #30 mins
               
    shell: 'mkdir -p {params.out_path} && wb_command -cifti-separate {input.nii_files} COLUMN -volume-all {output.nii_out}'


rule t_test:
    input: nii_files = expand(config['analysis_dir']+'gradients/projections/sbctx/nii/sub-{subject}_ses-{ses}_sbctx_{hemi}_aligned_grad{componant}_image.nii',subject=subjects, componant=componants, hemi=hemis, ses=sessions)
    
    params: out_path = config['analysis_dir']+'stats/'

    output: diff_grad = expand(config['analysis_dir']+'stats/grad1/LH/sub-{subject}_sbctx_LH_grad1_diff.nii.gz',subject=subjects), 
            nii_4D = config['analysis_dir']+'stats/grad1/LH/concatenated_diff_4D.nii',

    group: 'stat'

    resources: 
            mem_mb = 8000, 
            time = 240, #mins   
               
    script: 'scripts/t_stat.py'



            



    
  



    
