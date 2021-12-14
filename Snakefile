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
hemis=['L','R']
sessions=['month12','month24']

wildcard_constraints:
    subject="[a-zA-Z0-9]+",
    componanat="[a-zA-Z0-9]+",
    hemi="[a-zA-Z0-9]+",
    ses="[a-zA-Z0-9]+"




diffparc_dir = config['diffparc_dir']

rule all:
    input: 
        #cleaned_12 = expand(config['cleaned_dir']+'sub-{subject}_Month12_clean.dtseries.nii',subject=subjects),
        #cleaned_24 = expand(config['cleaned_dir']+'sub-{subject}_Month24_clean.dtseries.nii',subject=subjects),
        #smoothed_12 = expand(config['analysis_dir']+'processed_dtseries/sub-{subject}_Month12_clean_smoothed.dtseries.nii',subject=subjects),
        #smoothed_24 = expand(config['analysis_dir']+'processed_dtseries/sub-{subject}_Month24_clean_smoothed.dtseries.nii',subject=subjects),
        #cor_matrix_12 = expand(config['analysis_dir']+'corr_matrix/month12/sub-{subject}_month12_corr-matrix.npy',subject=subjects),
        #cor_matrix_24 = expand(config['analysis_dir']+'corr_matrix/month24/sub-{subject}_month24_corr-matrix.npy',subject=subjects),
    	##grad_12 = [config['analysis_dir']+'gradients/bs_emb/emb_ctx_month12.pickle', config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_month12.pickle', 
        ##config['analysis_dir']+ 'gradients/bs_emb/emb_sbctx_L_month12.pickle', config['analysis_dir']+'gradients/bs_emb/emb_sbctx_R_month12.pickle',
        ##config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_L_month12.pickle', config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_R_month12.pickle'],
        ##grad_24 = [config['analysis_dir']+'gradients/bs_emb/emb_ctx_month24.pickle', config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_month24.pickle', 
        ##config['analysis_dir']+'gradients/bs_emb/emb_sbctx_L_month24.pickle', config['analysis_dir']+'gradients/bs_emb/emb_sbctx_R_month24.pickle',
        ##config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_L_month24.pickle', config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_R_month24.pickle'],
        ##ctx_L_12 = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month12_ctx_L_aligned_grad.func.gii',subject=subjects),
        ##ctx_R_12 = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month12_ctx_R_aligned_grad.func.gii',subject=subjects),
        ##mean_ctx_L_12 = [config['analysis_dir']+'gradients/projections/cortex/mean_month12_ctx_L_grad.func.gii',
        ##config['analysis_dir']+'gradients/projections/cortex/mean_month12_ctx_R_grad.func.gii'],
        ##ctx_L_24 = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month24_ctx_L_aligned_grad.func.gii',subject=subjects),
        ##ctx_R_24 = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month24_ctx_R_aligned_grad.func.gii',subject=subjects),
        ##mean_ctx_L_24 = [config['analysis_dir']+'gradients/projections/cortex/mean_month24_ctx_L_grad.func.gii',
        ##config['analysis_dir']+'gradients/projections/cortex/mean_month24_ctx_R_grad.func.gii'],
        ##project_sbctx_grads_12 = [expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_month12_sbctx_R_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants),
        ##expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_month12_sbctx_R_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants)],
        ##project_sbctx_grads_24 = [expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_month24_sbctx_R_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants),
        ##expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_month24_sbctx_R_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants)],
        ##nii_all_files = expand(config['analysis_dir']+'gradients/projections/sbctx/nii/sub-{subject}_{ses}_sbctx_{hemi}_aligned_grad{componant}_image.nii',subject=subjects, componant=componants, hemi=hemis, ses=sessions),
        diff_grads = expand(config['analysis_dir']+'gradients/projections/sbctx/ttest/grad3/Rh/sub-{subject}_sbctx_R_grad3_diff.nii.gz',subject=subjects)
        #diff_grads = expand(config['analysis_dir']+'gradients/projections/sbctx/ttest/grad{componant}/{hemi}h/sub-{subject}_sbctx_{hemi}_grad{componant}_diff.nii.gz',subject=subjects, componant=componants, hemi=hemis)
        #nii_4D_files = config['analysis_dir']+'gradients/projections/sbctx/nii/concatednated_4D.nii'
        #mean_sbctx_L_12 = [config['analysis_dir']+'gradients/projections/sbctx/mean_month12_sbctx_L_grad3_image.dscalar.nii',
        #config['analysis_dir']+'gradients/projections/sbctx/mean_month12_sbctx_R_grad3_image.dscalar.nii']
        #project_ctx_grads_24 = expand(config['analysis_dir']+'gradients/projections/sbctx/sub-{subject}_month24_sbctx_R_aligned_grad3_image.dscalar.nii',subject=subjects),
        #project_ctx_grads_124 = expand(config['analysis_dir']+'gradients/projections/sbctx/sub-{subject}_month12_sbctx_L_aligned_grad3_image.dscalar.nii',subject=subjects),
        #project_ctx_grads_24 = [expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month24_ctx_L_aligned_grad.func.gii',subject=subjects), 
        #expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month24_ctx_R_aligned_grad.func.gii',subject=subjects),
        #config['analysis_dir']+'gradients/projections/cortex/month24_ctx_L_mean_grad.func.gii', config['analysis_dir']+'gradients/projections/cortex/month24_ctx_R_mean_grad.func.gii']


        
"""

#Cleaning fmri data
####################################################################################
rule clean_dtseries_m12: 
    input: bold = config['fmriprep_m12_dir'],
    	   tsv = config['fmriprep_m12_tsv']
           
    output: cleaned = join(config['cleaned_dir'],'sub-{subject}_Month12_clean.dtseries.nii')

    params: out_path = config['cleaned_dir'],
            ciftify_container = config['ciftify_container']

    group: 'preprocess'
    shell: 'mkdir -p {params.out_path} && singularity exec {params.ciftify_container} ciftify_clean_img --output-file={output.cleaned} \
            --confounds-tsv={input.tsv} --clean-config=cfg/cleaning_ciftify.json {input.bold}'

rule clean_dtseries_m24: 
    input: bold = config['fmriprep_m24_dir'],
    	   tsv = config['fmriprep_m24_tsv']
           
    output: cleaned = join(config['cleaned_dir'],'sub-{subject}_Month24_clean.dtseries.nii')

    params: out_path = config['cleaned_dir'],
            ciftify_container = config['ciftify_container']

    group: 'preprocess'
    shell: 'mkdir -p {params.out_path} && singularity exec {params.ciftify_container} ciftify_clean_img --output-file={output.cleaned} \
            --confounds-tsv={input.tsv} --clean-config=cfg/cleaning_ciftify.json {input.bold}'

#smoothing 
####################################################################################

rule smoothed_dtseries_m12: 
    input: cleaned_dts = join(config['cleaned_dir'], 'sub-{subject}_Month12_clean.dtseries.nii'),
           left_surface = config['left-surface'],
	       right_surface = config['right-surface']

    params: out_path = config['analysis_dir']+'processed_dtseries/'
           
    output: smoothed_dts = join(config['analysis_dir'],'processed_dtseries/sub-{subject}_Month12_clean_smoothed.dtseries.nii')


    group: 'preprocess'
    shell: 'mkdir -p {params.out_path} && wb_command -cifti-smoothing {input.cleaned_dts} 2.55 2.55 COLUMN {output.smoothed_dts} -left-surface {input.left_surface} -right-surface  {input.right_surface} -fix-zeros-volume -fix-zeros-surface'

rule smoothed_dtseries_m24: 
    input: cleaned_dts = join(config['cleaned_dir'], 'sub-{subject}_Month24_clean.dtseries.nii'),
           left_surface = config['left-surface'],
	       right_surface = config['right-surface']

    params: out_path = config['analysis_dir']+'processed_dtseries/'

    output: smoothed_dts = join(config['analysis_dir'],'processed_dtseries/sub-{subject}_Month24_clean_smoothed.dtseries.nii')

    group: 'preprocess'
    shell: 'mkdir -p {params.out_path} && wb_command -cifti-smoothing {input.cleaned_dts} 2.55 2.55 COLUMN {output.smoothed_dts} -left-surface {input.left_surface} -right-surface  {input.right_surface} -fix-zeros-volume -fix-zeros-surface'




#Calculating connectivity matrices
####################################################################################
rule get_corr_matrix_12: 
    input: cleaned_bold = join(config['analysis_dir'],'processed_dtseries/sub-{subject}_Month12_clean_smoothed.dtseries.nii')

    params: out_path = join(config['analysis_dir'],'corr_matrix/month12')
           
    output: corr_matrix = join(config['analysis_dir'],'corr_matrix/month12/sub-{subject}_month12_corr-matrix.npy')

    group: 'participant'

    script: 'scripts/get_correlation.py'
    
rule get_corr_matrix_24: 
    input: cleaned_bold = join(config['analysis_dir'],'processed_dtseries/sub-{subject}_Month24_clean_smoothed.dtseries.nii')

    params: out_path = join(config['analysis_dir'],'corr_matrix/month24')
           
    output: corr_matrix = join(config['analysis_dir'],'corr_matrix/month24/sub-{subject}_month24_corr-matrix.npy')

    group: 'participant'

    script: 'scripts/get_correlation.py'



#Calculating gradients
####################################################################################
rule get_gradients_month12: 
    input: matrix_files = expand(config['analysis_dir']+'corr_matrix/month12/sub-{subject}_month12_corr-matrix.npy',subject=subjects)

    params: month = 'month12',
            subjects = subjects,
            grad_path = config['analysis_dir']+'gradients/bs_emb/'
           
    output: grad_ctx = config['analysis_dir']+'gradients/bs_emb/emb_ctx_month12.pickle',
            grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_L_month12.pickle',
            grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_R_month12.pickle',
            aligned_grad_ctx = config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_month12.pickle',
            aligned_grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_L_month12.pickle',
            aligned_grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_R_month12.pickle'


    group: 'group'
    script: 'scripts/gradients_new.py'

rule get_gradients_month24: 
    input: matrix_files = expand(config['analysis_dir']+'corr_matrix/month24/sub-{subject}_month24_corr-matrix.npy',subject=subjects)

    params: month = 'month24',
            subjects = subjects,
            grad_path = config['analysis_dir']+'gradients/bs_emb/'
           
    output: grad_ctx = config['analysis_dir']+'gradients/bs_emb/emb_ctx_month24.pickle',
            grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_L_month24.pickle',
            grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_R_month24.pickle',
            aligned_grad_ctx = config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_month24.pickle',
            aligned_grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_L_month24.pickle',
            aligned_grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_R_month24.pickle'


    group: 'group'
    script: 'scripts/gradients_new.py'

#Projections
####################################################################################
rule get_projection_month12: 
    input: grad_ctx = config['analysis_dir']+'gradients/bs_emb/emb_ctx_month12.pickle',
           aligned_grad_ctx = config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_month12.pickle',

    params: month = 'month12',
            subjects = subjects,
            out_path = config['analysis_dir']+'gradients/projections/',
            labels = config['schaefer_1000_labels']
          
    output: ctx_L = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month12_ctx_L_aligned_grad.func.gii',subject=subjects),
            ctx_R = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month12_ctx_R_aligned_grad.func.gii',subject=subjects), 
            mean_ctx_L = config['analysis_dir']+'gradients/projections/cortex/mean_month12_ctx_L_grad.func.gii',
            mean_ctx_R = config['analysis_dir']+'gradients/projections/cortex/mean_month12_ctx_R_grad.func.gii'

    group: 'group'
               
    script: 'scripts/projection_ctx.py'


rule get_projection_month24: 
    input: grad_ctx = config['analysis_dir']+'gradients/bs_emb/emb_ctx_month24.pickle',
           aligned_grad_ctx = config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_month24.pickle',

    params: month = 'month24',
            subjects = subjects,
            out_path = config['analysis_dir']+'gradients/projections/',
            labels = config['schaefer_1000_labels']
          
    output: ctx_L = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month24_ctx_L_aligned_grad.func.gii',subject=subjects),
            ctx_R = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month24_ctx_R_aligned_grad.func.gii',subject=subjects), 
            mean_ctx_L = config['analysis_dir']+'gradients/projections/cortex/mean_month24_ctx_L_grad.func.gii',
            mean_ctx_R = config['analysis_dir']+'gradients/projections/cortex/mean_month24_ctx_R_grad.func.gii'

    group: 'group'
               
    script: 'scripts/projection_ctx.py'

rule get_sbctx_projection_month12: 
    input: mean_grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_L_month12.pickle',
           mean_grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_R_month12.pickle',
           aligned_grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_L_month12.pickle',
           aligned_grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_R_month12.pickle'

    params: month = 'month12',
            subjects = subjects,
            out_path = config['analysis_dir']+'gradients/projections/',
            labels = config['schaefer_1000_labels']
          
    output: sbctx_L = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_month12_sbctx_L_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants),
            sbctx_R = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_month12_sbctx_R_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants),
            sbctx_mean_L = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/mean_month12_sbctx_L_grad{componant}_image.dscalar.nii',componant=componants),
            sbctx_mean_R = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/mean_month12_sbctx_R_grad{componant}_image.dscalar.nii',componant=componants)
            

    group: 'group'

    resources: time = 30, #30 mins
               
    script: 'scripts/projection_sbctx.py'

rule get_sbctx_projection_month24: 
    input: mean_grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_L_month24.pickle',
           mean_grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_R_month24.pickle',
           aligned_grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_L_month24.pickle',
           aligned_grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_R_month24.pickle'

    params: month = 'month24',
            subjects = subjects,
            out_path = config['analysis_dir']+'gradients/projections/',
            labels = config['schaefer_1000_labels']
          
    output: sbctx_L = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_month24_sbctx_L_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants),
            sbctx_R = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_month24_sbctx_R_aligned_grad{componant}_image.dscalar.nii',subject=subjects, componant=componants),
            sbctx_mean_L = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/mean_month24_sbctx_L_grad{componant}_image.dscalar.nii',componant=componants),
            sbctx_mean_R = expand(config['analysis_dir']+'gradients/projections/sbctx/cifti/mean_month24_sbctx_R_grad{componant}_image.dscalar.nii',componant=componants)
            

    group: 'group'

    resources: time = 30, #30 mins
               
    script: 'scripts/projection_sbctx.py'

rule cii_to_nii: 
    input: nii_files = config['analysis_dir']+'gradients/projections/sbctx/cifti/sub-{subject}_{ses}_sbctx_{hemi}_aligned_grad{componant}_image.dscalar.nii'

    params: out_path = config['analysis_dir']+'gradients/projections/sbctx/nii/'
            
    output: nii_out = config['analysis_dir']+'gradients/projections/sbctx/nii/sub-{subject}_{ses}_sbctx_{hemi}_aligned_grad{componant}_image.nii'

    group: 'group_2'

    resources: time = 30, #30 mins
               
    shell: 'mkdir -p {params.out_path} && wb_command -cifti-separate {input.nii_files} COLUMN -volume-all {output.nii_out}'

"""
rule t_test:
    input: nii_files = expand(config['analysis_dir']+'gradients/projections/sbctx/nii/sub-{subject}_{ses}_sbctx_{hemi}_aligned_grad{componant}_image.nii',subject=subjects, componant=componants, hemi=hemis, ses=sessions)
    
    params: out_path = config['analysis_dir']+'gradients/projections/sbctx/ttest/'

    output: diff_grad = expand(config['analysis_dir']+'gradients/projections/sbctx/ttest/grad3/Rh/sub-{subject}_sbctx_R_grad3_diff.nii.gz',subject=subjects),
            #nii_4D = config['analysis_dir']+'gradients/projections/sbctx/nii/concatednated_4D.nii',
            
            
    group: 'stat'
               
    script: 'scripts/t_stat.py'    
"""
rule get_projection_month24: 
    input: grad_ctx = config['analysis_dir']+'gradients/bs_emb/emb_ctx_month24.pickle',
            grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_L_month24.pickle',
            grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/emb_sbctx_R_month24.pickle',
            aligned_grad_ctx = config['analysis_dir']+'gradients/bs_emb/aligned_emb_ctx_month24.pickle',
            aligned_grad_sbctx_L = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_L_month24.pickle',
            aligned_grad_sbctx_R = config['analysis_dir']+'gradients/bs_emb/aligned_emb_sbctx_R_month24.pickle'

    params: month = 'month12',
            subjects = subjects,
            out_path = config['analysis_dir']+'gradients/projections/',
            labels = config['schaefer_1000_labels']
           
    output: ctx_L = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month24_ctx_L_aligned_grad.func.gii',subject=subjects),
            ctx_R = expand(config['analysis_dir']+'gradients/projections/cortex/sub-{subject}_month24_ctx_R_aligned_grad.func.gii',subject=subjects),
            mean_ctx_L = config['analysis_dir']+'gradients/projections/cortex/month24_ctx_L_mean_grad.func.gii',
            mean_ctx_R = config['analysis_dir']+'gradients/projections/cortex/month24_ctx_R_mean_grad.func.gii'
            
            

    group: 'group'
    script: 'scripts/projections.py'


 ########################################################################################   
    
rule get_group_plots:
    input:  grad_12_csv_path = directory('../derivatives/analysis/gradients/month12'),
    	    grad_24_csv_path = directory('../derivatives/analysis/gradients/month24')
                 
    params: subj = subjects,
            out_path = '../derivatives/analysis/plots/subjects'

    output: plots_sbctx = expand('../derivatives/analysis/plots/subjects/sub-{subject}_stat_sbctx.png',subject=subjects),
            plots_ctx = expand('../derivatives/analysis/plots/subjects/sub-{subject}_stat_ctx.png',subject=subjects)

    group: 'group_analysis'
    script: 'scripts/group_plot.py'


"""
"""
rule get_projections:
    input: gradient_csv = "../derivatives/analysis/cortex/gradients/sub-{subject}/sub-{subject}_gradients.csv",
           hcp_img = config['HCP_seg_nii']
    params: comp = '{componant}'

    output: projected_image = '../derivatives/analysis/cortex/gradients/sub-{subject}/gradient_{componant}_image.nii.gz',
    		projected_plot = '../derivatives/analysis/cortex/gradients/sub-{subject}/gradient_{componant}_image_plot.png'
    #conda: 'cfg/bspace.yml'
    group: 'pre_align'
    script: 'scripts/project_gradients.py'

rule get_surface_gradients:
	input: gradient_csv = "/home/dimuthu1/scratch/project2/derivatives/analysis/cortex/gradients/sub-{subject}/sub-{subject}_gradients.csv",
	       surf_lh = config['surf_label_lh'],
	       surf_rh = config['surf_label_rh']

	params: stat_out_path = "/home/dimuthu1/scratch/project2/derivatives/analysis/cortex/gradients/sub-{subject}/surfaces"

	output: projected_image = '/home/dimuthu1/scratch/project2/derivatives/analysis/cortex/gradients/sub-{subject}/surfaces/plotL_grad_{componant}.func.gii'

	#conda: 'cfg/bspace.yml'
	group: 'pre_align'
	shell: 'mkdir -p {params.stat_out_path} && bash scripts/get_surf {input.surf_lh} {input.surf_rh} {input.gradient_csv} {params.stat_out_path}'

rule test_group:
    input: grads_path = '../derivatives/analysis/cortex/gradients'
              
    params: subj = subjects,
            aligned_grads_path = '../derivatives/analysis/cortex/aligned_gradients'

    output: aligned_grads = expand('../derivatives/analysis/cortex/aligned_gradients/sub-{subject}_gradients.csv',subject=subjects)


    group: 'post_align'
    script: 'scripts/procrust.py'


rule get_stat:
    input: stat_lh = config['stat_path_lh'],
           stat_rh = config['stat_path_rh']
           

    params: stat_out_path = '../derivatives/analysis/cortex/hcp_stat/{subject}'

    output: stat_csv_lh = '../derivatives/analysis/cortex/hcp_stat/sub-{subject}/{subject}_hcp_stat_lh.csv',
    		stat_csv_rh = '../derivatives/analysis/cortex/hcp_stat/sub-{subject}/{subject}_hcp_stat_rh.csv'

    group: 'stat'
    script: 'scripts/hcp_stat.py'

rule get_myelin_stat:
    input: myelin_img = "/home/dimuthu1/scratch/project2/derivatives/myelin_volume/{subject}/T1wDividedByT2w.nii.gz",
           hcp_img = config['HCP_seg_nii'],
           hcp_360_labels = config['HCP_360_labels']
             

    params: stat_out_path = '/home/dimuthu1/scratch/project2/derivatives/analysis/cortex/hcp_stat/sub-{subject}'


    output: stat_csv_lh = '/home/dimuthu1/scratch/project2/derivatives/analysis/cortex/hcp_stat/sub-{subject}/sub-{subject}_mean-myelin_L.txt',
            stat_csv_rh = '/home/dimuthu1/scratch/project2/derivatives/analysis/cortex/hcp_stat/sub-{subject}/sub-{subject}_mean-myelin_R.txt'

    group: 'stat'

    resources: 
           time = 120 #30 mins

    shell: 'bash scripts/myelin_stat.sh {wildcards.subject} {input.myelin_img} {input.hcp_img} {input.hcp_360_labels} {params.stat_out_path}'






rule get_stat_plots:
    input: hcp_lh = '../derivatives/analysis/cortex/hcp_stat/sub-{subject}/{subject}_hcp_stat_lh.csv',
           hcp_rh = '../derivatives/analysis/cortex/hcp_stat/sub-{subject}/{subject}_hcp_stat_rh.csv',
           gradient = '../derivatives/analysis/cortex/aligned_gradients/sub-{subject}_gradients.csv'  

    output: lh_stat_plots = '../derivatives/analysis/cortex/hcp_stat/sub-{subject}/{subject}_hcp_all_stat_lh.png',
    		rh_stat_plots = '../derivatives/analysis/cortex/hcp_stat/sub-{subject}/{subject}_hcp_sall_stat_rh.png'

    group: 'stat'
    script: 'scripts/combine_csv.py'


rule get_group_plots:
    input: hcp_path = directory('../derivatives/analysis/cortex/hcp_stat'),
           grad_csv_path = directory('../derivatives/analysis/cortex/aligned_gradients')
              
    params: subj = subjects,
            out_path = '../derivatives/analysis/cortex/group_analysis'

    output: aligned_grads = ['../derivatives/analysis/cortex/group_analysis/group_stat_L.png','../derivatives/analysis/cortex/group_analysis/group_stat_R.png']

    group: 'group_analysis'
    script: 'scripts/group_plot.py'

"""





    

    
  



    
