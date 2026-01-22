#!/usr/local/bin/python3

import sys
sys.path.append('/oncoGAN/models')
sys.path.append('/oncoGAN/models/caloforest')

import os
import re
import copy
import click
import pickle
import itertools
import subprocess
import warnings
from multiprocessing import Pool
import torch
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, List, Tuple, Optional
from datetime import date
from liftover import ChainFile
from tqdm import tqdm
from pyfaidx import Fasta
from Bio.Seq import Seq
import pyranges as pr

VERSION = "1.0.0"

default_tumors:list = ["Biliary-AdenoCA","Bladder-TCC","Bone-Leiomyo","Bone-Osteosarc","Breast-AdenoCa","Cervix-SCC","CNS-GBM","CNS-Medullo","CNS-Oligo","CNS-PiloAstro","ColoRect-AdenoCA","Eso-AdenoCa","Head-SCC","Kidney-ChRCC","Kidney-RCC","Liver-HCC","Lung-AdenoCA","Lung-SCC","Lymph-BNHL","Lymph-CLL","Myeloid-MPN","Ovary-AdenoCA","Panc-AdenoCA","Panc-Endocrine","Prost-AdenoCA","Skin-Melanoma","Stomach-AdenoCA","Thy-AdenoCA","Uterus-AdenoCA"]

warnings.simplefilter(action='ignore', category=FutureWarning)

#################
# Miscellaneous #
#################

def out_path(outDir, tumor, prefix=None, n=0, custom=False) -> click.Path:

    """
    Get the absolute path and name for the outputs
    """

    if custom:
        output:click.Path = f"{outDir}/{prefix}.vcf"
    elif prefix is not None:
        output:click.Path = f"{outDir}/{prefix}_sim{n}.vcf"
    else:
        output:click.Path = f"{outDir}/{tumor}_sim{n}.vcf"
    
    return(output)

def chrom2int(chrom) -> int:

    """
    Convert the chromosome to an integer
    """

    if chrom.isdigit():
        return int(chrom)
    elif chrom == 'X':
        return 23
    elif chrom == 'Y':
        return 24
    else:
        return chrom

def chrom2str(chrom) -> str:

    """
    Convert the chromosome to a string
    """

    if chrom == 23:
        return 'X'
    elif chrom == 24:
        return 'Y'
    else:
        return str(chrom)

def sort_by_int_chrom(chrom) -> int:
    
    """
    Sort a dataframe using integer chromosomes
    """

    if chrom == 'X':
        return 23
    if chrom == 'Y':
        return 24
    else:
        return int(chrom)

def hg19tohg38(vcf=None, cna=None, sv=None) -> pd.DataFrame:

    """
    Convert hg19 coordinates to hg38
    """

    if vcf is not None:
        converter = ChainFile('/.liftover/hg19ToHg38.over.chain.gz')
        for i,row in vcf.iterrows():
            chrom:str = str(row['#CHROM'])
            pos:int = int(row['POS'])
            try:
                liftOver_result:tuple = converter[chrom][pos][0]
                vcf.loc[i, '#CHROM'] = liftOver_result[0]
                vcf.loc[i, 'POS'] = liftOver_result[1]
            except IndexError:
                vcf.loc[i, '#CHROM'] = 'Remove'
        vcf = vcf[~vcf['#CHROM'].str.contains('Remove', na=False)]
        return(vcf)
    elif cna is not None:
        hg19_end:list = [249250621,492449994,690472424,881626700,1062541960,1233657027,1392795690,1539159712,1680373143,1815907890,1950914406,2084766301,2199936179,2307285719,2409817111,2500171864,2581367074,2659444322,2718573305,2781598825,2829728720,2881033286,3036303846,3095677412]
        hg38_end:list = [248956422,491149951,689445510,879660065,1061198324,1232004303,1391350276,1536488912,1674883629,1808681051,1943767673,2077042982,2191407310,2298451028,2400442217,2490780562,2574038003,2654411288,2713028904,2777473071,2824183054,2875001522,3031042417,3088269832]
        hg19_hg38_ends:dict = dict(zip(hg19_end, hg38_end))

        cna['end'] = cna['end'].apply(lambda x: hg19_hg38_ends.get(x, x))
        return(cna)
    elif sv is not None:
        chroms:list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']
        hg19_end:list = [249250621,492449994,690472424,881626700,1062541960,1233657027,1392795690,1539159712,1680373143,1815907890,1950914406,2084766301,2199936179,2307285719,2409817111,2500171864,2581367074,2659444322,2718573305,2781598825,2829728720,2881033286,3036303846,3095677412]
        hg38_end:list = [248956422,491149951,689445510,879660065,1061198324,1232004303,1391350276,1536488912,1674883629,1808681051,1943767673,2077042982,2191407310,2298451028,2400442217,2490780562,2574038003,2654411288,2713028904,2777473071,2824183054,2875001522,3031042417,3088269832]
        hg19_dict:dict = dict(zip(chroms, hg19_end))
        hg38_dict:dict = dict(zip(chroms, hg38_end))

        for i, row in sv.iterrows():
            ## Chrom1
            hg19_end1:int = hg19_dict.get(row['chrom1'])
            hg38_end1:int = hg38_dict.get(row['chrom1'])
            if row['end1'] > hg38_end1:
                sv.loc[i, 'end1'] = hg38_end1 - (hg19_end1 - row['end1'])
                sv.loc[i, 'start1'] = sv.loc[i, 'end1']-1

            ## Chrom2
            hg19_end2:int = hg19_dict.get(row['chrom2'])
            hg38_end2:int = hg38_dict.get(row['chrom2'])
            if row['end2'] > hg38_end2:
                sv.loc[i, 'end2'] = hg38_end2 - (hg19_end2 - row['end2'])
                sv.loc[i, 'start2'] = sv.loc[i, 'end2']-1
        return(sv)

##########
# Models #
##########

def cna_sv_models(device) -> list:

    """
    Get the CNA and SV models
    """
    
    cna_sv_countModel = torch.load("/oncoGAN/trained_models/cna_sv/CNA_SV_counts.pkl", map_location=device)
    cnaModel = torch.load("/oncoGAN/trained_models/cna_sv/CNA_model.pkl", map_location=device)
    svModel:dict = {}
    with open("/oncoGAN/trained_models/cna_sv/SV_positions.pkl", 'rb') as f:
        svModel['pos'] = pickle.load(f)
    svModel['sv'] = torch.load("/oncoGAN/trained_models/cna_sv/SV_model.pkl", map_location=device)

    return(cna_sv_countModel, cnaModel, svModel)

def calo_forest_generation(load_dir, y_labels):
    
    """
    Generate samples using Calo-Forest models
    """
    
    # Load the forest model
    model = os.path.join(load_dir, 'forest_model.pkl')
    with open(model, 'rb') as file:
        model_dict = pickle.load(file)
    model_dict['model'].set_logdir(load_dir)
    model_dict['model'].set_solver_fn(model_dict['cfg']["solver"])
    reverse_mapping = {v: k for k, v in model_dict['mapping'].items()}

    # Prepare the number and type of samples to generate
    y_labels_map = [reverse_mapping[x] for x in y_labels]
    n = len(y_labels_map)
    y = np.array(y_labels_map)

    # Generate the samples
    Xy_fake = model_dict['model'].generate(batch_size=n, label_y=y)

    # Map back the labels to their original values
    Xy_fake:pd.DataFrame = pd.DataFrame.from_records(Xy_fake, columns=model_dict['columns'])
    pred_col = Xy_fake.columns[-1]
    Xy_fake[pred_col] = Xy_fake[pred_col].apply(lambda x: model_dict['mapping'][x])

    del model #release model memory
    return Xy_fake

def dae_reconstruction(z, dae_model: Literal['driver_profile', 'genomic_profile']) -> pd.DataFrame:
    
    """
    Function to reconstruct diffusion latent spaces using the DAE model
    """

    z_serialized = pickle.dumps(z)
    command = ['micromamba', 'run', '-n', 'dae',
               'python3', '/oncoGAN/dae_reconstruction.py',
               '--model', f'{dae_model}']
    reconstructed_serialized = subprocess.run(command,
                                              input=z_serialized,
                                              capture_output=True)
    reconstructed = pickle.loads(reconstructed_serialized.stdout)
    reconstructed = pd.DataFrame(reconstructed['df'], columns=reconstructed['columns'])

    return reconstructed

###############
# Simulations #
###############

def simulate_counts(tumor_f:str, nCases_f:int) -> pd.DataFrame:

    """
    Function to generate the number of each type of mutation per case
    """

    def clean_counts_apply(row):
        tumor = row['Tumor']
        for col in row.index[:-1]:
            val = row[col]
            stats = tumor_stats.get(tumor, {}).get(col)

            if stats is None:
                row[col] = round(val)
            elif stats.get('All_Zero', False):
                row[col] = 0
            elif val < stats['Min'] * 0.8:
                row[col] = 0
            elif val > stats['Max'] * 1.3:
                row[col] = np.nan
            else:
                row[col] = round(val)
        return row
    
    # Prepare the list of donors to simulate
    nCases_x5 = nCases_f * 5
    if tumor_f == "Lymph-CLL":
        mCases:int = round(nCases_x5*0.42)
        uCases:int = nCases_x5 - mCases
        cases_list = ['Lymph-MCLL']*mCases + ['Lymph-UCLL']*uCases
    else:
        cases_list = [tumor_f]*nCases_x5
    
    # Generate samples
    counts = calo_forest_generation('/oncoGAN/trained_models/donor_characteristics', cases_list)

    # Clean the output a bit (round, min and max boundaries)
    tumor_stats = pd.read_pickle('/oncoGAN/trained_models/donor_characteristics/donor_characteristics_stats.pkl')
    counts = counts.apply(clean_counts_apply, axis=1).dropna().reset_index(drop=True)
    counts = counts.sample(n=nCases_f, replace=False).reset_index(drop=True)

    #TODO - Simulate also medium and big size indels
    counts = counts.drop(columns=['medium_ins', 'big_ins', 'medium_del', 'big_del'])

    return counts

def simulate_sex(tumor_list_f:tuple) -> tuple:

    """
    Simulate the sex of each donor
    """

    tumor_list_f_updated:tuple = ("Lymph-CLL" if tumor in {"Lymph-MCLL", "Lymph-UCLL"} else tumor for tumor in tumor_list_f)
    tumor_sex_df:pd.DataFrame = pd.read_csv('/oncoGAN/trained_models/xy_usage_ranks.txt', sep='\t')
    tumor_sex_list:list = tumor_sex_df['label'].to_list()

    sex_list:list = []
    for tumor in tumor_list_f_updated:
        tumor_sex_options:list = [t[-1] for t in tumor_sex_list if t.startswith(tumor)]
        
        if 'F' in tumor_sex_options and 'M' in tumor_sex_options:
            sex:str = "M" if random.random() < 0.5 else "F"
        elif 'F' in tumor_sex_options and 'M' not in tumor_sex_options:
            sex:str = "F"
        elif 'F' not in tumor_sex_options and 'M' in tumor_sex_options:
            sex:str = "M"
        
        sex_list.append(sex)
            
    return(tuple(sex_list))

def simulate_signatures(counts_f:pd.DataFrame) -> dict:

    """
    Function to simulate the mutational signatures for each donor
    """
    
    def assign_indel_ref_alt_apply(context):
        #TODO - Handle indel size >5+, using maybe the medium|big-ins|del columns
        ref_list = []
        alt_list = []
        size, indel_type, base, context_length = context.split(':')
        if int(size) == 1:
            if base == 'C':
                repeat_base = random.choice(['C', 'G'])
            elif base == 'T':
                repeat_base = random.choice(['A', 'T'])

            if indel_type == 'Del':
                ref_list.append(repeat_base*(int(context_length)+1))
                alt_list.append(repeat_base*int(context_length))
            elif indel_type == 'Ins':
                ref_list.append(repeat_base*int(context_length))
                alt_list.append(repeat_base*(int(context_length)+1))
        elif base == "R":
            for _ in range(5):
                repeat_base = ''
                for _ in range(int(size)):
                    repeat_base +=  random.choice(['A', 'C', 'G', 'T'])

                if indel_type == 'Del':
                    ref_list.append(repeat_base*(int(context_length)+1))
                    alt_list.append(repeat_base*int(context_length))
                elif indel_type == 'Ins':
                    ref_list.append(repeat_base*int(context_length))
                    alt_list.append(repeat_base*(int(context_length)+1))
        elif base == 'M': 
            for _ in range(5):
                repeat_base = ''
                range_len = int(size) if int(size) < 5 else random.randint(5, 9)
                context_length_updated = int(context_length) if int(context_length) < 5 else random.randint(4, range_len-1)
                for _ in range(range_len):
                    if len(repeat_base) == 0:
                        repeat_base +=  random.choice(['A', 'C', 'G', 'T'])
                    else:
                        repeat_base +=  random.choice([nt for nt in ['A', 'C', 'G', 'T'] if nt != repeat_base[-1]])

                ref1 = repeat_base+repeat_base[:context_length_updated]
                ref2 = repeat_base[-context_length_updated:]+repeat_base
                ref_list.append(f"{ref1}|{ref2}")
                alt_list.append(f"NA|{repeat_base[-1]}")
        
        ref = ','.join(ref_list)
        alt = ','.join(alt_list)
        
        return (ref, alt)
    
    def reverse_complement_sbs_apply(row):

        if random.choice([True, False]):
            context = str(Seq(row['contexts']).reverse_complement())
            ref = str(Seq(row['ref']).reverse_complement())
            alt = str(Seq(row['alt']).reverse_complement())
            return(context, ref, alt)
        else:
            return(row['contexts'], row['ref'], row['alt'])

    def process_mutations(mut_df, type:Literal['sbs', 'id']):

        def distribute_diff_apply(group:pd.DataFrame) -> pd.DataFrame:
            total = int(group['total'].iloc[0])
            current = group['n'].sum()
            missing = total - current

            if missing > 0:
                idx = group['diff'].nlargest(missing).index
                group.loc[idx, 'n'] += 1

            return group
        
        # Normalize the context usage
        contexts = mut_df.drop(columns=['signature', 'total'])
        normalized_contexts = contexts.div(contexts.sum(axis=1), axis=0)
        normalized_contexts = pd.concat([mut_df['signature'], normalized_contexts], axis=1)
        # Pivot longer
        mut_df_long = pd.melt(
            normalized_contexts,
            id_vars=['signature'],
            var_name='contexts',
            value_name='perc'
        )
        # Calculate the number of mutations for each context
        mut_df_long['perc'] = pd.to_numeric(mut_df_long['perc'], errors='coerce')
        mut_df_long['total'] = pd.to_numeric(mut_df_long['signature'].map(row_features), errors='coerce')
        mut_df_long['exp'] = mut_df_long['perc'] * mut_df_long['total']
        mut_df_long['n'] = np.floor(mut_df_long['exp']).astype(int)
        mut_df_long['diff'] = mut_df_long['exp'] - mut_df_long['n']
        mut_df_long = mut_df_long.groupby('signature', group_keys=False).apply(distribute_diff_apply)
        mut_df_long = mut_df_long.drop(columns=['exp', 'diff'])
        mut_df_long_expanded = mut_df_long.loc[mut_df_long.index.repeat(mut_df_long['n'].astype(int))].reset_index(drop=True)

        if type == 'sbs':
            # Extract context, ref and alt bases
            mut_df_long_expanded[['pre', 'ref', 'alt', 'post']] = mut_df_long_expanded['contexts'].str.extract(r'([A-Z])\[([A-Z])>([A-Z])\]([A-Z])')
            mut_df_long_expanded['contexts'] = mut_df_long_expanded['pre'] + mut_df_long_expanded['ref'] + mut_df_long_expanded['post']
            mut_df_long_expanded[['contexts', 'ref', 'alt']] = mut_df_long_expanded.apply(reverse_complement_sbs_apply, axis=1, result_type='expand')
            mut_df_processed = mut_df_long_expanded[['signature', 'contexts', 'ref', 'alt']]
        elif type == 'id':
            mut_df_long_expanded[['ref', 'alt']] = mut_df_long_expanded['contexts'].apply(assign_indel_ref_alt_apply).apply(pd.Series)
            mut_df_processed = mut_df_long_expanded[['signature', 'contexts', 'ref', 'alt']]
            
        return mut_df_processed[['signature', 'contexts', 'ref', 'alt']]
    
    # Count how many donors present each signature
    signatures2sim = (counts_f[counts_f.columns[:-1]] != 0).sum()

    # Prepare the list of signatures to simulate
    sbs_list = [feature for feature, count in signatures2sim.items() for _ in range(count*100) if feature.startswith("SBS")]
    indel_list = [feature for feature, count in signatures2sim.items() for _ in range(count*100) if feature.startswith("ID")]

    # Generate the signatures
    sbs = calo_forest_generation('/oncoGAN/trained_models/sbs_context', sbs_list)
    indels = calo_forest_generation('/oncoGAN/trained_models/indel_context', indel_list)
    
    # Assign signatures for each donor
    signatures_dict = {}
    for donor_id, row in counts_f.iterrows():
        row_features = row[row.index.str.startswith(('SBS', 'ID', 'DNP', 'TNP'))] #TODO - Simulate also medium and big size indels
        row_features = row_features[row_features != 0]
        total = row_features.sum()
        signatures_donor_sbs = pd.DataFrame()
        signatures_donor_indels = pd.DataFrame()
        donor_dnp = pd.DataFrame()
        donor_tnp = pd.DataFrame()
        for signature, count in row_features.items():
            # Find the signature where the difference between 'total' and this donor's total number of mutations is minimized
            if signature.startswith("SBS"):
                sbs_signature = sbs[sbs['signature'] == signature]
                idx = (sbs_signature['total'] - total).abs().idxmin()
                selected_sbs = sbs_signature.loc[idx].to_frame().T
                selected_sbs['total'] = count
                signatures_donor_sbs = pd.concat([signatures_donor_sbs, selected_sbs], ignore_index=True)
                sbs = sbs.drop(idx).reset_index(drop=True)
            elif signature.startswith("ID"):
                indels_signature = indels[indels['signature'] == signature]
                idx = (indels_signature['total'] - total).abs().idxmin()
                selected_indels = indels_signature.loc[idx].to_frame().T
                selected_indels['total'] = count
                signatures_donor_indels = pd.concat([signatures_donor_indels, selected_indels], ignore_index=True)
                indels = indels.drop(idx).reset_index(drop=True)
            elif signature.startswith('DNP'):
                nucleotides = ['A', 'C', 'G', 'T']
                while donor_dnp.shape[0] != count:
                    dnp_ref = random.choice([f"{n1}{n2}" for n1, n2 in itertools.product(nucleotides, repeat=2)])
                    n1_alt_list = [nt for nt in nucleotides if nt != dnp_ref[0]]
                    n2_alt_list = [nt for nt in nucleotides if nt != dnp_ref[1]]
                    dnp_alt = random.choice([f"{n1}{n2}" for n1, n2 in itertools.product(n1_alt_list, n2_alt_list)])
                    selected_dnp = pd.DataFrame({'signature':['DNP'], 'contexts':['DNP'], 'ref':[dnp_ref], 'alt':[dnp_alt]})
                    donor_dnp = pd.concat([donor_dnp, selected_dnp], ignore_index=True)
            elif signature.startswith('TNP'):
                nucleotides = ['A', 'C', 'G', 'T']
                while donor_tnp.shape[0] != count:
                    tnp_ref = random.choice([f"{n1}{n2}{n3}" for n1, n2, n3 in itertools.product(nucleotides, repeat=3)])
                    n1_alt_list = [nt for nt in nucleotides if nt != tnp_ref[0]]
                    n2_alt_list = [nt for nt in nucleotides if nt != tnp_ref[1]]
                    n3_alt_list = [nt for nt in nucleotides if nt != tnp_ref[2]]
                    tnp_alt = random.choice([f"{n1}{n2}{n3}" for n1, n2, n3 in itertools.product(n1_alt_list, n2_alt_list, n3_alt_list)])
                    selected_tnp = pd.DataFrame({'signature':['TNP'], 'contexts':['TNP'], 'ref':[tnp_ref], 'alt':[tnp_alt]})
                    donor_tnp = pd.concat([donor_tnp, selected_tnp], ignore_index=True)
        signatures_donor_sbs = process_mutations(signatures_donor_sbs, type='sbs')
        signatures_donor_indels = process_mutations(signatures_donor_indels, type='id')
        signatures_dict[donor_id] = pd.concat([signatures_donor_sbs, signatures_donor_indels, donor_dnp, donor_tnp], ignore_index=True)

    return signatures_dict

def simulate_genomic_profile(tumor_list_f:tuple, counts_total_f:pd.Series) -> pd.DataFrame:

    """
    Function to simulate the genomic pattern profiles for each donor
    """
    
    # Generate latent profile
    latent_profiles = calo_forest_generation('/oncoGAN/trained_models/positional_pattern', tumor_list_f)
    latent_profiles = latent_profiles.drop(columns=['Tumor'])

    # Reconstruct the profile
    raw_genomic_profiles = dae_reconstruction(latent_profiles, 'genomic_profile')

    # Clean the output (normalize the row to sum 100%)
    prop_genomic_profiles = raw_genomic_profiles.div(raw_genomic_profiles.sum(axis=1), axis=0)

    # Get numbers instead of percentages
    exp_genomic_profiles = prop_genomic_profiles.multiply(counts_total_f, axis=0)
    floor_genomic_profiles = np.floor(exp_genomic_profiles)

    # Check that the each profile sums exactly the total number of mutations
    genomic_profiles = floor_genomic_profiles.copy().astype(int)
    dif_genomic_profiles = exp_genomic_profiles - floor_genomic_profiles
    for idx, total in enumerate(counts_total_f):
        current = genomic_profiles.iloc[idx].sum()
        missing = int(total - current)
        if missing > 0:
            cols = dif_genomic_profiles.iloc[idx].nlargest(missing).index
            genomic_profiles.loc[genomic_profiles.index[idx], cols] += 1

    return genomic_profiles

def simulate_driver_profile(tumor_list_f:tuple) -> pd.DataFrame:

    """
    Function to simulate the driver profiles for each donor
    """

    # Generate latent profile
    latent_profiles = calo_forest_generation('/oncoGAN/trained_models/driver_profile', tumor_list_f)
    latent_profiles = latent_profiles.drop(columns=['Tumor'])

    # Reconstruct the profile
    driver_profiles = dae_reconstruction(latent_profiles, 'driver_profile')

    # Get numbers instead of percentages
    driver_profiles = driver_profiles.round(0).astype(int)

    return driver_profiles

def simulate_vaf_rank(tumor_list_f:tuple) -> tuple:

    """
    Function to simulate the VAF range for each donor
    """

    rank_file:pd.DataFrame = pd.read_csv("/oncoGAN/trained_models/donor_vaf_rank.tsv", sep='\t') 

    donor_vafs:list = []
    for tumor in tumor_list_f:
        if tumor in ["Lymph-MCLL", "Lymph-UCLL"]:
            tumor = "Lymph-CLL"
        rank_file_i = rank_file.loc[rank_file["tumor"] == tumor]
        vaf = random.choices(rank_file_i.columns[1:], weights=rank_file_i.values[0][1:],k=1)[0]
        donor_vafs.append(vaf)

    return tuple(donor_vafs)

def simulate_mut_vafs(tumor_list_f:tuple, vaf_ranks_list:tuple, counts_total_f:pd.Series) -> dict: 

    """
    A function to simulate the VAF of each mutation
    """
    
    def vaf_rank2float(case_mut_vafs_f:list) -> list:

        """
        Convert the VAF rank to a float value
        """

        final_vaf_list:list = []
        for vaf in case_mut_vafs_f:
            vaf:str = re.sub(r'[\[\)\]]', '', vaf)
            start:float = float(vaf.split(',')[0])
            end:float = float(vaf.split(',')[1])
            final_vaf_list.append(round(random.uniform(start, end), 2))

        return(final_vaf_list)

    prop_vaf_file:pd.DataFrame = pd.read_csv(f"/oncoGAN/trained_models/mutation_vaf_rank.tsv", sep='\t')
    mut_vafs:dict = {}
    for idx, (tumor, vaf_rank, n) in enumerate(zip(tumor_list_f, vaf_ranks_list, counts_total_f)):
        case_prop_vaf_file = prop_vaf_file.loc[prop_vaf_file["tumor"]==tumor, ['vaf_range', vaf_rank]]
        case_mut_vafs:list = random.choices(list(case_prop_vaf_file['vaf_range']), weights=list(case_prop_vaf_file[vaf_rank]), k=n)
        case_mut_vafs = vaf_rank2float(case_mut_vafs)
        mut_vafs[idx] = tuple(case_mut_vafs)
    
    return(mut_vafs)

########################
# Complete simulations #
########################

def process_chunk(chunk_data, refGenome, n_attempt):
    
    def get_sequence(chromosomes_f:np.ndarray, positions_f:np.ndarray, fasta, window:int=5000) -> list:
        
        """
        Function to get a DNA sequence around each position
        """

        # Update chrom variable
        chrom_prefix:bool = 'chr1' in fasta.keys()
        if chrom_prefix:
            chromosomes_f = np.char.add("chr", chromosomes_f)
        
        # Define the window
        positions_end:np.ndarray = positions_f + window

        sequences:list = [fasta[c][s:e].seq for c, s, e in zip(chromosomes_f, positions_f, positions_end)]
        return sequences

    def match_pos2ctx(signatures_f:pd.DataFrame, chromosomes_f:np.ndarray, positions_f:np.ndarray, sequences_f:list) -> pd.DataFrame:

        """
        Function to assign a position to each mutation
        """

        def create_indels_pattern(ref_f:str, alt_f:str, context_f:str) -> str:

            """
            Function to manually create indel patterns. The patterns to be used are located in the ref and alt columns. However, to avoid creating more microhomology than needs to be simulated, we generate all possible contexts that break the extra microhomology
            """

            nucleotides = ['A', 'C', 'G', 'T']

            # Exclude the first base from N
            if ':M:' in context_f:
                size, _, _, context_length = context_f.split(':')
                ref1,ref2 = ref_f.split('|')
                nt_options1 = [nt for nt in nucleotides if nt != ref1[int(context_length)]]
                nt_options2 = [nt for nt in nucleotides if nt != ref2[int(size)-1]]
                pattern1 = [f"{ref1}{nt}" for nt in nt_options1]
                pattern2 = [f"{nt}{ref2}" for nt in nt_options2]
                pattern = pattern1+pattern2
            else:
                if ref_f != "":
                    nt_options = [nt for nt in nucleotides if nt != ref_f[0]]
                else:
                    nt_options = [nt for nt in nucleotides if nt != alt_f[0]]
                # Create all combinations: N1 + ref + N2
                pattern = [f"{n1}{ref_f}{n2}" for n1, n2 in itertools.product(nt_options, repeat=2)]

            return "|".join(pattern)

        def find_context_in_sequence(signature_i:str, context_i:str, ref_i:str, alt_i:str, sequence_i:str) -> Tuple[List[int], Optional[List[str]]]:

            if signature_i.startswith("SBS"):
                indexes = [m.start() for m in re.finditer(context_i, sequence_i)]
                return (indexes, None)

            elif signature_i.startswith("ID"):
                refs = ref_i.split(",")
                alts = alt_i.split(",")

                for r, a in zip(refs, alts):
                    pattern = create_indels_pattern(r, a, context_i)
                    matches = [(m.start(), m.group()) for m in re.finditer(pattern, sequence_i)]
                    if matches:
                        return tuple(zip(*matches))
                return ([], [])

            elif signature_i in ("DNP", "TNP"):
                indexes = [m.start() for m in re.finditer(ref_i, sequence_i)]
                return (indexes, None)

            return ([], None)

        def update_mutations(mut_i, chrom_i:str, position_i, ctx_indexes_i:list, indel_patterns_i:list|None, fasta) -> Tuple[int, int, int]:

            if not ctx_indexes_i:
                return (None, None, None)

            index_choice:int = random.randrange(len(ctx_indexes_i))

            # Microhomology INDELs
            if ':M:' in mut_i.contexts:
                m_size, _, _, m_context_length = mut_i.contexts.split(':')
                m_ref = indel_patterns_i[index_choice]

                m_option = int(not (mut_i.ref.split('|')[0] in m_ref))

                if m_option == 0:
                    m_pos = position_i + ctx_indexes_i[index_choice]
                    m_prev_base = fasta[chrom_i][m_pos-1:m_pos].seq
                    ref = m_prev_base + m_ref[:int(m_size)]
                    alt = m_prev_base
                else:
                    m_pos = position_i + ctx_indexes_i[index_choice] + int(m_context_length) + 1
                    ref = m_ref[-(int(m_size)+1):]
                    alt = ref[0]

            # INDELs
            elif mut_i.signature.startswith('ID'):
                m_pos = position_i + ctx_indexes_i[index_choice] + 1
                ref = indel_patterns_i[index_choice][:-1]

                if mut_i.alt == "":
                    alt = ref[0]
                else:
                    alt_list = mut_i.alt.split(',')
                    ref_list = mut_i.ref.split(',')
                    alt = ref[0] + alt_list[ref_list.index(ref[1:])]

            # SBSs
            elif mut_i.signature.startswith('SBS'):
                m_pos = position_i + ctx_indexes_i[index_choice] + 2
                ref = mut_i.ref
                alt = mut_i.alt

            # DNPs and TNPs
            elif mut_i.signature in ('DNP', 'TNP'):
                m_pos = position_i + ctx_indexes_i[index_choice] + 1
                ref = mut_i.ref
                alt = mut_i.alt

            else:
                return (None, None, None)

            return (int(m_pos), ref, alt)
        
        ctx_indexes, indel_patterns = zip(*[find_context_in_sequence(row.signature, row.contexts, row.ref, row.alt, seq) for row, seq in zip(signatures_f.itertuples(index=False), sequences_f)])
        updated_position, updated_ref, updated_alt = zip(*[update_mutations(row, chrom, pos, ctx_idxs, id_pat, fasta) for row, chrom, pos, ctx_idxs, id_pat in zip(signatures_f.itertuples(index=False), chromosomes_f, positions_f, ctx_indexes, indel_patterns)])
        
        updated_signatures = signatures_f.copy()
        updated_signatures['chrom'] = chromosomes_f
        updated_signatures['pos'] = updated_position
        updated_signatures['updated_ref'] = updated_ref
        updated_signatures['updated_alt'] = updated_alt
        
        return updated_signatures
    
    fasta = Fasta(refGenome)
    
    # Unpack the chunked data
    donor_df_chunk_f, chrom_chunk_f, pos_chunk_f = chunk_data

    # Extract a genomic sequence around each position
    if n_attempt >= 5:
        asgn_sequences_chunk:list = get_sequence(chrom_chunk_f, pos_chunk_f, fasta, window=50000) #TODO - Create a way to check chromosome boundaries to not break the code
    else:    
        asgn_sequences_chunk:list = get_sequence(chrom_chunk_f, pos_chunk_f, fasta)

    # Match the context with the position range
    tmp_donor_df_chunk: pd.DataFrame = match_pos2ctx(donor_df_chunk_f, chrom_chunk_f, pos_chunk_f, asgn_sequences_chunk)
    
    return tmp_donor_df_chunk
    
def select_driver_mutations(tumor_list_f:tuple, driver_profile_f:pd.DataFrame) -> dict:
    
    driver_database:pd.DataFrame = pd.read_csv("/oncoGAN/trained_models/driver_profile/driver_mutations_database.csv", delimiter=',')

    driver_mutations:dict = {}
    for idx, tumor in enumerate(tumor_list_f):
        case_driver_profile:pd.Series = driver_profile_f.iloc[idx]
        case_driver_mutations:list = []
        for gene, n in case_driver_profile.items():
            if n <= 0:
                continue
            else:
                driver_muts = (driver_database.query("gene_id == @gene and tumor == @tumor").sample(n=n, replace=False))
                case_driver_mutations.append(driver_muts)
        driver_mutations[idx] = pd.concat(case_driver_mutations, ignore_index=True)
    
    return driver_mutations

def assign_genomic_positions(sex_f:str, signatures_f:pd.DataFrame, genomic_pattern_f:pd.Series, refGenome, cpus) -> pd.DataFrame:
    
    def parse_range_map(genomic_interval:str) -> pd.MultiIndex:
        left, right = genomic_interval.strip("[]()").split(",")
        return int(float(left)), int(float(right))
    
    def assign_chromosome(positions_f:np.ndarray) -> tuple:
        chromosome_decode:np.array  = np.array(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y'])
        position_decode:np.array  = np.array([0,249250621,492449994,690472424,881626700,1062541960,1233657027,1392795690,1539159712,1680373143,1815907890,1950914406,2084766301,2199936179,2307285719,2409817111,2500171864,2581367074,2659444322,2718573305,2781598825,2829728720,2881033286,3036303846])
        
        position_encode:np.array  = np.digitize(positions_f, position_decode, right=True)
        
        map_positions:np.ndarray = positions_f - np.take(position_decode, position_encode-1)
        map_chromosomes:np.ndarray = np.take(chromosome_decode, position_encode-1)
        
        return (map_chromosomes.astype(str), map_positions.astype(int))
    
    # Randomize the mutational signatures input
    signatures_f = signatures_f.sample(frac=1).reset_index(drop=True)

    #TODO - Check sexual chrom usage
    # Initialize the output dataframe
    donor_df = signatures_f.copy()
    donor_df["chrom"] = None
    donor_df["pos"] = None
    donor_df["updated_ref"] = None
    donor_df["updated_alt"] = None

    # Define a genomic position for each mutation
    rng = np.random.default_rng()
    position_ranges:np.array = np.array(genomic_pattern_f.index.map(parse_range_map).tolist())
    position_ranges_expanded = np.repeat(position_ranges, genomic_pattern_f.values, axis=0)

    while_round = 0
    n_missing = donor_df.shape[0]
    while n_missing > 0 and while_round < 15:
        mask = donor_df["updated_ref"].isna()
        missing_indices = donor_df[mask].index

        # Sample positions only for missing rows
        positions:np.ndarray = np.concatenate([rng.integers(low=lo, high=hi, size=1) for lo, hi in position_ranges_expanded[mask]])
        asgn_chromosomes, asgn_positions = assign_chromosome(positions)
        
        # Prepare multiprocessing
        masked_donor_df = donor_df.loc[mask]
        if masked_donor_df.shape[0] >= cpus:
            donor_df_chunks = np.array_split(masked_donor_df, cpus)
            chrom_chunks = np.array_split(asgn_chromosomes, cpus)
            pos_chunks = np.array_split(asgn_positions, cpus)

            # Create the list of arguments for the worker function
            chunked_args = list(zip(donor_df_chunks, chrom_chunks, pos_chunks))
            pool_args = [(arg, refGenome, while_round) for arg in chunked_args]

            # Multiprocessing
            with Pool(cpus) as pool:
                results = pool.starmap(process_chunk, pool_args)
            tmp_donor_df = pd.concat(results)
        else:
            tmp_donor_df = process_chunk((masked_donor_df, asgn_chromosomes, asgn_positions), refGenome, while_round)

        tmp_donor_df = tmp_donor_df.reindex(missing_indices)

        # Update only missing rows in the original DataFrame
        donor_df.loc[mask, tmp_donor_df.columns] = tmp_donor_df.values
        
        n_missing = donor_df["updated_ref"].isna().sum()
        while_round += 1

    donor_df = donor_df.dropna()
    donor_df['pos'] = donor_df['pos'].astype(int)
    return donor_df

def pd2vcf(muts_f, driver_muts_f, vafs_f, idx=0, prefix=None) -> pd.DataFrame:

    """
    Convert the pandas DataFrames into a VCF
    """
    def create_info_field(signatures_list_f:pd.DataFrame, driver_genes_f:pd.Series, vafs_f:tuple) -> list:
        info:list = []
        for row in signatures_list_f.itertuples():
            tmp_info = f"AF={vafs_f[row.Index]};MS={row.signature}"

            if 'SBS' in row.signature:
                tmp_info = f"{tmp_info};SBSCTX={row.contexts}"
            elif 'ID' in row.signature:
                tmp_info = f"{tmp_info};IDCTX={row.contexts}"

            if len(row.updated_ref) != 1 and len(row.updated_alt) != 1 and 'Del:M' not in row.contexts:
                tmp_info = f"{tmp_info};HPR={row.updated_ref[1:]}"
            elif 'Del:M' in row.contexts:
                tmp_info = f"{tmp_info};MHR={row.updated_ref[1:]}"

            info.append(tmp_info)
        
        for idx, driver in driver_genes_f.items():
            info.append(f"AF={vafs_f[signatures_list_f.shape[0]+idx]};MS=driver_{driver}")

        return info

    def update_ref_alt_indels_apply(row) -> tuple[str, str]:
        indel_size = len(row['ALT']) - len(row['REF'])
        if (len(row['REF']) != 1 and len(row['ALT']) != 1) and indel_size != 0:
            if indel_size > 0: #INS
                ref = row['REF'][0]
                alt = row['ALT'][:indel_size+1]
            else: #DEL
                ref = row['REF'][:abs(indel_size)+1]
                alt = row['ALT'][0]
            return (ref, alt)
        else:
            return (row['REF'], row['ALT'])

    n_muts = muts_f.shape[0] + driver_muts_f.shape[0]
    vcf = pd.DataFrame({
        '#CHROM': muts_f['chrom'].tolist() + driver_muts_f['chrom'].tolist(),
        'POS': muts_f['pos'].tolist() + driver_muts_f['start'].tolist(),
        'ID': [f"sim{idx+1}"] * n_muts if prefix == None else [prefix] * n_muts,
        'REF': muts_f['updated_ref'].tolist() + driver_muts_f['ref'].tolist(),
        'ALT': muts_f['updated_alt'].tolist() + driver_muts_f['alt'].tolist(),
        'QUAL' : '.',
        'FILTER': '.',
        'INFO': create_info_field(muts_f, driver_muts_f['gene_name'], vafs_f[:n_muts])
    })

    # Update REF and ALT fields
    vcf[['REF', 'ALT']] = vcf.apply(update_ref_alt_indels_apply, axis=1, result_type="expand")

    # Sort the VCF
    vcf = vcf.sort_values(by=['#CHROM', 'POS'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)

    # Filter out some random and very infrequent DNP, TNP and repeated SNPs
    vcf['keep'] = abs(vcf['POS'].diff()) > 2
    vcf = vcf[vcf['keep'].shift(-1, fill_value=False)]
    vcf = vcf.drop(columns=['keep']).reset_index(drop=True)

    return(vcf)







############
# CNA & SV #
############

def select_cna_sv_counts(cna_sv_countModel, nCases, tumor, counts) -> pd.DataFrame:

    """
    Generate CNA and SV for each donor
    """

    # Calculate the total mutations per donor
    counts_totalmut:pd.Series = counts.sum(axis=1)

    # Generate CNA and SV samples
    cna_sv_counts:pd.DataFrame = cna_sv_countModel.generate_samples(nCases*100, var_column='study', var_class=tumor)
    
    # Select CNA-SV events for each donor based on the total number of mutations as link between the two models
    selected_cna_sv_counts:pd.DataFrame = pd.DataFrame()
    for count in counts_totalmut:
        ## Find the index of the closest 'total_mut' in cna_sv_counts
        cna_sv_index:int = (np.abs(cna_sv_counts['total_mut'] - count)).argmin()
        ## Select the best case
        tmp:pd.DataFrame = cna_sv_counts.iloc[[cna_sv_index],:]
        selected_cna_sv_counts = pd.concat([selected_cna_sv_counts, tmp], ignore_index=True)
        ## Drop the selected row to avoid duplicate selections
        cna_sv_counts = cna_sv_counts.drop(index=cna_sv_index).reset_index(drop=True)

    return(selected_cna_sv_counts)

def select_cnas(cnas_df, nCNAs, lenCNA, iterations=100000) -> list:
    
    """
    Select a subset of CNAs such that the sum of the subset is as close as possible to the generated length.
    """
    
    best_sum_difference:float = float('inf')
    for _ in range(iterations):
        subset:pd.DataFrame = cnas_df.sample(nCNAs)
        subset_sum = subset['len'].sum()
        
        # Check if the current subset's sum is closer to the generated lenght
        if abs(subset_sum - lenCNA) < best_sum_difference:
            best_sum_difference = abs(subset_sum - lenCNA)
            best_subset = subset
        
        # Early exit if we reach a close subset
        if best_sum_difference < 0.1*lenCNA:
            break
    
    return(best_subset, int(np.sum(best_subset['len'])))

def rescue_missing_chroms(cnas_df, gender, keys, chrom_size_dict) -> pd.DataFrame:

    """
    A function to recover missing chromsomes for CNA
    """

    # Detect missing chromosomes
    all_chroms:set = set(cnas_df['chrom'].astype(str))
    simulated_chroms:list = [str(chrom) for chrom in keys if str(chrom) in all_chroms]
    simulated_indices:dict = {chrom: keys.index(chrom) for chrom in simulated_chroms}
    missing_chroms:list = [str(chrom) for chrom in keys if str(chrom) not in all_chroms]

    if gender == 'F' and 'Y' in missing_chroms:
        missing_chroms.remove('Y')

    # Select the next available chromosome as template
    for missing_chrom in missing_chroms:
        missing_index:int = keys.index(missing_chrom)
        remaining_simulated = [chrom for chrom, index in simulated_indices.items() if index > missing_index]
        if remaining_simulated:
            next_chrom:str = remaining_simulated[0]
        else:
            next_chrom:str = random.choice(simulated_chroms)
        next_chrom_row:pd.DataFrame = pd.DataFrame([cnas_df[cnas_df['chrom'] == next_chrom].iloc[0].copy()])
        next_chrom_row['chrom'] = missing_chrom
        next_chrom_row['pos'] = int(chrom_size_dict[missing_chrom])
        cnas_df = pd.concat([cnas_df, next_chrom_row], ignore_index=True)
    cnas_df = cnas_df.sort_values(by=['chrom', 'pos'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)

    return(cnas_df)

def adjust_cna_position(cnas_df, gender) -> pd.DataFrame:

    """
    Adjust the lengths of the CNAs to fit the assigned chromosomes positions
    """

    keys:list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']
    values:list = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560,59373566]
    chrom_size_dict:dict = dict(zip(keys, values))

    # Add missing chromsomes
    cnas_df = rescue_missing_chroms(cnas_df, gender, keys, chrom_size_dict)
    
    # Get the maximum position for each chrom
    grouped:pd.DataFrame = cnas_df.groupby('chrom')['pos'].max().reset_index()
    grouped.columns = ['chrom', 'max_pos']
    
    adjusted_cnas_df:pd.DataFrame = pd.DataFrame()
    for chrom, max_pos in grouped.itertuples(index=False):
        real_chrom_length:int = chrom_size_dict[chrom]
        ratio:float = real_chrom_length / max_pos
        
        # Adjust 'pos' for each chrom group and set max 'pos' value to real chrom length
        tmp_df:pd.DataFrame = cnas_df[cnas_df['chrom'] == chrom].copy()
        tmp_df['pos'] = (tmp_df['pos'] * ratio).round().astype(int)
        tmp_df['pos'][-1] = real_chrom_length
        adjusted_cnas_df = pd.concat([adjusted_cnas_df, tmp_df], ignore_index=True)
    
    adjusted_cnas_df = adjusted_cnas_df.sort_values(by=['chrom', 'pos'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)

    # Set the first start position of each chromsome as 1
    adjusted_cnas_df['start'] = adjusted_cnas_df.groupby('chrom')['pos'].transform(lambda x: np.insert(x.values + 1, 0, 1)[:-1])
    adjusted_cnas_df = adjusted_cnas_df.rename(columns={"pos": "end"})

    return(adjusted_cnas_df)

def combine_same_cna_events(cnas_df) -> pd.DataFrame:
    
    """
    Combines consecutive cnas with the same 'major_cn' and 'minor_cn' events
    """

    # Group by unique clusters of consecutive rows with the same chrom, major_cn, and minor_cn
    combined_df:pd.DataFrame = (
        cnas_df.groupby((cnas_df[['chrom', 'major_cn', 'minor_cn', 'donor_id', 'study']].shift() != cnas_df[['chrom', 'major_cn', 'minor_cn', 'donor_id', 'study']]).any(axis=1).cumsum())
          .agg(
              chrom=('chrom', 'first'),
              start=('start', 'min'),
              end=('end', 'max'),
              major_cn=('major_cn', 'first'),
              minor_cn=('minor_cn', 'first'),
              donor_id=('donor_id', 'first'),
              study=('study', 'first')
          )
          .reset_index(drop=True)
    )
    
    return(combined_df)

def generate_driver_cna_major_cn(cn) -> int:

    """Transform driver CN number into the major/minor allele style"""

    if cn == -2:
        return 0
    elif cn == -1:
        return 1
    else:
        return cn+1

def generate_driver_cna_minor_cn(cn) -> int:

    """Transform driver CN number into the major/minor allele style"""

    if cn < 0:
        return 0
    else:
        return 1

def add_driver_cnas(case_cnas, tumor, refGenome) -> pd.DataFrame:

    """Generates driver CNA events"""

    cna_drivers:pd.DataFrame = pd.read_csv('/oncoGAN/trained_models/drivers/cna_drivers.csv')
    cna_drivers = cna_drivers[cna_drivers['study'] == tumor].reset_index(drop=True)

    # Randomly select events
    cna_drivers.loc[:, 'to_simulate'] = cna_drivers['perc'].apply(lambda x: np.random.rand() < x)
    cna_drivers = cna_drivers[cna_drivers['to_simulate']].reset_index(drop=True)

    # If there are no events to simulate skip
    if cna_drivers.empty:
        return(case_cnas)

    # Check if there are overlap events
    range_dicts:dict = {}
    for _,row in cna_drivers.iterrows():
        range_dicts[row['range_id']] = row['overlap'].split(";")

    removed_list:list = []
    for range_id in cna_drivers['range_id']:
        if range_id in removed_list:
            continue
        overlap_list:list = [range_id]
        for all_range_id in range_dicts:
            if range_id in range_dicts[all_range_id]:
                overlap_list.append(all_range_id)
        overlap_list = [i for i in overlap_list if i not in removed_list]
        if len(overlap_list) > 1:
            to_remove:str = random.choice(overlap_list)
            cna_drivers = cna_drivers.loc[~(cna_drivers['range_id'] == to_remove)]
            removed_list.append(to_remove)
    cna_drivers = cna_drivers.reset_index(drop=True)

    # Remove unnecessary columns
    cna_drivers = cna_drivers.drop(columns=['perc', 'range_id', 'overlap', 'to_simulate'])

    # Create major_cna and minor_cn columns
    cna_drivers.loc[:, 'major_cn'] = cna_drivers['cna'].apply(lambda x: generate_driver_cna_major_cn(x))
    cna_drivers.loc[:, 'minor_cn'] = cna_drivers['cna'].apply(lambda x: generate_driver_cna_minor_cn(x))

    # Adapt dataframe format to somatic cna style
    cna_drivers = cna_drivers.drop(columns=['cna'])
    cna_drivers['id'] = 'driver'
    cna_full:pd.DataFrame = pd.concat([case_cnas, cna_drivers])
    cna_full = cna_full.sort_values(by=['chrom', 'start'], key=lambda chrom: chrom.map(sort_by_int_chrom)).reset_index(drop=True)

    cna_full_updated:pd.DataFrame = pd.DataFrame()
    chrom_lengths:pd.DataFrame = pd.read_csv(f'{refGenome}.fai', delimiter='\t', names=['chrom', 'length'], usecols=[0,1])
    for chrom, group in cna_full.groupby('chrom', sort=False):
        group = group.reset_index(drop=True)
        rows_to_drop:list = []
        rows_to_append:pd.DataFrame = pd.DataFrame()
        i:int = 0
        while i < len(group):
            if not pd.isna(group.loc[i, 'id']):
                chrom_end:int = chrom_lengths.loc[chrom_lengths['chrom'].astype(str) == str(chrom), 'length'].reset_index(drop=True)[0]
                ## If the driver CNA is the first one on the chromosome
                if i == 0:
                    current_end:int = group.loc[i, 'end']
                    for j in range(i + 1, len(group)):
                        if group.loc[j, 'end'] < current_end:
                            rows_to_drop.append(j)
                        elif group.loc[j, 'end'] == current_end:
                            rows_to_drop.append(j)
                            break   
                        else:
                            group.loc[j, 'start'] = group.loc[i, 'end'] + 1
                            break
                ## If the driver CNA is the last one on the chromosome
                elif i == len(group)-1:
                    current_start:int = group.loc[i, 'start']
                    current_end:int = group.loc[i, 'end']
                    for j in range(len(group)-2, -1, -1):
                        if group.loc[j, 'start'] > current_start:
                            rows_to_drop.append(j)
                        elif group.loc[j, 'start'] == current_start:
                            rows_to_drop.append(j)
                            break   
                        else:
                            group.loc[j, 'end'] = group.loc[i, 'start'] - 1
                            break
                    ## Keep a normal diploid chromosome after the driver event if there are many events in the chrom or the same event if there was only one event before adding the driver event
                    if current_end < chrom_end and len(group) > 2:
                        rows_to_append = pd.DataFrame({'len': [0], 'major_cn': [1], 'minor_cn': [1], 'study': [group.loc[i, 'study']], 'end': [chrom_end], 'chrom': [chrom], 'start': [group.loc[i, 'end'] + 1], 'id': [group.loc[i, 'id']]})
                    elif current_end < chrom_end and len(group) == 2:
                        rows_to_append = pd.DataFrame({'len': [0], 'major_cn': [group.loc[i - 1, 'major_cn']], 'minor_cn': [group.loc[i - 1, 'minor_cn']], 'study': [group.loc[i-1, 'study']], 'end': [chrom_end], 'chrom': [chrom], 'start': [group.loc[i, 'end'] + 1], 'id': [group.loc[i, 'id']]})
                    
                # If the driver event happens completely within the previous segment
                elif group.loc[i, 'end'] < group.loc[i - 1, 'end']:
                    ## Update previous end
                    prev_end:int = group.loc[i - 1, 'end']
                    current_start:int = group.loc[i, 'start']
                    group.loc[i - 1, 'end'] = current_start - 1

                    ## Update current end
                    group.loc[i, 'end'] = prev_end
                # If the driver is between two segments overwrite the segment with which it overlaps the most
                elif group.loc[i, 'end'] < group.loc[i + 1, 'end']:
                    ## Check segment overlap
                    prev_diff:int = abs(group.loc[i - 1, 'end'] - group.loc[i, 'start'])
                    next_diff:int = abs(group.loc[i, 'end'] - group.loc[i + 1, 'start'])

                    if prev_diff >= next_diff:
                        group.loc[i, 'start'] = group.loc[i - 1, 'start']
                        group.loc[i + 1, 'start'] = group.loc[i, 'end'] + 1
                        rows_to_drop.append(i - 1)
                    else:
                        group.loc[i - 1, 'end'] = group.loc[i, 'start'] - 1
                        group.loc[i, 'end'] = group.loc[i + 1, 'end']
                        rows_to_drop.append(i + 1)
                # If the driver contains many downstream segments
                else:
                    ## Update the start
                    prev_end:int = group.loc[i - 1, 'end']
                    group.loc[i, 'start'] = prev_end + 1

                    ## Update the end
                    current_end:int = group.loc[i, 'end']
                    for j in range(i + 1, len(group)):
                        if group.loc[j, 'end'] < current_end:
                            rows_to_drop.append(j)
                        elif group.loc[j, 'end'] == current_end:
                            rows_to_drop.append(j)
                            break   
                        else:
                            group.loc[i, 'end'] = group.loc[j, 'start'] - 1
                            break
                i += 1
            else:
                i += 1
        group = group.drop(rows_to_drop).reset_index(drop=True)
        cna_full_updated = pd.concat([cna_full_updated, group, rows_to_append]).reset_index(drop=True)

    # Fix the start of driver CNAs if they are the first event of the chromosome
    cna_full_updated.loc[cna_full_updated['start'] == 0, 'start'] = 1

    return(cna_full_updated)

def simulate_cnas(nCNAs, lenCNA, tumor, cnaModel, gender, refGenome, idx=0, prefix=None) -> pd.DataFrame:
    
    """
    Generate CNAs
    """

    max_length:int = 3036303846 if gender == "F" else 3095677412 if gender == "M" else None
    case_cnas:pd.DataFrame = cnaModel.generate_samples(nCNAs*100, var_column='study', var_class=tumor)
    case_cnas = case_cnas[case_cnas['major_cn'] >= case_cnas['minor_cn']].reset_index(drop=True)

    # Update CNAs length
    lenCNA = round(np.exp(lenCNA))
    lenCNA_normal:int = max_length-lenCNA
    case_cnas['len'] = round(np.exp(case_cnas['len'])*10000)

    # Process altered haplotypes
    case_cnas_altered:pd.DataFrame = case_cnas[(case_cnas['major_cn'] != 1) | (case_cnas['minor_cn'] != 1)]
    selected_case_cnas_altered, selected_lenCNA = select_cnas(case_cnas_altered, nCNAs, lenCNA)
    len_adjust_ratio:float = lenCNA/selected_lenCNA
    selected_case_cnas_altered['len'] = round(selected_case_cnas_altered['len']*len_adjust_ratio).astype(int)

    # Process normal haplotypes
    case_cnas_normal:pd.DataFrame = case_cnas[(case_cnas['major_cn'] == 1) & (case_cnas['minor_cn'] == 1)]
    case_cnas_normal['cumsum'] = case_cnas_normal['len'].cumsum()
    selected_case_cnas_normal = case_cnas_normal[case_cnas_normal['cumsum'] <= lenCNA_normal]
    len_adjust_ratio_normal:float = lenCNA_normal/np.sum(selected_case_cnas_normal['len'])
    selected_case_cnas_normal['len'] = round(selected_case_cnas_normal['len']*len_adjust_ratio_normal).astype(int)
    selected_case_cnas_normal = selected_case_cnas_normal.drop(columns=['cumsum'])

    # Merge and shuffle CNAs
    case_cnas = pd.concat([selected_case_cnas_altered, selected_case_cnas_normal])
    case_cnas = case_cnas.sample(frac=1, replace=False, random_state=1).reset_index(drop=True)
    
    # Assign chromosomes and adjust CNAs by chromosome
    case_cnas = assign_chromosome(case_cnas, cna=True, gender=gender)
    case_cnas = adjust_cna_position(case_cnas, gender)
    
    # Sort by real integer chrom order
    case_cnas = case_cnas.sort_values(by=['chrom', 'start'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)

    # Add driver events
    case_cnas = add_driver_cnas(case_cnas, tumor, refGenome)

    # Add donor id
    if prefix == None:
        case_cnas["donor_id"] = f"sim{idx}"
    else:
        case_cnas["donor_id"] = prefix
    case_cnas = case_cnas[["chrom", "start", "end", "major_cn", "minor_cn", "donor_id", "study"]]

    # Combine same CNAs events
    case_cnas = combine_same_cna_events(case_cnas)

    # Create an ID for each CNA segment
    case_cnas['cna_id'] = 'cna' + case_cnas.index.astype(str)

    return(case_cnas)

def assign_cna_plot_color(y) -> str:

    """
    Asign a color to CNA segements depending on the copy number
    """
    y = int(y)
    if y == 1:
        return "Normal"
    elif y > 1:
        return "Gain"
    else:
        return "Loss"
    
def plot_cnas(cna_profile, sv_profile, tumor, output, idx=0, prefix=None) -> None:

    """
    Plot CNA segments
    """

    cna_profile_copy:pd.DataFrame = cna_profile.copy()
    sv_profile_copy:pd.DataFrame = sv_profile.copy()

    # Change chrom format to str
    cna_profile_copy['chrom'] = cna_profile_copy['chrom'].astype(str)
    sv_profile_copy['chrom1'] = sv_profile_copy['chrom1'].astype(str)
    sv_profile_copy['chrom2'] = sv_profile_copy['chrom2'].astype(str)

    # Define chromosome lengths
    chrom_list:list = list(range(1, 23)) + ["X", "Y"]
    cumlength_list:list = [0, 249250621, 492449994, 690472424, 881626700, 1062541960, 1233657027, 1392795690, 1539159712, 1680373143, 1815907890, 1950914406, 2084766301, 2199936179, 2307285719, 2409817111, 2500171864, 2581367074, 2659444322, 2718573305, 2781598825, 2829728720, 2881033286, 3036303846]
    cumlength_end_list:list = [249250621, 492449994, 690472424, 881626700, 1062541960, 1233657027, 1392795690, 1539159712, 1680373143, 1815907890, 1950914406, 2084766301, 2199936179, 2307285719, 2409817111, 2500171864, 2581367074, 2659444322, 2718573305, 2781598825, 2829728720, 2881033286, 3036303846, 3095677412]
    chrom_size_list:list = [end - start for start, end in zip(cumlength_list, cumlength_end_list)]
    chrom_cumsum_length:pd.DataFrame = pd.DataFrame({
        'chrom': [str(chr) for chr in chrom_list],
        'cumlength': cumlength_list,
        'cumlength_end': cumlength_end_list,
        'chrom_size': chrom_size_list
    })

    if 'Y' not in set(cna_profile_copy['chrom']):
        chrom_cumsum_length = chrom_cumsum_length.iloc[:-1]

    # Preprocess the data
    ## Pivot longer SVs
    sv_profile_copy['sv_id'] = ['sv{}'.format(i) for i in range(len(sv_profile_copy))]
    ### Inversions
    sv_profile_inv = sv_profile_copy[sv_profile_copy['svclass'].isin(['h2hINV', 't2tINV'])]
    sv_profile_inv = sv_profile_inv.rename(columns={'chrom1': 'chrom', 'start1': 'start', 'start2': 'end'})
    sv_profile_inv = sv_profile_inv[['chrom', 'start', 'end', 'svclass', 'sv_id']]
    ### Translocations
    sv_profile_tra = sv_profile_copy[sv_profile_copy['svclass']=='TRA']
    sv_profile_long_tra = pd.concat([
        sv_profile_tra[['chrom1', 'start1', 'end1', 'svclass', 'sv_id']].rename(columns={'chrom1': 'chrom', 'start1': 'start', 'end1': 'end'}),
        sv_profile_tra[['chrom2', 'start2', 'end2', 'svclass', 'sv_id']].rename(columns={'chrom2': 'chrom', 'start2': 'start', 'end2': 'end'})])
    sv_profile_long_tra.reset_index(drop=True, inplace=True)
    ### Concatenate
    sv_profile_long = pd.concat([sv_profile_inv, sv_profile_long_tra])
    ## Left join CNAs and SVs with chromosome lengths
    cna_profile_copy = cna_profile_copy.merge(chrom_cumsum_length[['chrom', 'cumlength']], on='chrom', how='left')
    sv_profile_long = sv_profile_long.merge(chrom_cumsum_length[['chrom', 'cumlength']], on='chrom', how='left')
    ## Update segment positions
    cna_profile_copy['start'] = cna_profile_copy['start'] + cna_profile_copy['cumlength']
    cna_profile_copy['end'] = cna_profile_copy['end'] + cna_profile_copy['cumlength']
    sv_profile_long['start'] = sv_profile_long['start'] + sv_profile_long['cumlength']
    sv_profile_long['end'] = sv_profile_long['end'] + sv_profile_long['cumlength']
    ## Remove unnecesary columns
    cna_profile_copy = cna_profile_copy.drop(columns=['cumlength'])
    sv_profile_long = sv_profile_long.drop(columns=['cumlength'])
    ## Add a group column, one for each segment
    cna_profile_copy['group'] = np.arange(1, len(cna_profile_copy) + 1)
    sv_profile_long['group'] = np.arange(1, len(sv_profile_long) + 1)
    ## Calculate linewidth
    cna_profile_copy['linewidth'] = np.where(cna_profile_copy['major_cn'] == cna_profile_copy['minor_cn'], 5, 3)
    ## Pivot longer 'major_cn' and 'minor_cn' columns for CNAs
    ### CNAs
    id_vars:list = [col for col in cna_profile_copy.columns if col not in ['major_cn', 'minor_cn']]
    cna_profile_long = cna_profile_copy.melt(
        id_vars=id_vars,
        value_vars=['major_cn', 'minor_cn'],
        var_name='cn',
        value_name='y'
    )
    cna_profile_long = cna_profile_long[cna_profile_long['y'].notna()]
    ### SVs
    ymax = max(cna_profile_long['y'])
    sv_profile_long['overlap'] = sv_profile_long['start'] <= sv_profile_long['end'].shift()
    overlap = sv_profile_long['overlap'].tolist()
    for i in range(1, len(overlap)):
        if overlap[i] and overlap[i - 1]: 
            overlap[i] = False
    sv_profile_long['overlap'] = overlap
    sv_profile_long['y'] = sv_profile_long['svclass'].apply(lambda x: ymax+1.5 if x == 'TRA' else ymax+1) + sv_profile_long['overlap'].apply(lambda x: 0.2 if x else 0)
    sv_profile_long = sv_profile_long.drop(columns=['overlap'])
    ## Pivot longer 'start' and 'end' columns
    ### CNAs
    id_vars:list = [col for col in cna_profile_long.columns if col not in ['start', 'end']]
    cna_profile_long:pd.DataFrame = cna_profile_long.melt(
        id_vars=id_vars,
        value_vars=['start', 'end'],
        var_name='position',
        value_name='x'
    )
    cna_profile_long = cna_profile_long.sort_values('group').reset_index(drop=True)
    ### SVs
    id_vars:list = [col for col in sv_profile_long.columns if col not in ['start', 'end']]
    sv_profile_long_long:pd.DataFrame = sv_profile_long.melt(
        id_vars=id_vars,
        value_vars=['start', 'end'],
        var_name='position',
        value_name='x'
    )
    sv_profile_long_long = sv_profile_long_long.sort_values('group').reset_index(drop=True)

    # Plot
    ## Assign colors
    cna_profile_long['color'] = cna_profile_long['y'].apply(assign_cna_plot_color)
    cna_profile_long['color'] = pd.Categorical(
        cna_profile_long['color'],
        categories=['Gain', 'Normal', 'Loss'],
        ordered=True
    )
    cna_color_mapping:dict = {'Gain': "#2a9d8f", 'Normal': "#264653", 'Loss': "#f4a261"}
    sv_profile_long_long['svclass'] = pd.Categorical(
        sv_profile_long_long['svclass'],
        categories=['h2hINV', 't2tINV', 'TRA'],
        ordered=True
    )
    sv_color_mapping:dict = {'h2hINV': "#FF9898FF", 't2tINV': "#DC3262", 'TRA': "#7A0425"}
    ## Create a figure and axis object
    plt.figure(figsize=(16, 8))
    ## Plot the segments
    ### CNAs
    for (grp, cn), data in cna_profile_long.groupby(['group', 'cn']):
        plt.plot(
            data['x'], 
            data['y'], 
            color=cna_color_mapping[data['color'].iloc[0]], 
            linewidth=data['linewidth'].iloc[0],
            label='_nolegend_',
            solid_capstyle='butt'
        )
    ### SVs
    for grp, data in sv_profile_long_long.groupby(['group']):
        plt.plot(
            data['x'], 
            data['y'], 
            color=sv_color_mapping[data['svclass'].iloc[0]], 
            linewidth=4,
            label='_nolegend_',
            solid_capstyle='butt'
        )
    ## Separate chroms by vertical dashed lines
    chrom_vlines:pd.DataFrame = chrom_cumsum_length.iloc[:-1]
    for x in chrom_vlines['cumlength_end']:
        plt.axvline(x=x, color='gray', linewidth=0.2, linestyle='--')
    ## Calculate ymax for setting y-axis limits and label positions
    ymax:int = int(cna_profile_long['y'].max())
    plt.ylim(-0.5, ymax + 2.2)
    plt.yticks(range(0,ymax+1))
    ## Calculate chromosome label positions
    ### X
    chrom_cumsum_length['x_label_pos'] = chrom_cumsum_length['cumlength'] + chrom_cumsum_length['chrom_size'] / 2
    ### Y
    num_labels:int = len(chrom_cumsum_length)
    y_positions:list = [ymax + 2 if i % 2 == 0 else ymax + 2.3 for i in range(num_labels - 1)]
    y_positions.append(ymax + 2)  # Add the last position
    chrom_cumsum_length['y_label_pos'] = y_positions
    ## Add chromosome labels
    for _, row in chrom_cumsum_length.iterrows():
        plt.text(
            row['x_label_pos'], 
            row['y_label_pos'], 
            str(row['chrom']), 
            ha='center', 
            va='bottom', 
            fontsize=11
        )
    ## Minimal style
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.despine(trim=True, left=False, bottom=False)
    ## Set axis labels and subtitle
    ax = plt.gca()
    if prefix == None:
        plt.title(f'{tumor} - Donor{idx}', fontsize=16, y=1.1)
    else:
        plt.title(f'{tumor} - {prefix}', fontsize=16, y=1.1)
    plt.xlabel('Genome', fontsize=14)
    plt.tick_params(axis='x', which='both', length=0, labelbottom=False)
    plt.ylabel('CNA', fontsize=14)
    plt.tick_params(axis='y', which='both', length=5)
    loc, label = plt.yticks()
    ax.yaxis.set_label_coords(-0.02, np.mean(loc), transform=ax.get_yaxis_transform())
    ## Adjust legend position
    handles = [
        plt.Line2D([0], [0], color="#FF9898", lw=4, linestyle='-', label='h2hINV'),
        plt.Line2D([0], [0], color="#DC3262", lw=4, linestyle='-', label='t2tINV'),
        plt.Line2D([0], [0], color="#7A0425", lw=4, linestyle='-', label='TRA')
    ]
    plt.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        fontsize=10,
        frameon=False
    )

    # Save the plot
    plt.savefig(output)

def get_sv_coordinates(n, svModel, gender) -> pd.DataFrame:
    
    """
    Generate genomic coordinates for SV
    """

    # Generate position ranks
    tmp_pos:pd.DataFrame = svModel['pos']['step1'].sample(num_rows = round(n*5))

    # Remove Y coordinates
    if gender == "F":
        y_ranks:list = ['[3.03e+07;3.06e+07)', '[3.06e+07;3.09e+07)', '[3.09e+07;3.1e+07]']
        tmp_pos = tmp_pos[~tmp_pos['rank'].isin(y_ranks)]
    
    # Keep the correct number of SV
    tmp_pos = tmp_pos.sample(n=n)

    # Simulate the exact positions
    step1:pd.Series = tmp_pos['rank'].value_counts()
    positions:pd.DataFrame = pd.DataFrame()
    for rank, m in zip(step1.index, step1):
        try:
            positions = pd.concat([positions, svModel['pos'][rank].sample(num_rows = m)])
        except KeyError:
            continue
    positions['start'] = positions['start']*100
    positions.reset_index(drop=True, inplace=True)

    return(positions['start'])

def check_sv_strand_patterns(sv_profile) -> pd.DataFrame:

    """
    Check SV strand patterns
    """

    expected_patterns:dict = {
        'DEL': ('+', '-'),
        'DUP': ('-', '+'),
        'h2hINV': ('+', '+'),
        't2tINV': ('-', '-'),
    }

    # Get expected strands for the svclass
    sv_profile['expected_strand1'], sv_profile['expected_strand2'] = zip(*sv_profile['svclass'].apply(lambda svclass: expected_patterns.get(svclass, (None, None))))

    # If the current strands don't match the expected pattern, invert them
    sv_profile['keep'] = False
    for index, row in sv_profile.iterrows():
        if row['svclass'] == "TRA" or (row['strand1'] == row['expected_strand1'] and row['strand2'] == row['expected_strand2']):
            sv_profile.at[index, 'keep'] = True
        elif row['strand1'] == row['expected_strand2'] and row['strand2'] == row['expected_strand1']:
            # Swap values
            sv_profile.at[index, 'start1'], sv_profile.at[index, 'start2'] = row['start2'], row['start1']
            sv_profile.at[index, 'chrom1'], sv_profile.at[index, 'chrom2'] = row['chrom2'], row['chrom1']
            sv_profile.at[index, 'strand1'], sv_profile.at[index, 'strand2'] = row['strand2'], row['strand1']
            sv_profile.at[index, 'keep'] = True
        else:
            sv_profile.at[index, 'keep'] = False
    
    # Generate end1 and end2
    sv_profile['end1'] = sv_profile['start1'] + 1
    sv_profile['end2'] = sv_profile['start2'] + 1

    # Remove SV with the wrong pattern
    sv_profile = sv_profile[sv_profile['keep']].drop(columns=['keep'])
    sv_profile.reset_index(drop=True, inplace=True)

    # Sort columns
    sv_profile = sv_profile[["chrom1", "start1", "end1", "chrom2", "start2", "end2", "strand1", "strand2", "svclass"]]

    return(sv_profile)

def check_inv_overlaps(sv_profile) -> pd.DataFrame:

    """
    Check if there is any overlap between inversions
    """

    # Check for overlaps within each chromosome
    sv_profile['n_overlaps'] = 0
    for chrom, group in sv_profile.groupby('chrom1'):
        group_indices:list = group.index
        for i in range(len(group_indices)):
            current_row:pd.Series = group.iloc[i]
            ## Find the next row
            n:int = 0
            while True:
                try:
                    next_row:pd.Series = group.iloc[i+1+n]
                    ## Check if there is an overlap
                    if current_row['start2'] > next_row['start1']:
                        n += 1
                    else:
                        sv_profile.loc[group_indices[i], 'n_overlaps'] = n
                        break
                except IndexError:
                    sv_profile.loc[group_indices[i], 'n_overlaps'] = n
                    break

    # Remove only events that overlap with events that also overlap
    sv_profile['keep'] = True
    for chrom, group in sv_profile.groupby('chrom1'):
        for idx, row in group.iterrows():
            n_overlaps:int = row['n_overlaps']
            if n_overlaps == 0:
                continue
            else:
                check_idx:list = list(range(idx + 1, idx + 1 + n_overlaps))
                overlap_sum:int = group.loc[group.index.isin(check_idx), 'n_overlaps'].sum()
                if overlap_sum != 0:
                    sv_profile.loc[idx, 'keep'] = False
                else:
                    continue
    
    sv_profile = sv_profile[sv_profile['keep']]
    sv_profile = sv_profile.drop(columns=['n_overlaps', 'keep']).reset_index(drop=True)

    return(sv_profile)

def sort_cna_ids(row):

    """
    Sort cna_id and allele columns for each TRA event
    """

    cna_id_list:list = row['cna_id'].split(',')
    alleles_list:list = row['allele'].split(',')
    
    # Extract numeric parts
    cna_id1:int = int(cna_id_list[0][3:])
    cna_id2:int = int(cna_id_list[1][3:])
    if cna_id1 > cna_id2:
        row['cna_id'] = f'{cna_id_list[1]},{cna_id_list[0]}'
        row['allele'] = f'{alleles_list[1]},{alleles_list[0]}'
        return row
    else:
        return row

def check_tra_overlaps(sv_profile) -> pd.DataFrame:

    """
    Check if there is any overlap between translocations
    """

    # Pivot longer second chrom events
    sv_profile['sv_id'] = ['sv{}'.format(i) for i in range(len(sv_profile))]
    sv_profile_long = pd.concat([
        sv_profile[['sv_id', 'chrom1', 'start1', 'end1', 'strand1', 'svclass', 'cna_id', 'allele']].rename(columns={'chrom1': 'chrom', 'start1': 'start', 'end1': 'end', 'strand1': 'strand'}),
        sv_profile[['sv_id', 'chrom2', 'start2', 'end2', 'strand2', 'svclass', 'cna_id', 'allele']].rename(columns={'chrom2': 'chrom', 'start2': 'start', 'end2': 'end', 'strand2': 'strand'})])
    sv_profile_long.reset_index(drop=True, inplace=True)
    ## Sort the new dataframe
    sv_profile_long[["start", "end"]] = sv_profile_long[["start", "end"]].astype(int)
    sv_profile_long['chrom'] = sv_profile_long['chrom'].apply(chrom2int)
    sv_profile_long = sv_profile_long.sort_values(by=['chrom', 'start'], ignore_index=True)
    sv_profile_long['chrom'] = sv_profile_long['chrom'].apply(chrom2str)
    
    # Check for overlaps within each chromosome
    sv_profile_long['n_overlaps'] = 0
    for chrom, group in sv_profile_long.groupby('chrom'):
        group_indices:list = group.index
        for i in range(len(group_indices)):
            current_row:pd.Series = group.iloc[i]
            ## Find the next row
            n:int = 0
            while True:
                try:
                    next_row:pd.Series = group.iloc[i+1+n]
                    ## Check if there is an overlap
                    if current_row['end'] > next_row['start']:
                        n += 1
                    else:
                        sv_profile_long.loc[group_indices[i], 'n_overlaps'] = n
                        break
                except IndexError:
                    sv_profile_long.loc[group_indices[i], 'n_overlaps'] = n
                    break

    # Remove only events that overlap with events that also overlap
    sv_profile_long['keep'] = True
    for chrom, group in sv_profile_long.groupby('chrom'):
        for idx, row in group.iterrows():
            n_overlaps:int = row['n_overlaps']
            if n_overlaps == 0:
                continue
            else:
                check_idx:list = list(range(idx + 1, idx + 1 + n_overlaps))
                overlap_sum:int = group.loc[group.index.isin(check_idx), 'n_overlaps'].sum()
                if overlap_sum != 0:
                    sv_profile_long.loc[idx, 'keep'] = False
                else:
                    continue

    sv_profile_long = sv_profile_long[sv_profile_long['keep']]
    sv_profile_long = sv_profile_long.drop(columns=['n_overlaps', 'keep']).reset_index(drop=True)
    sv_profile_long = sv_profile_long.groupby('sv_id').filter(lambda group: len(group) == 2)

    # Pivot wider the dataframe to the original shape
    sv_profile = sv_profile_long.groupby('sv_id').apply(
        lambda group: pd.Series({
            'chrom1': group.iloc[0]['chrom'],
            'start1': group.iloc[0]['start'],
            'end1': group.iloc[0]['end'],
            'chrom2': group.iloc[1]['chrom'],
            'start2': group.iloc[1]['start'],
            'end2': group.iloc[1]['end'],
            'strand1': group.iloc[0]['strand'],
            'strand2': group.iloc[1]['strand'],
            'svclass': group.iloc[0]['svclass'],
            'cna_id': group.iloc[0]['cna_id'],
            'allele': group.iloc[0]['allele']})).reset_index()

    # Restore the cna_id and allele position after sorting the TRA
    sv_profile = sv_profile.apply(sort_cna_ids, axis=1)

    return(sv_profile)

def sort_sv(sv) -> pd.DataFrame: 

    """
    Sort SV dataframe
    """

    if sv.empty:
        return(sv)
    else:
        sv[["start1", "end1", "start2", "end2"]] = sv[["start1", "end1", "start2", "end2"]].astype(int)
        sv['chrom1'] = sv['chrom1'].apply(chrom2int)
        sv['chrom2'] = sv['chrom2'].apply(chrom2int)
        sv = sv.sort_values(by=['chrom1', 'start1', 'chrom2', 'start2'], ignore_index=True)
        sv['chrom1'] = sv['chrom1'].apply(chrom2str)
        sv['chrom2'] = sv['chrom2'].apply(chrom2str)    
        return(sv)

def cna2sv_dupdel(cna) -> pd.DataFrame:

    """
    Automatically create DUP/DEL SVs based on CNA events
    """
    
    rows:list = []
    for _,row in cna.iterrows():
        chrom, start, end, major_cn, minor_cn, cna_id,  = row['chrom'], row['start'], row['end'], row['major_cn'], row['minor_cn'], row['cna_id']
    
        # Duplications
        if major_cn > 1:
            rows.append({
                "chrom1": chrom, "start1": start, "end1": start + 1,
                "chrom2": chrom, "start2": end, "end2": end + 1,
                "strand1": "-", "strand2": "+", "svclass": "DUP",
                "cna_id": cna_id, "allele": "major"})
        if minor_cn > 1:
            rows.append({
                "chrom1": chrom, "start1": start, "end1": start + 1,
                "chrom2": chrom, "start2": end, "end2": end + 1,
                "strand1": "-", "strand2": "+", "svclass": "DUP",
                "cna_id": cna_id, "allele": "minor"})
        # Deletions
        if major_cn < 1:
            rows.append({
                "chrom1": chrom, "start1": start, "end1": start + 1,
                "chrom2": chrom, "start2": end, "end2": end + 1,
                "strand1": "+", "strand2": "-", "svclass": "DEL",
                "cna_id": cna_id, "allele": "major"})
        if minor_cn < 1:
            rows.append({
                "chrom1": chrom, "start1": start, "end1": start + 1,
                "chrom2": chrom, "start2": end, "end2": end + 1,
                "strand1": "+", "strand2": "-", "svclass": "DEL",
                "cna_id": cna_id, "allele": "minor"})
    
    sv_dupdel:pd.DataFrame = pd.DataFrame(rows)
    return(sv_dupdel)

def find_closest_range(row, cna) -> tuple:

    """
    Find the closest CNA event to each of the simulated SVs
    """

    cna_copy = cna.copy()

    # Remove homozygous CNA deletions events
    cna_hom_del:list = list(cna_copy.loc[cna_copy['major_cn'] == 0, 'cna_id'])

    # Convert positions to a continous range
    keys:list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']
    values:list = [0,249250621,492449994,690472424,881626700,1062541960,1233657027,1392795690,1539159712,1680373143,1815907890,1950914406,2084766301,2199936179,2307285719,2409817111,2500171864,2581367074,2659444322,2718573305,2781598825,2829728720,2881033286,3036303846]
    chrom_cumsum_dict:dict = dict(zip(keys, values))
    cna_copy['start_continous'] = cna_copy.apply(lambda x: x['start'] + chrom_cumsum_dict[str(x['chrom'])], axis=1)
    row['start1'] = int(row['start1']) + chrom_cumsum_dict[str(row['chrom1'])]
    row['start2'] = int(row['start2']) + chrom_cumsum_dict[str(row['chrom2'])]

    # Find closest start
    cna_copy['start_distance'] = cna_copy.apply(lambda x: (row['start1'] - x['start_continous']) if (row['start1'] - x['start_continous']) > 0 else float('inf'), axis=1)
    closest_start:pd.DataFrame = cna_copy.loc[cna_copy['start_distance'].idxmin()]
    closest_start.rename({"cna_id": "start1_id"}, inplace=True)
    closest_start['start1_id'] = '-' if closest_start['start1_id'] in cna_hom_del else closest_start['start1_id']
    
    # Find closest end
    cna_copy['end_distance'] = cna_copy.apply(lambda x: (row['start2'] - x['start_continous']) if (row['start2'] - x['start_continous']) > 0 else float('inf'), axis=1)
    closest_end:pd.DataFrame = cna_copy.loc[cna_copy['end_distance'].idxmin()]
    closest_end.rename({"cna_id": "start2_id"}, inplace=True)
    closest_end['start2_id'] = '-' if closest_end['start2_id'] in cna_hom_del else closest_end['start2_id']

    return(closest_start['start1_id'], closest_end['start2_id'])

def assign_inv_alleles(row, cna) -> str:

    """
    Assign major/minor allele tags for inversions
    """

    # Extract CNA information
    row['keep'] = True
    row['allele'] = np.random.choice(['major', 'minor'])
    if row['start1_id'] == row['start2_id']:
        row['cna_id'] = row['start1_id']
        cn:pd.Series = cna.loc[cna['cna_id'] == row['start1_id'], ['major_cn', 'minor_cn']].iloc[0]
    elif row['svclass'] == "h2hINV":
        row['cna_id'] = row['start1_id']
        cn:pd.Series = cna.loc[cna['cna_id'] == row['start1_id'], ['major_cn', 'minor_cn']].iloc[0]
    elif row['svclass'] == "t2tINV":
        row['cna_id'] = row['start2_id']
        cn:pd.Series = cna.loc[cna['cna_id'] == row['start2_id'], ['major_cn', 'minor_cn']].iloc[0]
    else:
        pass

    # Assign alleles
    if cn['major_cn'] == 0:
            row['keep'] = False
    elif cn['minor_cn'] == 0:
        row['allele'] = 'major'
    else:
        pass

    return(row['cna_id'], row['allele'], row['keep'])

def assign_inv(cna, sv, sv_deldup) -> pd.DataFrame:

    """
    Assign simulated inversions to CNA events
    """

    # Assign the inversions based on CNA events
    sv_inv:pd.DataFrame = sv[sv['svclass'].isin(['h2hINV', 't2tINV'])]
    sv_inv = sv_inv.reset_index(drop=True)
    sv_inv = check_inv_overlaps(sv_inv)
    sv_inv[['start1_id', 'start2_id']] = sv_inv.apply(lambda row: find_closest_range(row, cna), axis=1, result_type='expand')
    sv_inv = sv_inv[(sv_inv['start1_id'] != '-') & (sv_inv['start2_id'] != '-')].reset_index(drop=True)
    sv_inv[['cna_id', 'allele', 'keep']] = sv_inv.apply(lambda row: assign_inv_alleles(row, cna), axis=1, result_type='expand')
    sv_inv = sv_inv[sv_inv['keep']].reset_index(drop=True)
    sv_inv = sv_inv.drop(columns=['start1_id', 'start2_id', 'keep'])
    
    # Concatenate del, dup and inv
    sv_deldup_inv:pd.DataFrame = pd.concat([sv_deldup, sv_inv], ignore_index = True)
    sv_deldup_inv = sort_sv(sv_deldup_inv)

    return(sv_deldup_inv)

def assign_tra_alleles_len(row, cna) -> str:

    """
    Assign major/minor allele tags for translocations and define the length
    """

    # Extract CNA information
    cna = cna[cna['chrom'].isin([row['chrom1'], row['chrom2']])]
    ## In 20% of TRA create a more complex event
    if np.random.rand() < 0.2:
        cn_alt_id:str = np.random.choice(['start1_id', 'start2_id'])
        cn_norm_id:str = 'start1_id' if cn_alt_id == 'start2_id' else 'start2_id'

        cna_alt_id_value:str = f'cna{int(row[cn_alt_id].replace("cna", "")) + 1}'
        cna_norm_id_value:str = row[cn_norm_id]
        if cna_alt_id_value in cna['cna_id'] and cna_norm_id_value in cna['cna_id']:
            if cn_alt_id == 'start1_id':
                cn1:pd.Series = cna.loc[cna['cna_id'] == cna_alt_id_value].squeeze()
            else:
                cn2:pd.Series = cna.loc[cna['cna_id'] == cna_alt_id_value].squeeze()

            if cn_norm_id == 'start1_id':
                cn1:pd.Series = cna.loc[cna['cna_id'] == cna_norm_id_value].squeeze()
            else:
                cn2:pd.Series = cna.loc[cna['cna_id'] == cna_norm_id_value].squeeze()
        else:
            # In case the next CNA event is not located in the same chromosome
            cn1:pd.Series = cna.loc[cna['cna_id'] == row['start1_id']].squeeze()
            cn2:pd.Series = cna.loc[cna['cna_id'] == row['start2_id']].squeeze()
    else:
        cn1:pd.Series = cna.loc[cna['cna_id'] == row['start1_id']].squeeze()
        cn2:pd.Series = cna.loc[cna['cna_id'] == row['start2_id']].squeeze()
    
    # Select an allele
    row['allele1'] = np.random.choice(['major', 'minor'])
    row['allele2'] = np.random.choice(['major', 'minor'])
    if cn1['minor_cn'] == 0 and cn2['minor_cn'] == 0 and row['allele1'] == "minor" and row['allele2'] == "minor":
        change_allele:str = np.random.choice(['allele1', 'allele2'])
        row[change_allele] = 'major'

    # Define TRA length
    ## First chrom
    if row['strand1'] == '+':
        row['start1'] = cn1['start']
    else:
        row['end1'] = cn1['end']
    ## Second chrom
    if row['strand2'] == '+':
        row['start2'] = cn2['start']
    else:
        row['end2'] = cn2['end']
    
    # Adapt row shape
    row['cna_id'] = f"{row['start1_id']},{row['start2_id']}"
    row['allele'] = f"{row['allele1']},{row['allele2']}"
    row.drop(labels=['start1_id', 'start2_id', 'allele1', 'allele2'], inplace=True)
    
    return(row)

def assign_tra(cna, sv, sv_deldup_inv) -> pd.DataFrame:

    """
    Assign simulated translocations to CNA events
    """

    # Assign the translocations based on CNA events
    sv_tra:pd.DataFrame = sv[sv['svclass']=='TRA']
    sv_tra = sv_tra.reset_index(drop=True)
    sv_tra[['start1_id', 'start2_id']] = sv_tra.apply(lambda row: find_closest_range(row, cna), axis=1, result_type='expand')
    sv_tra = sv_tra[(sv_tra['start1_id'] != '-') & (sv_tra['start2_id'] != '-')].reset_index(drop=True)
    sv_tra = sv_tra.apply(lambda row: assign_tra_alleles_len(row, cna), axis=1)
    sv_tra = check_tra_overlaps(sv_tra)

    # Concatenate all SV
    sv_deldup_inv_tra:pd.DataFrame = pd.concat([sv_deldup_inv, sv_tra], ignore_index = True)
    sv_deldup_inv_tra = sort_sv(sv_deldup_inv_tra)

    return(sv_deldup_inv_tra)

def align_cna_sv(cna, sv) -> pd.DataFrame:

    """
    Assign SV to CNA events
    """

    # Automatically create DUP/DEL SVs based on CNA events
    sv_assigned:pd.DataFrame = cna2sv_dupdel(cna)

    # Assign simulated INV based on CNA events
    if not sv[sv['svclass'].isin(['h2hINV', 't2tINV'])].empty:
        sv_assigned = assign_inv(cna, sv, sv_assigned)

    # Assign simulated TRA based on CNA events
    if not sv[sv['svclass']=='TRA'].empty:
        sv_assigned = assign_tra(cna, sv, sv_assigned)

    return(sv_assigned)

def simulate_sv(case_cna, nSV, tumor, svModel, gender, idx=0, prefix=None) -> pd.DataFrame:
    
    """
    Generate SVs
    """

    # Simulate the events
    case_sv:pd.DataFrame = pd.DataFrame()
    for sv_class in nSV.index.tolist():
        n:int = nSV[sv_class]
        if n != 0:
            concat_tmp_sv:pd.DataFrame = pd.DataFrame()
            while concat_tmp_sv.shape[0] < n:
                ## Simulate and assign SV positions
                tmp_sv:pd.DataFrame = svModel['sv'].generate_samples(n*5, var_column='svclass', var_class=sv_class)
                tmp_sv['start'] = get_sv_coordinates(n*5, svModel, gender)
                tmp_sv['len'] = round(np.exp(tmp_sv['len'])*10000)
                tmp_sv['end'] = tmp_sv['start'] + tmp_sv['len'] 
                tmp_sv = assign_chromosome(tmp_sv, sv=True, gender=gender)
                tmp_sv = check_sv_strand_patterns(tmp_sv)
                concat_tmp_sv = pd.concat([concat_tmp_sv, tmp_sv], ignore_index=True)
            
            concat_tmp_sv = concat_tmp_sv.sample(n = n, ignore_index=True)
            case_sv = pd.concat([case_sv, concat_tmp_sv])
        else:
            continue
    
    # Sort
    case_sv = sort_sv(case_sv)

    # Assign SV to CNA events
    case_sv = align_cna_sv(case_cna, case_sv)
    
    # Add donor, tumor and sv_id columns
    if prefix == None:
        case_sv["donor_id"] = f"sim{idx}"
    else:
        case_sv["donor_id"] = prefix
    try:
        case_sv = case_sv.drop(columns=['study'])
    except KeyError:
        pass
    case_sv["study"] = tumor
    case_sv['sv_id'] = ['sv{}'.format(i) for i in range(len(case_sv))]
    case_sv = case_sv[['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'strand1', 'strand2', 'svclass', 'cna_id', 'sv_id', 'allele', 'donor_id', 'study']]

    return(case_sv)

def fix_sexual_chrom_cna_sv(case_cna, case_sv) -> tuple:
    
    """
    This function update sexual chromosomes to have only one allele
    """

    # Remove alleles in the CNA dataset
    case_cna['fix_male_cna'] = np.random.choice(['minor', 'major'], size=len(case_cna), replace=True)
    case_cna['fix_male_cna'] = np.where(case_cna['chrom'].isin(['X', 'Y']),
                                        np.where(case_cna['minor_cn'] == 1, 
                                                 'minor',
                                                 case_cna['fix_male_cna']),
                                        '.')
    case_cna['major_cn'] = np.where(case_cna['fix_male_cna'] == 'major', np.nan, case_cna['major_cn'])
    case_cna['minor_cn'] = np.where(case_cna['fix_male_cna'] == 'minor', np.nan, case_cna['minor_cn'])

    # Remove alleles in the SV dataset
    case_sv = case_sv.merge(case_cna[['donor_id', 'study', 'cna_id', 'fix_male_cna']],
                            left_on=['donor_id', 'study', 'cna_id'],
                            right_on=['donor_id', 'study', 'cna_id'],
                            how='left')
    case_sv = case_sv[case_sv['allele'] != case_sv['fix_male_cna']]

    # Drop temporary columns
    case_cna = case_cna.drop(columns=['fix_male_cna'])
    case_sv = case_sv.drop(columns=['fix_male_cna'])

    return(case_cna, case_sv)

def update_vaf(vcf, case_cna, case_sv, gender, nit) -> list:

    """
    Update random generated VAFs to match CNA number
    """

    # Process VCF
    vcf['snv_id'] = [f"snv{i+1}" for i in range(len(vcf))]

    # Process CNA
    case_cna = case_cna.drop(columns=['donor_id', 'study'])
    case_cna = case_cna.rename(columns={'major_cn': 'major', 'minor_cn': 'minor'})
    case_cna = case_cna.melt(id_vars=[col for col in case_cna.columns if col not in ['major', 'minor']],
                            value_vars=['major', 'minor'],
                            var_name='allele',
                            value_name='cn')

    # Process SV
    case_sv = case_sv[['chrom1', 'start1', 'start2', 'svclass', 'cna_id', 'allele', 'sv_id']]
    case_sv = case_sv.rename(columns={'chrom1': 'chrom', 'start1': 'start', 'start2': 'end'})

    # Join CNA and SV
    case_sv_cna:pd.DataFrame = case_sv.merge(case_cna[['cna_id', 'allele', 'cn']], on=['cna_id', 'allele'], how='left')
    case_sv_cna['cn_rep'] = case_sv_cna.apply(lambda row: 1 if row['svclass'] in ['h2hINV', 't2tINV', 'TRA'] else (1 if row['cn'] == 0 else row['cn'] - 1), axis=1)
    case_sv_cna = case_sv_cna.loc[case_sv_cna.index.repeat(case_sv_cna['cn_rep'])].copy()
    case_sv_cna = case_sv_cna.drop(columns=['cn', 'cn_rep'])
    case_sv_cna = case_sv_cna.rename(columns={'allele': 'major_minor'})

    # Match SV and SNV
    sv_range:pd.DataFrame = case_sv_cna.loc[case_sv_cna['svclass'].isin(['DUP', 'DEL']), ['chrom', 'start', 'end', 'cna_id']].drop_duplicates().reset_index(drop=True)
    vcf_range:pd.DataFrame = vcf.copy()
    vcf_range['POS2'] = vcf_range['POS']

    ## Create PyRanges objects
    sv_range:pr.PyRanges = pr.PyRanges(sv_range.rename(columns={'chrom': 'Chromosome', 'start': 'Start', 'end': 'End'}))
    snv_range:pr.PyRanges = pr.PyRanges(vcf_range.rename(columns={'#CHROM': 'Chromosome', 'POS': 'Start', 'POS2': 'End'}))

    ## Perform overlap
    overlapping_snvs:pr.PyRanges = snv_range.join(sv_range)
    if not overlapping_snvs.empty:
        overlapping_snvs:pd.DataFrame = overlapping_snvs.df.copy().drop(columns=['Start_b', 'End_b'])
        overlapping_snvs = overlapping_snvs.rename(columns={'Chromosome': 'chrom', 'Start': 'start', 'ID': 'donor'})
        overlapping_snvs = overlapping_snvs[['chrom', 'start', 'donor', 'snv_id', 'cna_id']]

        ## Get nonverlapping SNVs
        non_overlapping_snvs:pd.DataFrame = vcf[~vcf['snv_id'].isin(overlapping_snvs['snv_id'])].copy()
        non_overlapping_snvs['cna_id'] = "no_cna"
        non_overlapping_snvs = non_overlapping_snvs.rename(columns={'#CHROM': 'chrom', 'POS': 'start', 'ID': 'donor'})
        non_overlapping_snvs = non_overlapping_snvs[['chrom', 'start', 'donor', 'snv_id', 'cna_id']]

        ## Combine
        snv_ann:pd.DataFrame = pd.concat([overlapping_snvs, non_overlapping_snvs], ignore_index=True)
    else:
        snv_ann:pd.DataFrame = vcf.copy()
        snv_ann['cna_id'] = "no_cna"
        snv_ann = snv_ann.rename(columns={'#CHROM': 'chrom', 'POS': 'start', 'ID': 'donor'})
        snv_ann = snv_ann[['chrom', 'start', 'donor', 'snv_id', 'cna_id']]

    # Compute VAFs
    ## Set the order of the events
    sv_events:pd.DataFrame = case_sv_cna[['svclass', 'cna_id', 'major_minor', 'sv_id']].rename(columns={'svclass': 'class', 'sv_id': 'event_id'})
    snv_events:pd.DataFrame = snv_ann[['cna_id', 'snv_id']].rename(columns={'snv_id': 'event_id'}).copy()
    snv_events['class'] = 'MUT'
    snv_events['major_minor'] = pd.NA
    random_order_event:pd.DataFrame = pd.concat([sv_events, snv_events], ignore_index=True)
    random_order_event = random_order_event.sample(frac=1, random_state=42).reset_index(drop=True)
    random_order_event[['allele', 'from_allele', 'to_allele']] = pd.NA

    ## Create a dict of alleles for each cna
    def assign_alleles(x) -> list:
        if str(x).startswith("xX"):
            if gender == "F":
                return ["allele_1_minor", "allele_2_major"]
            else:
                return ["allele_1"]
        elif str(x).startswith("xY"):
            if gender == "F":
                return []
            else:
                return ["allele_1"]
        else:
            return ["allele_1_minor", "allele_2_major"]
    cna_ids:list = ["no_cna"] + case_sv_cna["cna_id"].unique().tolist()
    cna_ids_all:list = [y for x in cna_ids for y in x.split(',')]
    cna_ids_unique:set = set(cna_ids_all)
    allele_ploidy:dict = {cna_id: assign_alleles(cna_id) for cna_id in cna_ids_unique}
    allele_ploidy_original:dict = copy.deepcopy(allele_ploidy)
    allele_ploidy_normal:dict = copy.deepcopy(allele_ploidy)

    ## Create a dict for each cna and allele
    mut_dict:dict = {cna_id: {} for cna_id in cna_ids}

    ## Iterate over the events sequentially
    for i in range(len(random_order_event)):
        f_class:str = random_order_event.loc[i, "class"]
        f_cna_id:str = random_order_event.loc[i, "cna_id"]
        f_event_id:str = random_order_event.loc[i, "event_id"]
        f_major_minor:str = random_order_event.loc[i, "major_minor"]

        # MUTATION CASE
        if f_class == "MUT":
            ### Choose a random allele
            available_alleles:list = allele_ploidy.get(f_cna_id)
            if not available_alleles:
                continue
            allele:str = random.choice(available_alleles)
            random_order_event.loc[i, "allele"] = allele
            
            ### Add the mutation
            mut_df:pd.DataFrame = pd.DataFrame({"id": [f_event_id]})
            if allele not in mut_dict[f_cna_id]:
                mut_dict[f_cna_id][allele] = mut_df
            else:
                mut_dict[f_cna_id][allele] = pd.concat([mut_dict[f_cna_id][allele], mut_df], ignore_index=True)

        # DUPLICATION CASE
        elif f_class == "DUP":
            ### Select the allele to be duplicated
            if len(allele_ploidy_normal[f_cna_id]) == 1:
                available_alleles:list = allele_ploidy.get(f_cna_id)
                if not available_alleles:
                    continue
                allele:str = random.choice(available_alleles)
            else:
                major_minor_alleles:list = [a for a in allele_ploidy[f_cna_id] if re.search(f_major_minor, a)]
                allele:str = random.choice(major_minor_alleles)
            random_order_event.loc[i, "from_allele"] = allele

            ### Update the number of alleles
            new_len:int = len(allele_ploidy_original[f_cna_id]) + 1
            allele_ploidy_original[f_cna_id].append(f"allele_{new_len}_{f_major_minor}")
            allele_ploidy[f_cna_id].append(allele_ploidy_original[f_cna_id][-1])

            ### Select the new allele
            new_allele:str = allele_ploidy[f_cna_id][-1]
            random_order_event.loc[i, "to_allele"] = new_allele

            ### Duplicate the allele
            if allele in mut_dict[f_cna_id]:
                mut_dict[f_cna_id][new_allele] = mut_dict[f_cna_id][allele].copy()

        # DELETION CASE
        elif f_class == "DEL":
            ### Select the allele to be deleted
            if len(allele_ploidy_normal[f_cna_id]) == 1:
                available_alleles:list = allele_ploidy.get(f_cna_id)
                if not available_alleles:
                    continue
                allele:str = random.choice(available_alleles)
            else:
                major_minor_alleles:list = [a for a in allele_ploidy[f_cna_id] if re.search(f_major_minor, a)]
                allele:str = random.choice(major_minor_alleles)
            random_order_event.loc[i, "from_allele"] = allele
        
            ### Update the number of alleles
            if allele in allele_ploidy[f_cna_id]:
                allele_ploidy[f_cna_id].remove(allele)

            ### Remove the mutations in that allele
            mut_dict[f_cna_id].pop(allele, None)
        
        #INV CASE
        elif f_class in ['h2hINV', 't2tINV']:
            ### Choose a random allele
            major_minor_alleles:list = [a for a in allele_ploidy[f_cna_id] if re.search(f_major_minor, a)]
            if not major_minor_alleles:
                random_order_event.loc[i, "allele"] = 'remove'
                continue
            allele:str = random.choice(major_minor_alleles)
            random_order_event.loc[i, "allele"] = allele

        #TRA CASE
        elif f_class == 'TRA':
            ### Choose a random allele for each chrom
            alleles_list:list = []
            for f_cna_id_chrom, f_major_minor_chrom in zip(f_cna_id.split(','), f_major_minor.split(',')):
                major_minor_alleles:list = [a for a in allele_ploidy[f_cna_id_chrom] if re.search(f_major_minor_chrom, a)]
                if not major_minor_alleles:
                    alleles_list.append('remove')
                    continue
                allele:str = random.choice(major_minor_alleles)
                alleles_list.append(allele)
            random_order_event.loc[i, "allele"] = 'remove' if 'remove' in alleles_list else ','.join(alleles_list)
    random_order_event = random_order_event.loc[random_order_event['allele'] != 'remove'].reset_index(drop=True)

    ## Combine event reconstruction
    mut_dict_list:list = []
    for cna_id, allele_key in mut_dict.items():
        total_allele:int = len(allele_key)
        for allele, df in allele_key.items():
            temp_df:pd.DataFrame = df.copy()
            temp_df["cna_id"] = cna_id
            temp_df["allele"] = allele
            temp_df["total_allele"] = total_allele
            mut_dict_list.append(temp_df)
    mut_dict_df:pd.DataFrame = pd.concat(mut_dict_list, ignore_index=True)

    ## Count how many alleles each mutation is found in
    mut_dict_df = mut_dict_df.groupby("id").agg({
        "allele": lambda x: ",".join(sorted(x)),
        "cna_id": "first",
        "total_allele": "first"
    }).reset_index()
    mut_dict_df["n_alleles"] = mut_dict_df["allele"].apply(lambda x: len(x.split(",")))

    ## Simulate base VAFs and adjust
    nit_perc:float = 1-nit
    vaf:np.array = np.random.normal(loc=0.9, scale=0.15, size=len(mut_dict_df))
    vaf:list = list(vaf * (mut_dict_df["n_alleles"] / mut_dict_df["total_allele"]) * nit_perc)
    vaf = [v if v < 1 else 1 - np.random.normal(loc=0.1, scale=0.03) for v in vaf]
    vaf = [round(v, ndigits=2) for v in vaf]
    mut_dict_df["vaf"] = vaf

    # Export the VCF and the order of the events
    ## VCF
    updated_vcf:pd.DataFrame = vcf.merge(
        mut_dict_df.drop(columns=["n_alleles"], errors="ignore"),
        left_on="snv_id",
        right_on="id",
        how="left"
    )
    updated_vcf = updated_vcf[~updated_vcf["vaf"].isna()] #missing alleles
    updated_vcf['ID'] = updated_vcf.apply(lambda row: f"{row['snv_id']}_{row['ID']}", axis=1)
    def update_info(row):
        ms = row['INFO'].split(';')[1]
        cn_number = row['cna_id'].split('_')[-1]
        return f"AF={row['vaf']};{ms};TA={int(row['total_allele'])};AL={row['allele']};CN={cn_number}"
    updated_vcf.loc[:, 'INFO'] = updated_vcf.apply(update_info, axis=1)
    updated_vcf = updated_vcf.drop(columns=['snv_id', 'id', 'allele', 'cna_id', 'vaf', 'total_allele']).reset_index(drop=True)

    ## Events
    random_order_event = random_order_event.fillna('.').reset_index()

    return(updated_vcf, random_order_event)

def update_sv(case_sv, event_order) -> pd.DataFrame:
    
    """
    Update the SVs to remove those that were in a deleted allele
    """

    sv_id_list:list = [event for event in event_order['event_id'].to_list() if event.startswith('sv')]
    case_sv = case_sv.loc[case_sv['sv_id'].isin(sv_id_list)]

    return(case_sv)

@click.group()
def cli():
    pass

@click.command(name="availTumors")
def availTumors():

    """
    List of available tumors to simulate
    """
    
    default_tumors:str = '\n'.join(['\t\t'.join(x) for x in default_tumors])
    click.echo(f"\nThis is the list of available tumor types that can be simulated using oncoGAN:\n\n{default_tumors}\n")

@click.command(name="vcfGANerator")
@click.option("-@", "--cpus",
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of CPUs to use")
@click.option("--tumor",
              type=click.Choice(default_tumors),
              metavar="TEXT",
              show_choices=False,
              required = True,
              help="Tumor type to be simulated. Run 'availTumors' subcommand to check the list of available tumors that can be simulated")
@click.option("-n", "--nCases", "nCases",
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of cases to simulate")
@click.option("--NinT", "nit",
              type=click.FLOAT,
              default=0.0,
              show_default=True,
              help="Normal in Tumor contamination to be taken into account when adjusting VAF for CNA-SV events (e.g. 0.20 = 20%)")
@click.option("-r", "--refGenome", "refGenome",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="hg19 reference genome in fasta format")
@click.option("--prefix",
              type=click.STRING,
              help="Prefix to name the output. If not, '--tumor' option is used as prefix")
@click.option("--outDir", "outDir",
              type=click.Path(exists=False, file_okay=False),
              default=os.getcwd(),
              show_default=False,
              help="Directory where save the simulations. Default is the current directory")
@click.option("--hg38", "hg38",
              is_flag=True,
              required=False,
              help="Transform the mutations to hg38")
@click.option("--mut/--no-mut", "simulateMuts",
              is_flag=True,
              required=False,
              default=True,
              show_default=True,
              help="Simulate mutations")
@click.option("--CNA-SV/--no-CNA-SV", "simulateCNA_SV",
              is_flag=True,
              required=False,
              default=True,
              show_default=True,
              help="Simulate CNA and SV events")
@click.option("--plots/--no-plots", "savePlots",
              is_flag=True,
              required=False,
              default=True,
              show_default=True,
              help="Save plots")
@click.version_option(version=VERSION,
                      package_name="OncoGAN",
                      prog_name="OncoGAN")
def oncoGAN(cpus, tumor, nCases, nit, refGenome, prefix, outDir, hg38, simulateMuts, simulateCNA_SV, savePlots):

    """
    Command to simulate mutations (VCF), CNAs and SVs for different tumor types using a GAN model
    """
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # Torch options
    device:str = torch.device("cpu")

    # Load models
    # cna_sv_countModel, cnaModel, svModel = cna_sv_models(device)

    #TODO - Add an option here to load a template to simulate specific cases
    
    # Simulate counts for each type of mutation
    counts:pd.DataFrame = simulate_counts(tumor, nCases)
    counts_tumor_tag:tuple = tuple(counts.pop('Tumor').to_list())
    counts_total:pd.Series = counts.sum(axis=1)
    
    # Simulate sex
    sex:tuple = simulate_sex(counts_tumor_tag)
    
    if simulateMuts:
        # Simulate mutational signatures (SBS and ID)
        signatures:dict = simulate_signatures(counts)
        
        # Simulate genomic pattern profiles
        genomic_patterns:pd.DataFrame = simulate_genomic_profile(counts_tumor_tag, counts_total)

        # Simulate driver profiles #TODO - Add a database to assign N number of driver mutations per driver event
        driver_profiles:pd.DataFrame = simulate_driver_profile(counts_tumor_tag)
        driver_mutations:dict = select_driver_mutations(counts_tumor_tag, driver_profiles)
        
        # Simulate donor and mutations VAFs
        donor_vaf_ranks:tuple = simulate_vaf_rank(counts_tumor_tag)
        counts_drivers_total:pd.Series = counts_total + driver_profiles.sum(axis=1)
        mut_vafs:dict = simulate_mut_vafs(counts_tumor_tag, donor_vaf_ranks, counts_drivers_total)
    
    # if simulateCNA_SV:
        # Simulate CNA and SV counts
        # cna_sv_counts:pd.DataFrame = select_cna_sv_counts(cna_sv_countModel, nCases, tumor, counts)

    # Simulate one donor at a time
    for idx, case_tumor in tqdm(enumerate(counts_tumor_tag), desc = "Donors"):
        output:str = out_path(outDir, tumor=case_tumor, prefix=prefix, n=idx+1)
        
        case_counts:pd.Series = counts.iloc[idx]
        case_counts_total:int = counts_total.iloc[idx]
        case_counts_drivers_total:int = counts_drivers_total.iloc[idx]
        case_sex:str = sex[idx]
        
        if simulateMuts:
            case_signatures:pd.DataFrame = signatures[idx]
            case_genomic_pattern:pd.Series = genomic_patterns.iloc[idx]
            case_driver_mutations:pd.Series = driver_mutations[idx]
            case_mut_vafs:tuple = mut_vafs[idx] * (1-nit)

            # Generate the chromosome and position of the mutations
            case_genomic_positions = assign_genomic_positions(case_sex, case_signatures, case_genomic_pattern, refGenome, cpus)

            # Create the VCF output 
            vcf:pd.DataFrame = pd2vcf(case_genomic_positions, case_driver_mutations, case_mut_vafs, idx=idx)

            # Write the VCF
            if not simulateCNA_SV:
                ## Convert from hg19 to hg38
                if hg38:
                    vcf = hg19tohg38(vcf=vcf)
                with open(output, "w+") as out:
                    out.write("##fileformat=VCFv4.2\n")
                    out.write(f"##fileDate={date.today().strftime('%Y%m%d')}\n")
                    out.write(f"##source=OncoGAN-v{VERSION}\n")
                    out.write(f"##reference={'hg38' if hg38 else 'hg19'}\n")
                    out.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">\n')
                    out.write('##INFO=<ID=MS,Number=A,Type=String,Description="Mutation type or mutational signature assigned to each mutation. Available options are: SBS (single base substitution signature), DNP (dinucleotide polymorphism), TNP (trinucleotide polymorphism), ID (indel signature), driver_* (driver mutation sampled from real donors)">\n')
                    out.write('##INFO=<ID=SBSCTX,Number=A,Type=String,Description="SBS96 context">\n')
                    out.write('##INFO=<ID=IDCTX,Number=A,Type=String,Description="Indel context">\n')
                    out.write('##INFO=<ID=HPR,Number=A,Type=String,Description="Homopolymer reference">\n')
                    out.write('##INFO=<ID=MHR,Number=A,Type=String,Description="Microhomology reference">\n')
                vcf.to_csv(output, sep="\t", index=False, mode="a")

        # if simulateCNA_SV:
        #    case_cna_sv:pd.Series = cna_sv_counts.iloc[idx]
        #     # Simulate CNAs
        #     case_cna:pd.DataFrame = simulate_cnas(case_cna_sv['cna'], case_cna_sv['len'], tumor, cnaModel, gender, refGenome, idx=idx+1)

        #     # Simulate SVs
        #     case_sv:pd.DataFrame = simulate_sv(case_cna, case_cna_sv.loc['DEL':'t2tINV'], tumor, svModel, gender, idx=idx+1)

        #     # Sexual chrom must have only one allele when sex is male
        #     if gender == "M":
        #         case_cna, case_sv = fix_sexual_chrom_cna_sv(case_cna, case_sv)

        #     # Update mutation VAFs according to CNAs
        #     if simulateMuts:
        #         vcf, events_order = update_vaf(vcf, case_cna, case_sv, gender, nit)
        #         case_sv = update_sv(case_sv, events_order)

        #         ## Convert from hg19 to hg38
        #         if hg38:
        #             vcf = hg19tohg38(vcf=vcf)
        #         with open(output, "w+") as out:
        #             out.write("##fileformat=VCFv4.2\n")
        #             out.write(f"##fileDate={date.today().strftime('%Y%m%d')}\n")
        #             out.write(f"##source=OncoGAN-v{VERSION}\n")
        #             out.write(f"##reference={'hg38' if hg38 else 'hg19'}\n")
        #             out.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">\n')
        #             out.write('##INFO=<ID=MS,Number=A,Type=String,Description="Mutation type or mutational signature assigned to each mutation. Available options are: SBS (single base substitution signature), DNP (dinucleotide polymorphism), TNP (trinucleotide polymorphism), DEL (deletion), INS (insertion), driver* (driver mutation sampled from real donors)">\n')
        #             out.write('##INFO=<ID=TA,Number=A,Type=Integer,Description="Total number of alleles in which the mutation can appear">\n')
        #             out.write('##INFO=<ID=AL,Number=A,Type=String,Description="Alleles in which the mutation appears">\n')
        #             out.write('##INFO=<ID=CN,Number=A,Type=String,Description="Copy Number ID in which the mutation is located">\n')
        #         vcf.to_csv(output, sep="\t", index=False, mode="a")
        #         events_order.to_csv(output.replace(".vcf", "_events_order.tsv"), sep="\t", index=False, mode="w")

        #     # Plots
        #     if savePlots:
        #         plot_cnas(case_cna, case_sv, tumor, output.replace(".vcf", "_cna.png"), idx=idx+1) 
            
        #     # Convert from hg19 to hg38
        #     if hg38:
        #         case_cna = hg19tohg38(cna=case_cna)
        #         case_sv = hg19tohg38(sv=case_sv)

        #     # Save simulations
        #     case_cna['major_cn'] = case_cna['major_cn'].astype('Int64')
        #     case_cna['minor_cn'] = case_cna['minor_cn'].astype('Int64')
        #     case_cna.to_csv(output.replace(".vcf", "_cna.tsv"), sep ='\t', index=False)
        #     case_sv.to_csv(output.replace(".vcf", "_sv.tsv"), sep ='\t', index=False)
