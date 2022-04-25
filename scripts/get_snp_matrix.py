#!/bin/python

import sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import scipy.io
from pathlib import Path
import json
import gzip
import pickle
from tqdm import trange


def process_snp_phasing(cellsnp_folder, eagle_folder, outputfile):
    # create a (snp_id, GT) map from eagle2 output
    snp_gt_map = {}
    for c in range(1, 23):
        fname = [str(x) for x in Path(eagle_folder).glob("*chr{}.phased.vcf.gz".format(c))]
        assert len(fname) > 0
        fname = fname[0]
        tmpdf = pd.read_table(fname, compression = 'gzip', comment = '#', sep="\t", names=["CHR","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT","PHASE"])
        this_snp_ids = [ "{}_{}_{}_{}".format(c, row.POS, row.REF, row.ALT) for i,row in tmpdf.iterrows() ]
        this_gt = list(tmpdf.iloc[:,-1])
        assert len(this_snp_ids) == len(this_gt)
        snp_gt_map.update( {this_snp_ids[i]:this_gt[i] for i in range(len(this_gt))} )
    # cellsnp DP (read depth) and AD (alternative allele depth)
    # first get a list of snp_id and spot barcodes
    tmpdf = pd.read_csv(cellsnp_folder + "/cellSNP.base.vcf", header=1, sep="\t")
    snp_list = np.array([ "{}_{}_{}_{}".format(row["#CHROM"], row.POS, row.REF, row.ALT) for i,row in tmpdf.iterrows() ])
    tmpdf = pd.read_csv(cellsnp_folder + "/cellSNP.samples.tsv", header=None)
    sample_list = np.array(list(tmpdf.iloc[:,0]))
    # then get the DP and AD matrix
    DP = scipy.io.mmread(cellsnp_folder + "/cellSNP.tag.DP.mtx").tocsr()
    AD = scipy.io.mmread(cellsnp_folder + "/cellSNP.tag.AD.mtx").tocsr()
    # remove SNPs that are not phased
    is_phased = np.array([ (x in snp_gt_map) for x in snp_list ])
    DP = DP[is_phased,:]
    AD = AD[is_phased,:]
    snp_list = snp_list[is_phased]
    # generate a new dataframe with columns (cell, snp_id, DP, AD, CHROM, POS, GT)
    rows, cols = DP.nonzero()
    cell = sample_list[cols]
    snp_id = snp_list[rows]
    DP_df = DP[DP.nonzero()].A.flatten()
    AD_df = AD[DP.nonzero()].A.flatten()
    GT = [snp_gt_map[x] for x in snp_id]
    df = pd.DataFrame({"cell":cell, "snp_id":snp_id, "DP":DP_df, "AD":AD_df, \
                       "CHROM":[int(x.split("_")[0]) for x in snp_id], "POS":[int(x.split("_")[1]) for x in snp_id], "GT":GT})
    df.to_csv(outputfile, sep="\t", index=False, header=True, compression={'method': 'gzip'})
    return df


def read_cell_by_snp(allele_counts_file):
    df = pd.read_csv(allele_counts_file, sep="\t", header=0)
    index = np.array([i for i,x in enumerate(df.GT) if x=="0|1" or x=="1|0"])
    df = df.iloc[index, :]
    df.CHROM = df.CHROM.astype(int)
    return df


def cell_by_gene_lefthap_counts(df_cell_snp, hg_table_file, gene_list, barcode_list):
    # index of genes and barcodes in the current gene expression matrix
    barcode_mapper = {x:i for i,x in enumerate(barcode_list)}
    gene_mapper = {x:i for i,x in enumerate(gene_list)}
    # read gene ranges in genome
    # NOTE THAT THE FOLLOWING CODE REQUIRES hg_table_file IS SORTED BY GENOMIC POSITION!
    df_genes = pd.read_csv(hg_table_file, header=0, index_col=0, sep="\t")
    index = np.array([ i for i in range(df_genes.shape[0]) if (not "_" in df_genes.chrom.iloc[i]) and \
                      (df_genes.chrom.iloc[i] != "chrX") and (df_genes.chrom.iloc[i] != "chrY") and (df_genes.chrom.iloc[i] != "chrM") and \
                      (not "GL" in df_genes.chrom.iloc[i]) and (not "KI" in df_genes.chrom.iloc[i]) ])
    df_genes = df_genes.iloc[index, :]
    tmp_gene_ranges = {df_genes.name2.iloc[i]:(int(df_genes.chrom.iloc[i][3:]), df_genes.cdsStart.iloc[i], df_genes.cdsEnd.iloc[i]) for i in np.arange(df_genes.shape[0]) }
    gene_ranges = [(gname, tmp_gene_ranges[gname]) for gname in gene_list if gname in tmp_gene_ranges]
    del tmp_gene_ranges
    # aggregate snp counts to genes
    N = np.unique(df_cell_snp.cell).shape[0]
    G = len(gene_list)
    i = 0
    j = 0
    cell_gene_DP = np.zeros((N, G), dtype=int)
    cell_gene_AP = np.zeros((N, G), dtype=int)
    cell_gene_snpcount = np.zeros((N, G), dtype=int)
    gene_snp_map = {gname:set() for gname in gene_list}
    for i in trange(df_cell_snp.shape[0]):
        # check cell barcode
        if not df_cell_snp.cell.iloc[i] in barcode_mapper:
            continue
        cell_idx = barcode_mapper[df_cell_snp.cell.iloc[i]]
        # if the SNP is not within any genes
        if j < len(gene_ranges) and (df_cell_snp.CHROM.iloc[i] < gene_ranges[j][1][0] or \
                                     (df_cell_snp.CHROM.iloc[i] == gene_ranges[j][1][0] and df_cell_snp.POS.iloc[i] < gene_ranges[j][1][1])):
            continue
        # if the SNP position passes gene j
        while j < len(gene_ranges) and (df_cell_snp.CHROM.iloc[i] > gene_ranges[j][1][0] or \
                                        (df_cell_snp.CHROM.iloc[i] == gene_ranges[j][1][0] and df_cell_snp.POS.iloc[i] > gene_ranges[j][1][2])):
            j += 1
        if j < len(gene_ranges) and df_cell_snp.CHROM.iloc[i] == gene_ranges[j][1][0] and \
        df_cell_snp.POS.iloc[i] >= gene_ranges[j][1][1] and df_cell_snp.POS.iloc[i] <= gene_ranges[j][1][2]:
            gene_idx = gene_mapper[gene_ranges[j][0]]
            cell_gene_DP[cell_idx, gene_idx] += df_cell_snp.DP.iloc[i]
            cell_gene_snpcount[cell_idx, gene_idx] += 1
            gene_snp_map[ gene_ranges[j][0] ].add( df_cell_snp.snp_id.iloc[i] )
            if df_cell_snp.GT.iloc[i] == "0|1":
                cell_gene_AP[cell_idx, gene_idx] += (df_cell_snp.DP.iloc[i] - df_cell_snp.AD.iloc[i])
            else:
                cell_gene_AP[cell_idx, gene_idx] += df_cell_snp.AD.iloc[i]
    return cell_gene_DP, cell_gene_AP, cell_gene_snpcount, gene_snp_map


if __name__ == "__main__":
    numbatdir = sys.argv[1]
    sample_id = sys.argv[2]
    hg_table_file = sys.argv[3]
    cellranger_filtered_dir = sys.argv[4]
    
    outputfile = f"{numbatdir}/{sample_id}_myallele_counts.tsv.gz"
    _ = process_snp_phasing(f"{numbatdir}/pileup/{sample_id}/", f"{numbatdir}/phasing/", outputfile)
    
    df_cell_snp = read_cell_by_snp(f"{numbatdir}/{sample_id}_myallele_counts.tsv.gz")
    print("sum DP = {}".format(np.sum(df_cell_snp.DP)))

    barcode_list = list(pd.read_csv(f"{cellranger_filtered_dir}/barcodes.tsv.gz", header=None).iloc[:,0])
    gene_list = list(pd.read_csv(f"{cellranger_filtered_dir}/features.tsv.gz", header=None, sep="\t").iloc[:,1])
    if not Path(f"{dir}/numbatprep/cell_gene_DP.npy").exists():
        cell_gene_DP, cell_gene_AD, cell_gene_snpcount, gene_snp_map = cell_by_gene_lefthap_counts(df_cell_snp, hg_table_file, gene_list, barcode_list)
        np.save(f"{numbatdir}/cell_gene_DP.npy", cell_gene_DP)
        np.save(f"{numbatdir}/cell_gene_AD.npy", cell_gene_AD)
        np.save(f"{numbatdir}/cell_gene_snpcount.npy", cell_gene_snpcount)
        pickle.dump(gene_snp_map, open(f"{numbatdir}/gene_snp_map.pkl", 'wb'))
