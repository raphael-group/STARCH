#!/bin/python

import sys
import numpy as np
import pandas as pd


def main(sample_id, outdir):
    df_snp = pd.read_csv(f"{outdir}/pileup/{sample_id}/cellSNP.base.vcf", comment="#", sep="\t", names=["tmpCHR", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"])
    df_snp["CHROM"] = [f"chr{x}" for x in df_snp.tmpCHR]
    df_snp["AD"] = [int(x.split(";")[0].split("=")[-1]) for x in df_snp.INFO]
    df_snp["DP"] = [int(x.split(";")[1].split("=")[-1]) for x in df_snp.INFO]
    df_snp["OTH"] = [int(x.split(";")[2].split("=")[-1]) for x in df_snp.INFO]
    # remove records with DP == 0
    df_snp = df_snp[df_snp.DP > 0]
    # keep het SNP (0.1 <= AD/DP <= 0.9) and hom ALT SNP (AD == DP >= 10)
    df_snp = df_snp[((df_snp.AD / df_snp.DP >= 0.1) & (df_snp.AD / df_snp.DP <= 0.9)) | ((df_snp.AD == df_snp.DP) & (df_snp.DP >= 10))]
    # add addition columns
    df_snp["FORMAT"] = "GT"
    df_snp[f"{sample_id}"] = ["0/1" if row.AD < row.DP else "1/1" for i,row in df_snp.iterrows()]
    # output chromosome to folder
    for c in range(1, 23):
        df = df_snp[ (df_snp.tmpCHR == c) | (df_snp.tmpCHR == str(c)) ]
        # remove records that have duplicated snp_id
        snp_id = [f"{row.tmpCHR}_{row.POS}_{row.REF}_{row.ALT}" for i,row in df.iterrows()]
        df["snp_id"] = snp_id
        df = df.groupby("snp_id").agg({"CHROM":"first", "POS":"first", "ID":"first", "REF":"first", "ALT":"first", "QUAL":"first", "FILTER":"first", \
                                       "INFO":"first", "FORMAT":"first", f"{sample_id}":"first", "AD":"sum", "DP":"sum", "OTH":"sum"})
        info = [f"AD={row.AD};DP={row.DP};OTH={row.OTH}" for i,row in df.iterrows()]
        df["INFO"] = info
        df = df[["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", f"{sample_id}"]]
        df.sort_values(by="POS", inplace=True)
        fp = open(f"{outdir}/phasing/{sample_id}_chr{c}.vcf", 'w')
        fp.write("##fileformat=VCFv4.2\n")
        fp.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Consensus Genotype across all datasets with called genotype\">\n")
        fp.write("#" + "\t".join(df.columns) + "\n")
        df.to_csv(fp, sep="\t", index=False, header=False)
        fp.close()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
