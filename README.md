# STITCH
Spatial Transcriptomics Inference of Tumor Copy-number Heterogeneity

## Required Python Libraries
Numpy, Pandas, Scipy, SKlearn, multiprocessing, HMMlearn

## Required input
1. Gene x Spot STRNA-seq expression matrix. This can be a text file (.csv, .tsv, .txt) where the first column is contains the gene names in HUGO format and the first row contains the spot coordinates in the format '1x2'. STITCH can also read in the output from 10Xs Space Ranger pipeline. In this case, the input is the directory containing the following files: barcodes.tsv, features.tsv, matrix.mtx, and tissue_positions_list.csv

2. Gene mapping file which maps each HUGO gene name to chromosomal positions. The mapping file for human assembly hg19 is provided by default (hgTables_hg19.txt). For other organisms, an hgTables file can be downloaded from https://genome.ucsc.edu/cgi-bin/hgTables.

## Running from Command Line

python run_STITCH.py -i gene_expression_matrix.csv (or 10X_directory/) -o name --n_clusters 3 --outdir output/directory/
