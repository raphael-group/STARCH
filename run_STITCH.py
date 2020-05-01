from STITCH import STITCH
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
 
parser.add_argument('-method','--method',required=True,type=str,help="method (hmm or hmrf)") 
parser.add_argument('-t','--t',required=True,type=float,help="gene") 
parser.add_argument('-beta_spot','--beta_spot',required=True,type=float,help="spot")  
parser.add_argument('-n_clusters','--n_clusters',required=True,type=int,help="spot")   
parser.add_argument('-n','--name',required=True,type=str,help="name of input file (either expression matrix in .csv, .tsv, .txt format or 10X directory containing barcodes.tsv, features.tsv, matrix.mtx, and tissue_positions_list.csv)")   
parser.add_argument('-normal_spots','--normal_spots',required=False,type=str,help="name of input file containing indices of normal spots",default=0)
parser.add_argument('-returnnormal','--returnnormal',required=False,type=int,default=1)   
parser.add_argument('-o','--output',required=False,type=str,default='output') 
args = parser.parse_args()

method = args.method
t = args.t
beta_spot = args.beta_spot
n_clusters = args.n_clusters
returnnormal = args.returnnormal
name = args.name
normal_spots = args.normal_spots
out = args.output

if normal_spots !=0:
	normal_spots = np.asarray(pd.read_csv(normal_spots,header=None)).flatten()
else:
	normal_spots = []

operator = STITCH(name,n_clusters=n_clusters,num_states=3,normal_spots=normal_spots,beta_spots = beta_spot,nthreads=20,gene_mapping_file_name='hgTables.txt')
posteriors = operator.callCNA(t=t,beta_spots=beta_spot,nthreads=20,maxiters=20,returnnormal=returnnormal)

print('Posterior Energy: ',posteriors)

pd.DataFrame(operator.states).to_csv('/n/fs/ragr-research/projects/spatial_transcriptomics/states_%s_%s.csv'%(out,method),sep=',')
pd.DataFrame(operator.labels).to_csv('/n/fs/ragr-research/projects/spatial_transcriptomics/labels_%s_%s.csv'%(out,method),sep=',')
