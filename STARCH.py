import numpy as np
import pandas as pd
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
import math
import copy
import sklearn
import sklearn.cluster
import random
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score,v_measure_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
import multiprocessing as mp
from functools import partial
from scipy.spatial import distance
import os
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from hmmlearn import hmm
from scipy.io import mmread
from scipy.sparse import csr_matrix
import multiprocessing
import warnings
from pathlib import Path
os.environ['NUMEXPR_MAX_THREADS'] = '50'


def jointLikelihoodEnergyLabels_helper(label,data,states,norms):
	e = 1e-50
	r0 = [x for x in range(data.shape[0]) if states[x,label]==0]
	l0 = np.sum(-np.log(np.asarray(norms[0].pdf(data[r0,:])+e)),axis=0) 
	r1 = [x for x in range(data.shape[0]) if states[x,label]==1]
	l1 = np.sum(-np.log(np.asarray(norms[1].pdf(data[r1,:])+e)),axis=0) 
	r2 = [x for x in range(data.shape[0]) if states[x,label]==2]
	l2 = np.sum(-np.log(np.asarray(norms[2].pdf(data[r2,:])+e)),axis=0) 
	return l0 + l1 + l2

def init_helper(i,data, n_clusters,normal,diff,labels,c):
	l = []
	for k in range(n_clusters):
		pval = ks_2samp(data[i,labels==k],normal[i,:])[1]
		mn = np.mean(normal[i,:])
		if c[i,k]< mn and pval <= diff:
			l.append(0)
		elif c[i,k]> mn and pval <= diff:
			l.append(2)
		else:
			l.append(1)
	return np.asarray(l).astype(int)

def HMM_helper(inds, data, means, sigmas ,t, num_states, model,normal):
	ind_bin,ind_spot,k = inds
	data = data[np.asarray(ind_bin)[:, None],np.asarray(ind_spot)]
	data2 = np.mean(data,axis=1)
	X = np.asarray([[x] for x in data2])
	C = np.asarray(model.predict(X))
	score = model.score(X)
	#bootstrap
	b=3
	for i in range(b):
		inds = random.sample(range(data.shape[1]),int(data.shape[1]*.8+1))
		data2 = np.mean(data[:,inds],axis=1)
		X = np.asarray([[x] for x in data2])
		C2 = np.asarray(model.predict(X))
		for j,c in enumerate(C2):
			if C[j] != c:
				C[j] = 1
	return [C,score]

class STARCH:
	"""
	This is a class for Hidden Markov Random Field for calling Copy Number Aberrations
	using spatial relationships and gene adjacencies along chromosomes
	"""

	def __init__(self,data,normal_spots=[],labels=[],beta_spots=2,n_clusters=3,num_states=3,gene_mapping_file_name='hgTables_hg19.txt',nthreads=0,platform="ST"):
		"""
		The constructor for HMFR_CNA

		Parameters:
			data (pandas data frame): gene x spot (or cell). 
				colnames = 2d or 3d indices (eg. 5x18, 5x18x2 if multiple layers). 
				rownames = HUGO gene name
		"""
		assert( platform == "ST" or platform == "Visium" )
		self.platform = platform
		logger.info("platform is {}".format(self.platform))
		if nthreads == 0:
			nthreads = int(multiprocessing.cpu_count() / 2 + 1)
			logger.info('Running with ' + str(nthreads) + ' threads')
		logger.info("initializing HMRF...")
		self.beta_spots = beta_spots
		self.gene_mapping_file_name = gene_mapping_file_name
		self.n_clusters = int(n_clusters)
		dat,data = self.preload(data)
		logger.info(str(self.rows[0:20]))
		logger.info(str(len(self.rows)) + ' ' + str(len(self.columns)) + ' ' + str(data.shape))
		if isinstance(normal_spots, str):
			self.read_normal_spots(normal_spots)
		if normal_spots == []:
			self.get_normal_spots(data)
		else:
			self.normal_spots = np.asarray([int(x) for x in normal_spots])
		logger.info('normal spots ' + str(len(self.normal_spots)))
		dat = self.preprocess_data(data,dat)
		logger.info('done preprocessing...')
		self.data = self.data * 1000
		self.bins = self.data.shape[0]
		self.spots = self.data.shape[1]
		self.tumor_spots = np.asarray([int(x) for x in range(self.spots) if int(x) not in self.normal_spots])
		self.normal = self.data[:,self.normal_spots]
		self.data = self.data[:,self.tumor_spots]
		self.bins = self.data.shape[0]
		self.spots = self.data.shape[1]
		self.num_states = int(num_states)
		self.normal_state = int((self.num_states-1)/2)

		logger.info('getting spot network...')
		self.get_spot_network(self.data,self.columns[self.tumor_spots])
		if isinstance(labels, str):
			self.get_labels(labels)
		if len(labels)>0:
			self.labels = labels
		else:
			logger.info('initializing labels...')
			self.initialize_labels()
		logger.debug('starting labels: '+str(self.labels))
		np.fill_diagonal(self.spot_network, 0)
		logger.info('getting params...')
		count_valueerror = 0
		for d in range(10 ,20,1):
			try:
				self.init_params(d/10,nthreads)
				break
			except ValueError:
				count_valueerror += 1
				continue
		logger.info('Count of ValueError in init_params is {}'.format(count_valueerror))
		self.states = np.zeros((self.bins,self.n_clusters))
		logger.info('starting means: '+str(self.means))
		logger.info('starting cov: '+str(self.sigmas))
		logger.info(str(len(self.rows)) + ' ' + str(len(self.columns)) + ' ' + str(self.data.shape))


	def to_transpose(self,sep,data):
		dat = pd.read_csv(data,sep=sep,header=0,index_col=0)
		if 'x' in dat.index.values[0] and 'x' in dat.index.values[1] and 'x' in dat.index.values[2]:
			return True
		return False

	def which_sep(self,data):
		dat = np.asarray(pd.read_csv(data,sep='\t',header=0,index_col=0)).size
		dat2 = np.asarray(pd.read_csv(data,sep=',',header=0,index_col=0)).size
		dat3 = np.asarray(pd.read_csv(data,sep=' ',header=0,index_col=0)).size
		if dat > dat2 and dat > dat3:
			return '\t'
		elif dat2 > dat and dat2 > dat3:
			return ','
		else:
			return ' '

	def get_bin_size(self,data,chroms):
		for bin_size in range(20,100):
			test =  self.bin_data2(data[:,self.normal_spots],chroms,bin_size=bin_size,step_size=1)
			test = test[test!=0] 
			logger.debug(str(bin_size)+' mean expression binned ' + str(np.mean(test)))
			logger.debug(str(bin_size)+' median expression binned ' + str(np.median(test)))
			if np.median(test) >= 10:
				break
		logger.info('selected bin size: ' + str(bin_size))
		return bin_size

	def preload(self,l):
		if isinstance(l,list): # list of multiple datasets
			offset = 0
			dats = []
			datas = []
			for data in l:
				dat,data = self.load(data)
				datas.append(data)
				dats.append(dat)
			conserved_genes = []
			inds = []
			for dat in dats:
				inds.append([])
			for gene in dats[0].index.values:
				inall = True
				for dat in dats:
					if gene not in dat.index.values:
						inall = False
				if inall:
					conserved_genes.append(gene)
					for i,dat in enumerate(dats):
						ind = inds[i]
						ind.append(np.where(dat.index.values == gene)[0][0])
						inds[i] = ind
			conserved_genes = np.asarray(conserved_genes)
			logger.info(str(conserved_genes))
			newdatas = []
			newdats = []
			for i in range(len(datas)):
				data = datas[i]
				dat = dats[i]
				ind = np.asarray(inds[i])
				newdatas.append(data[ind,:])
				newdats.append(dat.iloc[ind,:])
			for dat in newdats:
				spots = np.asarray([[float(y) for y in x.split('x')] for x in dat.columns.values])
				for spot in spots:
					spot[0] += offset
				spots = ['x'.join([str(y) for y in x]) for x in spots]
				dat.columns = spots
				offset += 100
			data = np.concatenate(newdatas,axis=1)
			dat = pd.concat(newdats,axis=1)
			self.rows = dat.index.values
			self.columns = dat.columns.values
		else:
			dat,data = self.load(l)
		return dat,data

	def load(self,data):
		try:
			if isinstance(data, str) and ('.csv' in data or '.tsv' in data or '.txt' in data):
				logger.info('Reading data...')
				sep = self.which_sep(data)
				if self.to_transpose(sep,data):
					dat = pd.read_csv(data,sep=sep,header=0,index_col=0).T
				else:
					dat = pd.read_csv(data,sep=sep,header=0,index_col=0)
			elif isinstance(data,str):
				logger.info('Importing 10X data from directory. Directory must contain barcodes.tsv, features.tsv, matrix.mtx, tissue_positions_list.csv')
				# find the barcodes file from 10X directory
				file_barcodes = [str(x) for x in Path(data).rglob("*barcodes.tsv*")]
				if len(file_barcodes) == 0:
					logger.error('There is no barcode.tsv file in the 10X directory.')
				file_barcodes = file_barcodes[0]
				barcodes = np.asarray(pd.read_csv(file_barcodes,header=None)).flatten()
				# find the features file from 10X directory
				file_features = [str(x) for x in Path(data).rglob("*features.tsv*")]
				if len(file_features) == 0:
					logger.error('There is no features.tsv file in the 10X directory.')
				file_features = file_features[0]
				genes = np.asarray(pd.read_csv(file_features,sep='\t',header=None))
				genes = genes[:,1]
				# find the tissue_position_list file from 10X directory
				file_coords = [str(x) for x in Path(data).rglob("*tissue_positions_list.csv*")]
				if len(file_coords) == 0:
					logger.error('There is no tissue_positions_list.csv file in the 10X directory.')
				file_coords = file_coords[0]
				coords = np.asarray(pd.read_csv(file_coords,sep=',',header=None))
				d = dict()
				for row in coords:
					d[row[0]] = str(row[2]) + 'x' + str(row[3])
				inds = []
				coords2 = []
				for i,barcode in enumerate(barcodes):
					if barcode in d.keys():
						inds.append(i)
						coords2.append(d[barcode])
				# find the count matrix file
				file_matrix = [str(x) for x in Path(data).rglob("*matrix.mtx*")]
				if len(file_matrix) == 0:
					logger.error('There is no matrix.mtx file in the 10X directory.')
				file_matrix = file_matrix[0]
				matrix = mmread(file_matrix).toarray()
				logger.info(str(barcodes) + ' ' + str(barcodes.shape))
				logger.info(str(genes) + ' ' + str(genes.shape))
				logger.info(str(coords) + ' ' + str(coords.shape))
				logger.info(str(matrix.shape))

				matrix = matrix[:,inds]
				genes,inds2 = np.unique(genes, return_index=True)
				matrix = matrix[inds2,:]
				dat = pd.DataFrame(matrix,index = genes,columns = coords2)
				
				logger.info(str(dat))
			else:
				dat = pd.DataFrame(data)
		except:
			raise Exception("Incorrect input format")
		logger.info('coords ' + str(len(dat.columns.values)))
		logger.info('genes ' + str(len(dat.index.values)))
		data = dat.values
		logger.info(str(data.shape))
		self.rows = dat.index.values
		self.columns = dat.columns.values
		return(dat,data)



	def preprocess_data(self,data,dat):
		logger.info('data shape ' + str(data.shape))
		data,inds = self.filter_genes(data,min_cells=int(data.shape[1]/20))
		logger.info('Filtered genes, now have ' + str(data.shape[0]) + ' genes')
		data[data>np.mean(data)+np.std(data)*2]=np.mean(data)+np.std(data)*2
		dat = dat.T[dat.index.values[inds]].T
		self.rows = dat.index.values
		self.columns = dat.columns.values
		logger.info('filter ' + str(len(self.rows)) + ' ' + str(len(self.columns)) + ' ' + str(data.shape))
		data,chroms,pos,inds = self.order_genes_by_position(data,dat.index.values)
		dat = dat.T[dat.index.values[inds]].T
		self.rows = dat.index.values
		self.columns = dat.columns.values
		logger.info('order ' + str(len(self.rows)) + ' ' + str(len(self.columns)) + ' ' + str(data.shape))
		logger.info('zero percentage ' + str((data.size - np.count_nonzero(data)) / data.size))

		bin_size = self.get_bin_size(data,chroms)

		data = np.log(data+1) 
		data = self.library_size_normalize(data) #2
		data = data-np.mean(data[:,self.normal_spots],axis=1).reshape(data.shape[0],1)
		data = self.threshold_data(data,max_value=3.0)
		data =  self.bin_data(data,chroms,bin_size=bin_size,step_size=1) 
		data = self.center_at_zero(data) #7
		data = data-np.mean(data[:,self.normal_spots],axis=1).reshape(data.shape[0],1)
		data = np.exp(data)-1
		self.data = data
		self.pos = np.asarray([str(x) for x in pos])
		logger.info('preprocess ' + str(len(self.rows)) + ' ' + str(len(self.columns)) + ' ' + str(data.shape))
		return(dat)

	def read_normal_spots(self,normal_spots):
		normal_spots = pd.read_csv(data,sep=',')
		self.normal_spots = np.asarray([int(x) or x in np.asarray(normal_spots)])

	def get_normal_spots(self,data):
		data,k = self.filter_genes(data,min_cells=int(data.shape[1]/20)) # 1
		data = self.library_size_normalize(data) #2
		data = np.log(data+1) 
		data = self.threshold_data(data,max_value=3.0)
		pca = PCA(n_components=1).fit_transform(data.T)
		km = KMeans(n_clusters=2).fit(pca)
		clusters = np.asarray(km.predict(pca))
		if np.mean(data[:,clusters==0]) < np.mean(data[:,clusters==1]):
			self.normal_spots = np.asarray([x for x in range(data.shape[1])])[clusters==0]
		else:
			self.normal_spots = np.asarray([x for x in range(data.shape[1])])[clusters==1]

	def filter_genes(self,data,min_cells=20):
		keep = []
		for gene in range(data.shape[0]):
			if np.count_nonzero(data[gene,:]) >= min_cells:
				keep.append(gene)
		return data[np.asarray(keep),:],np.asarray(keep)

	def library_size_normalize(self,data):
		m = np.median(np.sum(data,axis=0))
		data = data / np.sum(data,axis=0)
		data = data * m
		return data

	def threshold_data(self,data,max_value=4.0):
		data[data> max_value] = max_value
		data[data< -max_value] = -max_value
		return data

	def center_at_zero(self,data):
		return data - np.median(data,axis=0).reshape(1,data.shape[1])


	def bin_data2(self,data,chroms,bin_size,step_size):
		newdata = copy.deepcopy(data)
		i=0
		c = np.asarray(list(set(chroms)))
		c.sort()
		for chrom in c:
			data2 = data[chroms==chrom,:]
			for gene in range(data2.shape[0]):
				start = max(0,gene-int(bin_size/2))
				end = min(data2.shape[0],gene+int(bin_size/2))
				r = np.asarray([x for x in range(start,end)])
				mean = np.sum(data2[r,:],axis=0)
				newdata[i,:] = mean
				i += 1
		return newdata


	def bin_data(self,data,chroms,bin_size,step_size):
		newdata = copy.deepcopy(data)
		i=0
		c = np.asarray(list(set(chroms)))
		c.sort()
		for chrom in c:
			data2 = data[chroms==chrom,:]
			for gene in range(data2.shape[0]):
				start = max(0,gene-int(bin_size/2))
				end = min(data2.shape[0],gene+int(bin_size/2))
				r = np.asarray([x for x in range(start,end)])
				weighting = np.asarray([x+1 for x in range(start,end)])
				weighting = abs(weighting - len(weighting)/2)
				weighting = 1/(weighting+1)
				weighting = weighting / sum(weighting) #pyramidinal weighting
				weighting = weighting.reshape(len(r),1)
				mean = np.sum(data2[r,:]*weighting,axis=0)
				newdata[i,:] = mean
				i += 1
		return newdata


	def order_genes_by_position(self,data,genes):
		mapping = pd.read_csv(self.gene_mapping_file_name,sep='\t')
		names = mapping['name2']
		chroms = mapping['chrom']
		starts = mapping['cdsStart']
		ends = mapping['cdsEnd']
		d = dict()
		d2 = dict()
		for i,gene in enumerate(names):
			try:
				if int(chroms[i][3:]) > 0:
					d[gene.upper()] = int(int(chroms[i][3:])*1e10 + int(starts[i]))
					d2[gene.upper()] = str(chroms[i][3:]) + ':' + str(starts[i])
			except:
				None
		positions = []
		posnames = []
		for gene in genes:
			gene = gene.upper()
			if gene in d.keys():
				positions.append(d[gene])
				posnames.append(d2[gene])
			else:
				positions.append(-1)
				posnames.append(-1)
		positions = np.asarray(positions)
		posnames = np.asarray(posnames)
		l = len(positions[positions==-1])
		order = np.argsort(positions)
		order = order[l:]
		positions = positions[order]/1e10
		posnames = posnames[order]
		return data[order,:],positions.astype('int'),posnames,order

	def get_labels(self,labels):
		labels = np.asarray(pd.read_csv(data,sep=','))
		self.labels = labels

	def init_params(self,d=1.3,nthreads=1):
		c = np.zeros((self.data.shape[0],self.n_clusters))
		for i in range(self.data.shape[0]):
			for k in range(self.n_clusters):
				c[i,k] = np.mean(self.data[i,self.labels==k])
		labels = np.zeros((self.data.shape[0],self.n_clusters))
		diffs = []
		for i in range(0,self.data.shape[0],10):
			diffs.append(ks_2samp(self.normal[i,:]+np.std(self.normal[i,:])/d,self.normal[i,:])[1])
		diff = np.mean(diffs)
		logger.info(str(diff))

		pool = mp.Pool(nthreads)
		results = pool.map(partial(init_helper, data=self.data, n_clusters=self.n_clusters,normal=self.normal,diff=diff,labels=self.labels,c=c), [x for x in range(self.data.shape[0])])
		for i in range(len(results)):
			labels[i,:] = results[i]
		labels = labels.astype(int)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			means = [np.mean(c[labels==cluster]) for cluster in range(self.num_states)]
			sigmas = [np.std(c[labels==cluster]) for cluster in range(self.num_states)]
			indices = np.argsort([x for x in means])
			states = copy.deepcopy(labels)
		m = np.zeros((3,1))
		s = np.zeros((3,1))
		i=0
		for index in indices:
			states[labels==index]=i # set states
			mean = means[index]
			sigma = sigmas[index]
			if np.isnan(mean) or np.isnan(sigma) or sigma < .01:
				raise ValueError()
			m[i] = [mean]
			s[i] = [sigma**2]
			i+=1
		self.means = m
		self.sigmas = s

	def init_params2(self):
		means = [[],[],[]]
		sigmas = [[],[],[]]
		for s in range(self.num_states):
			d=[]
			for cluster in range(self.n_clusters):
				dat = np.asarray(list(self.data[:,self.labels==cluster]))
				d += list(dat[np.asarray(list(self.states[:,cluster].astype(int)==int(s)))].flatten())
			means[s] = [np.mean(d)]
			sigmas[s] = [np.std(d)**2]
		logger.info(str(means))
		self.means = np.asarray(means)
		self.sigmas = np.asarray(sigmas)

	def initialize_labels(self):
		dat=self.data

		km = KMeans(n_clusters=self.n_clusters).fit(dat.T)
		clusters = np.asarray(km.predict(dat.T))
		self.labels = clusters
		
	def get_spot_network(self,data,spots,l=1):
		spots = np.asarray([[float(y) for y in x.split('x')] for x in spots])
		if self.platform == "Visium":
			logger.info("Using Visium platform layout.")
			# scale row and col coordinate to make them a regular hexagon with the adjacent hexagon center distance = 1
			scale_row = np.sqrt(3) / 2
			scale_col = 1.0 / 2
			spots[:,0] = spots[:,0] * scale_row
			spots[:,1] = spots[:,1] * scale_col
		spot_network = np.zeros((len(spots),len(spots)))
		for i in range(len(spots)):
			for j in range(i,len(spots)):
				dist = distance.euclidean(spots[i],spots[j])
				spot_network[i,j] = np.exp(-dist/(l)) # exponential covariance
				spot_network[j,i] = spot_network[i,j]
		self.spot_network = spot_network


	def get_gene_network(self,data,genes,l=1):
		genes = np.asarray(genes)
		gene_network = np.zeros((len(genes),len(genes)))
		for i in range(len(genes)):
			for j in range(i,len(genes)):
				dist = j-i
				gene_network[i,j] = np.exp(-dist/(l)) # exponential covariance
				gene_network[j,i] = gene_network[i,j]
		return gene_network

	def _optimalK(self,data, maxClusters=15):
		X_scaled = data
		km_scores= []
		km_silhouette = []
		db_score = []
		for i in range(2,maxClusters):
			km = KMeans(n_clusters=i).fit(X_scaled)
			preds = km.predict(X_scaled)

			silhouette = silhouette_score(X_scaled,preds)
			km_silhouette.append(silhouette)
			logger.info("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))

		best_silouette = np.argmax(km_silhouette)+2
		best_db = np.argmin(db_score)+2
		logger.info('silhouette: ' + str(best_silouette))
		return(int(best_silouette))


	def HMM_estimate_states_parallel(self,t,maxiters=100,deltoamp=0,nthreads=1):
		n_clusters = self.n_clusters
		self.EnergyPriors = np.zeros((self.data.shape[0],n_clusters,self.num_states))
		self.t = t
		chromosomes = [int(x.split(':')[0]) for x in self.pos]
		inds = []
		n_clusters = self.n_clusters
		if len(set(self.labels)) != self.n_clusters:
			labels = copy.deepcopy(self.labels)
			i=0
			for label in set(self.labels):
				labels[self.labels==label]=i
				i=i+1
			self.labels = labels
			self.n_clusters = len(set(self.labels))
		for chrom in set(chromosomes):
			for k in range(self.n_clusters):
				inds.append([np.asarray([i for i in range(len(chromosomes)) if chromosomes[i] == chrom]),np.asarray([i for i in range(len(self.labels)) if self.labels[i]==k]),k])
		pool = mp.Pool(nthreads)
		results = pool.map(partial(HMM_helper, data=self.data, means = self.means, sigmas = self.sigmas,t = self.t,num_states = self.num_states,model=self.model,normal=self.normal), inds)
		score = 0
		for i in range(len(results)):
			self.states[inds[i][0][:, None],inds[i][2]] = results[i][0].reshape((len(results[i][0]),1))
			score += results[i][1]
		return score


	def jointLikelihoodEnergyLabels(self,norms,pool):
		Z = (2*math.pi)**(self.num_states/2)
		n_clusters = self.n_clusters
		likelihoods = np.zeros((self.data.shape[1],n_clusters))
		results = pool.map(partial(jointLikelihoodEnergyLabels_helper, data=self.data, states=self.states,norms=norms), range(n_clusters))
		for label in range(n_clusters):
			likelihoods[:,label] += results[label]
		likelihoods = likelihoods / self.data.shape[0]
		likelihood_energies = likelihoods
		return(likelihood_energies)

	def jointLikelihoodEnergyLabelsapprox(self,means):
		e = 1e-20
		n_clusters = self.n_clusters
		likelihoods = np.zeros((self.data.shape[1],n_clusters))
		for spot in range(self.spots):
			ml=np.inf
			for label in range(n_clusters):
				likelihood = np.sum(abs(self.data[:,spot]-means[:,label]))/self.data.shape[0]
				if likelihood < ml:
					ml = likelihood
				likelihoods[spot,label] = likelihood
			likelihoods[spot,:]-=ml
		likelihood_energies = likelihoods
		return(likelihood_energies)

	def MAP_estimate_labels(self,beta_spots,nthreads,maxiters=20):
		inds_spot = []
		tmp_spot = []
		n_clusters = self.n_clusters
		prev_labels = copy.deepcopy(self.labels)
		for j in range(self.spots):
			inds_spot.append(np.where(self.spot_network[j,:] >= .25)[0])
			tmp_spot.append(self.spot_network[j,inds_spot[j]])
		logger.debug(str(tmp_spot))
		pool = mp.Pool(nthreads)
		norms = [norm(self.means[0][0],np.sqrt(self.sigmas[0][0])),norm(self.means[1][0],np.sqrt(self.sigmas[1][0])),norm(self.means[2][0],np.sqrt(self.sigmas[2][0]))]
		for m in range(maxiters):
			posteriors = 0
			means = np.zeros((self.bins,n_clusters))
			for label in range(n_clusters):
				means[:,label] = np.asarray([self.means[int(i)][0] for i in self.states[:,label]])
			likelihood_energies = self.jointLikelihoodEnergyLabels(norms,pool)
			#likelihood_energies = self.jointLikelihoodEnergyLabelsapprox(means)
			for j in range(self.spots):
				p = [((np.sum(tmp_spot[j][self.labels[inds_spot[j]] != label]))) for label in range(n_clusters)]
				val = [likelihood_energies[j,label]+beta_spots*1*p[label] for label in range(n_clusters)]
				arg = np.argmin(val)
				posteriors += val[arg]
				self.labels[j] = arg
			if np.array_equal(np.asarray(prev_labels),np.asarray(self.labels)): # check for convergence
				break
			prev_labels = copy.deepcopy(self.labels)
		return(-posteriors)


	def update_params(self):
		c = np.zeros((self.data.shape[0],self.n_clusters))
		for i in range(self.data.shape[0]):
			for k in range(self.n_clusters):
				c[i,k] = np.mean(self.data[i,self.labels==k])
		means = [np.mean(c[self.states==cluster]) for cluster in range(self.num_states)]
		sigmas = [np.std(c[self.states==cluster]) for cluster in range(self.num_states)]

		indices = np.argsort([x for x in means])

		m = np.zeros((3,1))
		s = np.zeros((3,1))
		i=0
		for index in indices:
			self.states[self.states==index]=i # set states
			mean = means[index]
			sigma = sigmas[index]
			m[i] = [mean]
			s[i] = [sigma**2]
			i+=1
		self.means = m
		self.sigmas = s
		logger.debug(str(self.means))
		logger.debug(str(self.sigmas))

	def callCNA(self,t=.00001,beta_spots=2,maxiters=20,deltoamp=0.0,nthreads=0,returnnormal=True):
		"""
		Run HMRF-EM framework to call CNA states by alternating between
		MAP estimate of states given current params and EM estimate of
		params given current states until convergence

		Returns:
			states (np array): integer CNA states (0 = del, 1 norm, 2 = amp)
		"""
		logger.info("running HMRF to call CNAs...")
		states = [copy.deepcopy(self.states),copy.deepcopy(self.states)]
		logger.debug('sum start:'+str(np.sum(states[-1])))
		logger.info('beta spots: '+str(beta_spots))
		if nthreads == 0:
			nthreads = int(multiprocessing.cpu_count() / 2 + 1)
			logger.info('Running with ' + str(nthreads) + ' threads')
		X = []
		lengths = []
		for i in range(self.data.shape[1]):
			X.append([[x] for x in self.data[:,i]])
			lengths.append(len(self.data[:,i]))
		X = np.concatenate(X)
		model = hmm.GaussianHMM(n_components=self.num_states, covariance_type="diag",init_params="mc", params="",algorithm='viterbi')
		model.transmat_ = np.array([[1-2*t, t, t],
									[t, 1-2*t, t],
									[t, t, 1-2*t]])
		model.startprob_ = np.asarray([.1,.8,.1])
		model.means_ = self.means
		model.covars_ = self.sigmas
		model.fit(X,lengths)
		logger.info("fitted HMM means: " + str(model.means_))
		logger.info("fitted HMM covariance matrices: " + str(model.covars_))
		logger.info("fitted HMM transition matrix: " + str(model.transmat_))
		logger.info("fitted HMM starting probability: " + str(model.startprob_))
		self.model = model
		for i in range(maxiters):
			score_state = self.HMM_estimate_states_parallel(t=t,deltoamp=deltoamp,nthreads=nthreads)
			self.init_params2()
			score_label = self.MAP_estimate_labels(beta_spots=beta_spots,nthreads=nthreads,maxiters=20)
			states.append(copy.deepcopy(self.states))
			logger.debug('sum iter:'+str(i) + ' ' + str(np.sum(states[-1])))
			if np.array_equal(states[-2],states[-1]) or np.array_equal(states[-3],states[-1]): # check for convergence
				logger.info('states converged')
				break
			if len(states) > 3:
				states = states[-3:]
		logger.info('Posterior Energy: ' + str(score_state + score_label))
		if returnnormal:
			labels = np.asarray([self.n_clusters for i in range(len(self.columns))])
			labels[self.tumor_spots] = self.labels
			states = np.ones((self.states.shape[0],self.n_clusters+1))
			for cluster in range(self.n_clusters):
				states[:,cluster] = self.states[:,cluster]
			self.labels = pd.DataFrame(data=labels,index=self.columns)
			self.states = states
			self.n_clusters += 1
		else:
			self.labels = pd.DataFrame(data=self.labels,index=self.columns[self.tumor_spots])
		states = pd.DataFrame(self.states)
		logger.info(str(len(self.rows)) + ' ' + str(len(np.asarray([i for i in range(self.states.shape[1])]))) + ' ' + str(self.states.shape))
		self.states = pd.DataFrame(self.states, index=self.rows,columns=np.asarray([i for i in range(self.states.shape[1])]))
		logger.debug(str(self.states))
		logger.debug(str(self.labels))
		return(score_state + score_label) # return CNA states





