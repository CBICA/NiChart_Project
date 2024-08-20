import numpy as np
import scipy
import pandas as pd
import itertools
import os
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn import metrics
from .data_loading import PTIterator, CNIterator, Paired_data_iterator, val_PT_construction, val_CN_construction, val_Gene_construction
from .model import GeneSGAN

import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
utils = rpackages.importr('utils')
packages = ('nnet')
nnet = importr('nnet')

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

def find_highest_ari_model(clustering_results):
	"""
	Find one of save models which have the highest average overlap (evaluated by ARI) with all other saved models
	The returned model will be used as a template such that all other models will \
	be reordered to achieve the highest overlap with it.
	:param clustering_results: list, list of clustering results given by all saved models; 
							   length of list equals the number of saved models.
	:return: int, the index of selected model with highest average overlap
	"""
	highest_ari=0
	best_model=0
	for i in range(len(clustering_results)):
		all_ari=[]
		for j in range(len(clustering_results)):
			if i!=j:all_ari.append(metrics.adjusted_rand_score(clustering_results[i], clustering_results[j]))
		if np.mean(all_ari)>highest_ari:
			best_model=i
			highest_ari=np.mean(all_ari)
	return best_model


def get_model_order(cluster_results,ncluster):
	"""
	Find best orders for results given by all saved models so that they reach highest agreements
	:param clustering_results: list, list of clustering results given by all saved models; 
							   length of list equals the number of saved models.
	:param ncluster: int, number of clusters
	:return: list, list of best orders for results given by all saved models.
	"""
	order_permutation = list(itertools.permutations(range(ncluster)))
	best_model = find_highest_ari_model(cluster_results)
	all_orders=[]
	for k in range(len(cluster_results)):
		if k==best_model: 
			all_orders.append(range(ncluster))
		elif k!=best_model:
			highest_intersection = 0
			for order in order_permutation:
				total_intersection = 0
				for i in range(ncluster):
					total_intersection+= np.intersect1d(np.where(cluster_results[best_model]==i),np.where(cluster_results[k]==order[i])).shape[0]
				if total_intersection>=highest_intersection:
					best_order=order
					highest_intersection=total_intersection
			all_orders.append(best_order)
	return all_orders

def highest_matching_clustering(clustering_results, label_probability, ncluster):
	"""
	The function which offers clustering result (cluster label and cluster probabilities) 
	by reordering and combining clustering results given by all saved models
	:param clustering_results: list, list of clustering results given by all saved models; 
							   length of list equals the number of saved models.
	:param label_probability: list, list of clutering results given by all saved modesl;
								length of list equals the number of saved models and each 
								model gives an n*k list.
	:param ncluster: int, number of clusters
	:return: two arrays, one n*1 array with cluster label for each participant
						 one n*k array with k cluster probabilites for each participant
	"""
	order = get_model_order(clustering_results, ncluster)
	class_index=0
	for i in range(len(clustering_results)):
		label_probability[i] = label_probability[i][:,order[i]]
	prediction_prob=np.mean(label_probability,axis=0)
	prediction_cluster=prediction_prob.argmax(axis=1)
	return prediction_cluster, prediction_prob, 

def Covariate_correction(cn_data,cn_cov,pt_data,pt_cov):
	"""
	Eliminate the confound of covariate, such as age and sex, from the disease-based changes.
	:param cn_data: array, control data
	:param cn_cov: array, control covariates
	:param pt_data: array, patient data
	:param pt_cov: array, patient covariates
	:return: corrected control data & corrected patient data
	"""
	min_cov = np.amin(cn_cov, axis=0)
	max_cov = np.amax(cn_cov, axis=0)
	pt_cov = (pt_cov-min_cov)/(max_cov-min_cov)
	cn_cov = (cn_cov-min_cov)/(max_cov-min_cov)
	beta = np.transpose(LinearRegression().fit(cn_cov, cn_data).coef_)
	corrected_cn_data = cn_data-np.dot(cn_cov,beta)
	corrected_pt_data = pt_data-np.dot(pt_cov,beta)
	correction_variables = {'max_cov':max_cov,'min_cov':min_cov,'beta':beta}
	return corrected_cn_data, corrected_pt_data, correction_variables

def Data_normalization(cn_data,pt_data):
	"""
	Normalize all data with respect to control data to ensure a mean of 1 and std of 0.1 
	among CN participants for each ROI
	:param cn_data: array, control data
	:param pt_data: array, patient data
	:return: normalized control data & normalized patient data
	"""
	cn_mean = np.mean(cn_data,axis=0)
	cn_std = np.std(cn_data,axis=0)
	normalized_cn_data = 1+(cn_data-cn_mean)/(10*cn_std)
	normalized_pt_data = 1+(pt_data-cn_mean)/(10*cn_std)
	normalization_variables = {'cn_mean':cn_mean, 'cn_std':cn_std}
	return normalized_cn_data, normalized_pt_data, normalization_variables

def apply_covariate_correction(data, covariate, correction_variables):
	covariate = (covariate-correction_variables['min_cov'])/(correction_variables['max_cov']-correction_variables['min_cov'])
	corrected_data = data-np.dot(covariate,correction_variables['beta'])
	return corrected_data

def apply_data_normalization(data, normalization_variables):
	normalized_data = 1+(data-normalization_variables['cn_mean'])/(10*normalization_variables['cn_std'])
	return normalized_data

def parse_train_data(imaging_data, gene_data, covariate, random_seed, data_fraction, batch_size):
	np.random.seed(random_seed)
	n_pt_data = imaging_data.loc[imaging_data['diagnosis'] != -1].shape[0]
	n_cn_data = imaging_data.loc[imaging_data['diagnosis'] == -1].shape[0]
	indices_pt = np.random.choice(n_pt_data, int(data_fraction*n_pt_data), replace=False)
	np.random.seed(random_seed)
	indices_cn = np.random.choice(n_cn_data, int(data_fraction*n_cn_data), replace=False)
	cn_data = imaging_data.loc[imaging_data['diagnosis'] == -1].drop(['participant_id','diagnosis'], axis=1).values[indices_cn]
	pt_data = imaging_data.loc[imaging_data['diagnosis'] !=- 1].drop(['participant_id','diagnosis'], axis=1).values[indices_pt]
	gene_data = imaging_data[['participant_id']].merge(gene_data,on='participant_id')
	gene_data_fillna = gene_data.fillna(gene_data.mean())
	gene_data = gene_data.drop(['participant_id'], axis=1).values[indices_pt]
	gene_data_fillna = gene_data_fillna.drop(['participant_id'], axis=1).values[indices_pt]
	correction_variables = None
	if covariate is not None:
		cn_cov = covariate.loc[covariate['diagnosis'] == -1].drop(['participant_id', 'diagnosis'], axis=1).values[indices_cn]
		pt_cov = covariate.loc[covariate['diagnosis'] != -1].drop(['participant_id','diagnosis'], axis=1).values[indices_pt]
		cn_data,pt_data,correction_variables = Covariate_correction(cn_data,cn_cov,pt_data,pt_cov)
	normalized_cn_data, normalized_pt_data,normalization_variables = Data_normalization(cn_data,pt_data)
	cn_train_dataset = CNIterator(normalized_cn_data, batch_size)
	pt_train_dataset = PTIterator(normalized_pt_data, batch_size)
	pair_train_dataset = Paired_data_iterator(gene_data, gene_data_fillna, normalized_pt_data,  data_fraction, batch_size, random_seed)
	return cn_train_dataset, pt_train_dataset, pair_train_dataset, correction_variables, normalization_variables

def parse_validation_data(imaging_data, gene_data, covariate, random_seed, data_fraction, correction_variables, normalization_variables):
	cn_data = imaging_data.loc[imaging_data['diagnosis'] == -1].drop(['participant_id','diagnosis'], axis=1).values
	pt_data = imaging_data.loc[imaging_data['diagnosis'] != -1].drop(['participant_id','diagnosis'], axis=1).values
	gene_data = gene_data.drop(['participant_id'], axis=1).values
	if covariate is not None:
		cn_cov = covariate.loc[covariate['diagnosis'] == -1].drop(['participant_id', 'diagnosis'], axis=1).values
		pt_cov = covariate.loc[covariate['diagnosis'] != -1].drop(['participant_id','diagnosis'], axis=1).values
		cn_data = apply_covariate_correction(cn_data, cn_cov, correction_variables)
		pt_data = apply_covariate_correction(pt_data, pt_cov, correction_variables)
	normalized_cn_data = apply_data_normalization(cn_data, normalization_variables)
	normalized_pt_data = apply_data_normalization(pt_data, normalization_variables)
	cn_train_dataset = val_CN_construction(normalized_cn_data, random_seed, data_fraction).load_train()
	pt_train_dataset = val_PT_construction(normalized_pt_data, random_seed, data_fraction).load_train()
	gene_train_dataset = val_Gene_construction(gene_data, random_seed, data_fraction,).load_train()
	pt_test_dataset = val_PT_construction(normalized_pt_data, random_seed, data_fraction).load_test()
	gene_test_dataset = val_Gene_construction(gene_data, random_seed, data_fraction,).load_test()
	return cn_train_dataset, pt_train_dataset, gene_train_dataset, pt_test_dataset, gene_test_dataset

def mnlogit(test_dataframe):
	with localconverter(ro.default_converter+pandas2ri.converter):
		r_dataframe_test = ro.conversion.py2rpy(test_dataframe)
	if test_dataframe.columns.shape[0]>1:
		formula = 'label~'+test_dataframe.columns[0]
		for i in range(1,test_dataframe.columns.shape[0]-1):
			formula+="+"+test_dataframe.columns[i]
	else:
		formula = 'label~1'
	fmla = Formula(formula)
	model = nnet.multinom(fmla,data=r_dataframe_test,trace=False)
	pred_prob = np.array(ro.r.predict(model,type="probs",newdata = r_dataframe_test))
	return -metrics.log_loss(test_dataframe['label'],pred_prob,normalize=False)


def n_sig_calculation(model_dir, image_data, covariate_data, gene_data, fraction):
	n_sig_snp = []
	fold_number = len(model_dir)
	if covariate_data is not None:
		pt_cov = covariate_data.loc[covariate_data['diagnosis'] != -1].drop(['participant_id','diagnosis'], axis=1).values
	else:
		pt_cov = None
	n_pt = gene_data.values.shape[0]
	i=0
	for saved_model in model_dir:
		i+=1
		np.random.seed(i)
		indices = np.random.choice(n_pt, int(fraction*n_pt), replace=False)
		select = np.in1d(range(n_pt), indices)
		if pt_cov is not None:
			test_cov = pt_cov[~select]
		else:
			test_cov = None
		model = GeneSGAN()
		model.load(saved_model)
		cn_data, train_pt_data, train_gene_data, val_pt_data, val_gene_data = parse_validation_data(image_data, gene_data, covariate_data, i, fraction, model.opt.correction_variables, model.opt.normalization_variables)
		test_label = np.argmax(model.predict_cluster(val_pt_data), axis=1)
		n_sig_snp.append(likelihood_ratio_n(test_label,val_gene_data,test_cov))
	return np.mean(n_sig_snp), np.std(n_sig_snp)


def likelihood_ratio_n(test_label, test_snp, test_cov):
	if test_cov is not None:
		test_cov = test_cov.astype(float)
		test_cov  = (test_cov -np.mean(test_cov ,axis=0))/np.std(test_cov ,axis=0)
		test_model_data = pd.DataFrame(data=test_cov,columns=['covariate'+str(_) for _ in range(test_cov.shape[1])])
		test_model_data['label']=test_label
	else:
		test_model_data = pd.DataFrame()
		test_model_data['label']=test_label
	reduce_llf = mnlogit(test_model_data)
	all_ll_difference = []
	n = 0
	for i in range(test_snp.shape[1]):
		if test_cov is not None:
			X_full_test = np.concatenate((test_cov,test_snp[:,i].numpy().reshape(-1,1)),axis=1)
			test_model_data = pd.DataFrame(data=X_full_test,columns=['covariate'+str(_) for _ in range(test_cov.shape[1])]+['snp'])
		else:
			test_model_data = pd.DataFrame(data=test_snp[:,i].numpy().reshape(-1,1),columns=['snp'])
		test_model_data['label']=test_label
		test_model_data = test_model_data.dropna()
		full_llf = mnlogit(test_model_data)
		all_ll_difference.append(full_llf-reduce_llf)
		if full_llf-reduce_llf>=3.84:
			n+=1
	return n



