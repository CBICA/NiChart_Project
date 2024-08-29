import pandas as pd
from GeneSGAN.Gene_SGAN_clustering import cross_validated_clustering
import os

if __name__ == '__main__':
	output_dir = './genesgan'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	image_data = pd.read_csv('toy_data_imaging.csv')
	gene_data = pd.read_csv('toy_data_gene.csv')

	fold_number = 1
	ncluster = 3
	start_saving_epoch = 20000
	max_epoch = 30000
	WD = 0.11
	AQ = 30
	cluster_loss = 0.01
	genelr = 0.0002


	cross_validated_clustering(image_data, gene_data, ncluster, fold_number, 0.8, start_saving_epoch, max_epoch, output_dir, WD, AQ, cluster_loss,\
		genelr = 0.0002, batchSize=120, lipschitz_k=0.5)



