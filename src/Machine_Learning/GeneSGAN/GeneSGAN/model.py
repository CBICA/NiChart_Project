import os
import torch
import torch.nn.functional as F
import torch.distributions as D
from collections import OrderedDict
from torch.autograd import Variable
from itertools import chain as ichain
from .networks import define_Linear_Transform, define_Linear_Inverse, define_Linear_Discriminator, define_Gene_Inference, define_z3_Posterior

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"


#####sample from discrete uniform random variable and construct SUB variable. 
def sample_z_categorical(x, ncluster, fix_class=-1):
	"""
	sample from discrete uniform random variable and construct SUB variable.

	:param x, torch tensor with real data 
	:param ncluster: int, defined number of clusters
	:param fix_class: int, can be set to certain mapping directions to generate data in only one cluster,
						   set to -1 for random sampling from discrete uniform distribution

	:return tensor with sampled or selected mapping directions & 
			tensors with shape n*k (each row is a one-hot vector depending on sampled directions) 
	"""
	Tensor = torch.FloatTensor
	z1 = Tensor(x.size(0), ncluster).fill_(0)
	z1_idx = torch.empty(x.size(0), dtype=torch.long)
	if (fix_class == -1):
		z1_idx = z1_idx.random_(ncluster)
		z1 = z1.scatter_(1, z1_idx.unsqueeze(1), 1.)
	else:
		z1_idx[:] = fix_class
		z1 = z1.scatter_(1, z1_idx.unsqueeze(1), 1.)
		z1[0,fix_class]
	z1_var = Variable(z1)
	return z1_var, z1_idx

def sample_z_continuous(x, n_z2):
	z2=Variable(x.data.new(x.size(0),n_z2).uniform_(0,1))
	return z2

def criterion_GAN(pred, target_is_real, prob):
	if target_is_real:
		target_var = Variable(pred.data.new(pred.shape[0]).long().fill_(0.))
		loss=(F.cross_entropy(pred, target_var, reduce=False).unsqueeze(-1)*prob).mean()
	else:
		target_var = Variable(pred.data.new(pred.shape[0]).long().fill_(1.))
		loss = (F.cross_entropy(pred, target_var, reduce=False).unsqueeze(-1)*prob).mean()
	return loss

def criterion_VAE(v, z, p_z, p_v_z, q_z_v):
		v_nan_mask = ~v.isnan()
		v = torch.nan_to_num(v,nan=0)
		return (- torch.masked_select(p_v_z.log_prob(v),v_nan_mask).sum() \
			   + q_z_v.log_prob(z).sum()\
			   - p_z.log_prob(z).sum())/v.shape[0]

def KL_divergence(P,Q):
	return (P*(torch.log(P)-torch.log(Q))).sum()

class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

class GeneSGAN(object):
	def __init__(self):
		self.opt = None

		##### definition of all netwotks
		self.netMapping = None
		self.netClustering= None
		self.netDiscriminator = None

		##### definition of all optimizers
		self.optimizer_M = None
		self.optimizer_D = None

		##### definition of all criterions
		self.criterionGAN = criterion_GAN
		self.criterionChange = F.l1_loss
		self.criterionCluster = F.cross_entropy
		self.criterionLatent = F.mse_loss
		self.criterionVAE = criterion_VAE


	def create(self, opt):

		self.opt = opt

		## definition of networkds for GAN
		self.netTransform = define_Linear_Transform(self.opt.nROI,self.opt.ncluster,self.opt.n_z2)
		self.netInverse = define_Linear_Inverse(self.opt.nROI,self.opt.ncluster,self.opt.n_z2)
		self.netDiscriminator = define_Linear_Discriminator(self.opt.nROI)
		self.phi = torch.nn.Parameter(torch.ones(self.opt.ncluster)/self.opt.ncluster)

		### definition of networkds for VAE
		self.netGeneInference = define_Gene_Inference(self.opt.nGene,self.opt.n_z3,self.opt.ncluster)
		self.netGeneZ3Posterior = define_z3_Posterior(self.opt.nGene,self.opt.nROI,self.opt.n_z3)

		## definition of all optimizers
		self.optimizer_M = torch.optim.Adam(ichain(self.netTransform.parameters(),self.netInverse.parameters()),
										lr=self.opt.lr, betas=(self.opt.beta1, 0.999),weight_decay=2.5*1e-3)
		self.optimizer_D = torch.optim.Adam(ichain(self.netDiscriminator.parameters()),
											lr=self.opt.lr/5., betas=(self.opt.beta1, 0.999))
		self.optimizer_Gene_VAE = torch.optim.Adam(ichain(self.netGeneInference.parameters(),self.netInverse.parameters(),self.netGeneZ3Posterior.parameters()),
											lr=self.opt.genelr)
		self.optimizer_phi = torch.optim.Adam(iter([self.phi]),
											lr=self.opt.lr/40., betas=(self.opt.beta1, 0.999))

		##### definition of prior probability
		self.p_z3 = D.Normal(torch.tensor(0.), torch.tensor(1.))


	def train_instance(self, x, real_y, pair_gene, pair_gene_filled, pair_y, kappa):

		initial = torch.ones(self.opt.ncluster)/self.opt.ncluster
		z1, z1_index = sample_z_categorical(x, self.opt.ncluster)
		z2 = sample_z_continuous(x, self.opt.n_z2)
		z_1_2 = torch.cat((z1,z2),1)
		fake_y = self.netTransform.forward(x,z_1_2)+x
		softmax_phi = F.softmax(self.phi, dim=0)
		post_prob = torch.gather(torch.stack([softmax_phi for _ in range(z1_index.shape[0])], dim=0),1,z1_index.unsqueeze(-1))

		## Discriminator loss
		pred_fake_y = self.netDiscriminator.forward(fake_y.detach())
		loss_D_fake_y = self.criterionGAN(pred_fake_y, False,post_prob.detach())
		pred_true_y = self.netDiscriminator.forward(real_y)
		loss_D_true_y = self.criterionGAN(pred_true_y, True,post_prob.detach())
		loss_D= 0.5* (loss_D_fake_y + loss_D_true_y)

		## update weights of discriminator
		self.optimizer_D.zero_grad()
		loss_D.backward()
		gnorm_D = torch.nn.utils.clip_grad_norm_(self.netDiscriminator.parameters(), self.opt.max_gnorm)
		self.optimizer_D.step()

		## Phi loss
		loss_phi = self.criterionGAN(self.netDiscriminator.forward(fake_y.detach()).detach(), True, post_prob)+kappa*KL_divergence(initial, softmax_phi)

		self.optimizer_phi.zero_grad()
		loss_phi.backward()
		self.optimizer_phi.step()

		## GAN/Change/Recons loss
		pred_fake_y = self.netDiscriminator.forward(fake_y)
		loss_GAN = self.criterionGAN(pred_fake_y, True, post_prob.detach())

		reconst_z1_softmax, reconst_z1, reconst_z2 = self.netInverse.forward(fake_y)
		z1_loss = self.criterionCluster(reconst_z1, z1_index)
		z2_loss = self.criterionLatent(reconst_z2, z2)

		change_loss= self.criterionChange(fake_y, x)
		loss_G = (1/torch.mean(post_prob.detach()))*loss_GAN+self.opt.lam*(z1_loss+z2_loss)+self.opt.mu*change_loss

		## update weights of Transformation and Inverse function
		self.optimizer_M.zero_grad()
		loss_G.backward()
		gnorm_M = torch.nn.utils.clip_grad_norm_(self.netTransform.parameters(), self.opt.max_gnorm)
		self.optimizer_M.step()

		##########################
		#VAE step for Genetic Data
		##########################
		estimated_z1, __ , __ = self.netInverse.forward(pair_y)

		pair_v_y = torch.cat([(pair_gene_filled-1.0)/5., pair_y], dim=1)
		q_z3 = self.netGeneZ3Posterior.forward(pair_v_y)
		z3 = q_z3.rsample()
		p_infered_v, v_prob = self.netGeneInference.forward(estimated_z1, z3)
		VAE_loss = self.criterionVAE(pair_gene, z3, self.p_z3, p_infered_v, q_z3)

		self.optimizer_Gene_VAE.zero_grad()
		VAE_loss.backward()
		self.optimizer_Gene_VAE.step()

		losses=OrderedDict([('Vae_loss', VAE_loss.item()),('Discriminator_loss', loss_D.item()),('Mapping_loss', loss_GAN.item()),('loss_change', change_loss.item()),('loss_cluster', z1_loss.item())])

		## perform weight clipping
		for p in self.netTransform.parameters():
			p.data.clamp_(-self.opt.lipschitz_k, self.opt.lipschitz_k)
		for p in self.netInverse.parameters():
			p.data.clamp_(-self.opt.lipschitz_k, self.opt.lipschitz_k)

		return losses
	
	## return a numpy array of pattern type probabilities for all input subjects
	def predict_cluster(self,real_y):
		prediction,z1,z2=self.netInverse.forward(real_y)
		return prediction.detach().numpy()

	## return generated patient data with given zub variable
	def predict_Y(self, x, z):
		return self.netTransform.forward(x, z)+x

	def predict_gene_prob(self, z1, z2):
		return self.netGeneInference.forward(z1, z2)[0].probs

	def get_phi(self):
		return F.softmax(self.phi.detach(), dim=0)

	## save checkpoint    
	def save(self, save_dir, chk_name):
		chk_path = os.path.join(save_dir, chk_name)
		checkpoint = {
			'netTransform':self.netTransform.state_dict(),
			'netDiscriminator':self.netDiscriminator.state_dict(),
			'optimizer_D':self.optimizer_D.state_dict(),
			'optimizer_M':self.optimizer_M.state_dict(),
			'netInverse':self.netInverse.state_dict(),
			'netGeneInference':self.netGeneInference.state_dict(),
			'netZLatentGene':self.netGeneZ3Posterior.state_dict(),
			'optimizer_Gene_VAE':self.optimizer_Gene_VAE.state_dict(),
			'optimizer_phi':self.optimizer_phi.state_dict()
		}
		checkpoint.update(self.opt)
		torch.save(checkpoint, chk_path)

	def load_opt(self,checkpoint):
		self.opt = dotdict({})
		for key in checkpoint.keys():
			if key not in ['netTransform','netDiscriminator','netInverse','netGeneInference','netZLatentGene','optimizer_M','optimizer_D','optimizer_Gene_VAE','optimizer_phi']:
				self.opt[key] = checkpoint[key]
		

	## load trained model
	def load(self, chk_path):
		checkpoint = torch.load(chk_path)
		self.load_opt(checkpoint)

		## definition of networkds for GAN
		self.netTransform = define_Linear_Transform(self.opt.nROI,self.opt.ncluster,self.opt.n_z2)
		self.netInverse = define_Linear_Inverse(self.opt.nROI,self.opt.ncluster,self.opt.n_z2)
		self.netDiscriminator = define_Linear_Discriminator(self.opt.nROI)
		self.phi = torch.nn.Parameter(torch.ones(self.opt.ncluster)/self.opt.ncluster)

		### definition of networkds for VAE
		self.netGeneInference = define_Gene_Inference(self.opt.nGene,self.opt.n_z3,self.opt.ncluster)
		self.netGeneZ3Posterior = define_z3_Posterior(self.opt.nGene,self.opt.nROI,self.opt.n_z3)


		## definition of all optimizers
		self.optimizer_M = torch.optim.Adam(ichain(self.netTransform.parameters(),self.netInverse.parameters()),
										lr=self.opt.lr, betas=(self.opt.beta1, 0.999),weight_decay=2.5*1e-3)
		self.optimizer_D = torch.optim.Adam(ichain(self.netDiscriminator.parameters()),
											lr=self.opt.lr/5., betas=(self.opt.beta1, 0.999))
		self.optimizer_Gene_VAE = torch.optim.Adam(ichain(self.netGeneInference.parameters(),self.netInverse.parameters(),self.netGeneZ3Posterior.parameters()),
											lr=self.opt.genelr)
		self.optimizer_phi = torch.optim.Adam(iter([self.phi]),
											lr=self.opt.lr/40., betas=(self.opt.beta1, 0.999))

		##### definition of prior probability
		self.p_z3 = D.Normal(torch.tensor(0.), torch.tensor(1.))
		
		self.netTransform.load_state_dict(checkpoint['netTransform'])
		self.netDiscriminator.load_state_dict(checkpoint['netDiscriminator'])
		self.netInverse.load_state_dict(checkpoint['netInverse'])
		self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
		self.optimizer_M.load_state_dict(checkpoint['optimizer_M'])
		self.netGeneInference.load_state_dict(checkpoint['netGeneInference'])
		self.netGeneZ3Posterior.load_state_dict(checkpoint['netZLatentGene'])
		self.optimizer_Gene_VAE.load_state_dict(checkpoint['optimizer_Gene_VAE'])
		self.optimizer_phi.load_state_dict(checkpoint['optimizer_phi'])
		self.load_opt(checkpoint)
			


		
