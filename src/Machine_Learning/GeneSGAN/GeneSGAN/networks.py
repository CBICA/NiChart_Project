import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from .modules import TwoInputSequential, Sub_Adder

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
       m.weight.data.normal_(0, 0.12)

def define_Linear_Transform(nROI,nCluster,nLatent):
    netG = LTransformGenerator(nCluster,nLatent, nROI)
    netG.apply(weights_init)
    return netG

def define_Linear_Discriminator(nROI):
    netD=LDiscriminator(nROI)
    netD.apply(weights_init)
    return netD

def define_Linear_Inverse(nROI,nCluster,nLatent):
    netC = LInverse(nCluster,nLatent, nROI)
    netC.apply(weights_init)
    return netC

def define_Gene_Inference(nGene,nLatent,nCluster):
    netG = gene_inference(nGene,nLatent,nCluster)
    netG.apply(weights_init)
    return netG

def define_z3_Posterior(nGene,nROI,nLatent):
    netI = z3_posterir(nGene,nROI,nLatent)
    netI.apply(weights_init)
    return netI

##############################################################################
# Network Classes
##############################################################################

class LTransformGenerator(nn.Module):
    def __init__(self, nCluster, nLatent, nROI, product_layer=Sub_Adder):
        super(LTransformGenerator, self).__init__()
        model=[]
        def block(in_layer, out_layer, normalize=False):
            layers = [nn.Linear(in_layer, out_layer,bias=False)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        model+=block(nROI,int(nROI/2))+block(int(nROI/2), int(nROI/4))
        model.append(product_layer(int(nROI/4),nCluster+nLatent))
        model+=block(int(nROI/4),int(nROI/2))+block(int(nROI/2),nROI)
        model.append(nn.Linear(nROI, nROI,bias=False))
        self.model = TwoInputSequential(*model)
    def forward(self, input_x,input_z):
        return self.model(input_x,input_z)

class LInverse(nn.Module):
    def __init__(self, nCluster, nLatent, nROI):
        super(LInverse, self).__init__()
        model=[]
        self.nLatent = nLatent
        def block(in_layer, out_layer, normalize=True):
            layers=[nn.LeakyReLU(0.2, inplace=True)]
            layers.append(nn.Linear(in_layer, out_layer))
            return layers
        model.append(nn.Linear(nROI, nROI))
        model+=block(nROI, int(nROI/2), normalize=False)\
            +block(int(nROI/2), int(nROI/4))\
            +block(int(nROI/4), nCluster+nLatent)
        self.model = nn.Sequential(*model)

    def forward(self, input_y):
        z = self.model(input_y)
        z2 = z[:, 0:self.nLatent]
        z1 = z[:, self.nLatent:]
        z1_softmax = F.softmax(z1, dim=1)
        return z1_softmax, z1, z2

class LDiscriminator(nn.Module):
    def __init__(self, nROI):
        super(LDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(nROI, int(nROI/2),bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(nROI/2), int(nROI/4),bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(nROI/4), 2,bias=True),
        )
    def forward(self, input_y):
        pred = self.model(input_y)
        return pred

class z3_posterir(nn.Module):
    def __init__(self,nGene,nROI,nLatent):
        super(z3_posterir, self).__init__()
        self.model=nn.Sequential(nn.Linear((nGene+nROI),int((nGene+nROI)/2)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int((nGene+nROI)/2),int((nGene+nROI)/4)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(int((nGene+nROI)/4), nLatent*2))

    def forward(self, input_x):
        mu, logsigma = self.model(input_x).chunk(2, dim=-1)
        return D.Normal(mu, abs(logsigma.exp()))

class gene_inference(nn.Module):
    def __init__(self,nGene,nLatent,nCluster):
        super(gene_inference, self).__init__()
        self.model=nn.Sequential(nn.Linear((nCluster+nLatent),2*(nCluster+nLatent)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(2*(nCluster+nLatent),4*(nCluster+nLatent)),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(4*(nCluster+nLatent), nGene))

    def forward(self, input_y,input_z):
        yz = torch.cat([input_y, input_z], dim=1)
        logits = self.model(yz)
        return D.Binomial(total_count=2,logits=logits), torch.sigmoid(logits)


