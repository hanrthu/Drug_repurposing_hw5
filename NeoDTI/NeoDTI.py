import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

class NeoDTI(nn.Module):
    def __init__(self, num_drug, num_human, num_virus, dim):
        super(NeoDTI, self).__init__()
        # self.drug_embedding = Parameter(torch.normal(mean=torch.zeros([num_drug,dim]),std=0.1))
        self.drug_embedding = Parameter(torch.zeros((num_drug,dim)))
        # self.human_embedding = Parameter(torch.normal(mean=torch.zeros([num_human,dim]),std=0.1))
        self.human_embedding = Parameter(torch.zeros((num_human,dim)))
        # self.virus_embedding = Parameter(torch.normal(mean=torch.zeros([num_virus,dim]),std=0.1))
        self.virus_embedding = Parameter(torch.zeros((num_virus,dim)))
        # self.W0 = Parameter(torch.normal(mean=torch.zeros([2*dim,dim]),std=0.1))
        self.W0 = Parameter(torch.zeros((2*dim,dim)))
        self.b0 = Parameter(torch.normal(mean=torch.zeros([dim]),std=0.1))
        # self.b0 = Parameter(torch.zeros([dim]))
        init.xavier_normal_(self.drug_embedding)
        init.xavier_normal_(self.human_embedding)
        init.xavier_normal_(self.virus_embedding)
        init.xavier_normal_(self.W0)
        # init.xavier_normal_(self.b0)

        self.dd_layer = nn.Sequential(
            nn.Linear(dim,dim,bias=True),
            nn.ReLU()
        )
        self.dv_layer = nn.Sequential(
            nn.Linear(dim,dim,bias=True),
            nn.ReLU()
        )
        self.dh_layer = nn.Sequential(
            nn.Linear(dim,dim,bias=True),
            nn.ReLU()
        )
        self.hd_layer = nn.Sequential(
            nn.Linear(dim,dim,bias=True),
            nn.ReLU()
        )
        self.hv_layer = nn.Sequential(
            nn.Linear(dim,dim,bias=True),
            nn.ReLU()
        )
        self.hh_layer = nn.Sequential(
            nn.Linear(dim,dim,bias=True),
            nn.ReLU()
        )
        self.hhi_layer = nn.Sequential(
            nn.Linear(dim,dim,bias=True),
            nn.ReLU()
        )
        self.vd_layer = nn.Sequential(
            nn.Linear(dim,dim,bias=True),
            nn.ReLU()
        )
        self.vh_layer = nn.Sequential(
            nn.Linear(dim,dim,bias=True),
            nn.ReLU()
        )
        self.vv_layer = nn.Sequential(
            nn.Linear(dim,dim,bias=True),
            nn.ReLU()
        )

    def bi_layer(self,x0,x1,sym,dim_pred):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_tensor_type(torch.DoubleTensor)
        if sym == False:
            # W0p = Parameter(torch.normal(mean=torch.zeros([x0.shape[1],dim_pred]),std=0.1)).to(device)
            # W1p = Parameter(torch.normal(mean=torch.zeros([x1.shape[1],dim_pred]),std=0.1)).to(device)
            W0p = Parameter(torch.zeros((x0.shape[1],dim_pred))).to(device)
            W1p = Parameter(torch.zeros((x1.shape[1],dim_pred))).to(device)
            init.xavier_normal_(W0p)
            init.xavier_normal_(W1p)
            return torch.matmul(torch.matmul(x0, W0p), 
                                (torch.matmul(x1, W1p)).T)
        else:
            # W0p = Parameter(torch.normal(mean=torch.zeros([x0.shape[1],dim_pred]),std=0.1)).to(device)
            W0p = Parameter(torch.zeros((x0.shape[1],dim_pred))).to(device)
            init.xavier_normal_(W0p)
            return torch.matmul(torch.matmul(x0, W0p), 
                                (torch.matmul(x1, W0p)).T)


    def forward(self,drug_protein,drug_protein_norm,drug_human,drug_human_norm,drug_drug,drug_drug_norm,human_human,human_human_norm,
    human_human_integration,human_human_integration_norm,human_drug,human_drug_norm,human_virus,human_virus_norm,virus_virus,virus_virus_norm,
    virus_human,virus_human_norm,protein_drug,protein_drug_norm,drug_protein_mask): 
        self.drug_representation = F.normalize(F.relu(torch.matmul(
            torch.cat([torch.matmul(drug_drug_norm, self.dd_layer(self.drug_embedding)) + \
            torch.matmul(drug_protein_norm, self.dv_layer(self.virus_embedding)) + \
            torch.matmul(drug_human_norm, self.dh_layer(self.human_embedding)),\
            self.drug_embedding], axis=1), self.W0)+self.b0),dim=1)
        # print(self.drug_representation)
        
        self.virus_representation  = F.normalize(F.relu(torch.matmul(
            torch.cat([torch.matmul(virus_virus_norm, self.vv_layer(self.virus_embedding)) + \
            torch.matmul(protein_drug_norm, self.vd_layer(self.drug_embedding)) + \
            torch.matmul(virus_human_norm, self.vh_layer(self.human_embedding)),\
            self.virus_embedding], axis=1), self.W0)+self.b0),dim=1)

        # print(virus_vector1.shape)
        self.human_representation = F.normalize(F.relu(torch.matmul(
            torch.cat([torch.matmul(human_human_norm, self.hh_layer(self.human_embedding)) + \
            torch.matmul(human_human_integration_norm, self.hhi_layer(self.human_embedding)) + \
            torch.matmul(human_drug_norm, self.hd_layer(self.drug_embedding)) + \
            torch.matmul(human_virus_norm, self.hv_layer(self.virus_embedding)),\
            self.human_embedding], axis=1), self.W0)+self.b0),dim=1)
        # print(human_vector1.shape)

        self.drug_drug_reconstruct = self.bi_layer(self.drug_representation,self.drug_representation, sym=True, dim_pred=512)
        self.drug_drug_reconstruct_loss = torch.sum(torch.multiply((self.drug_drug_reconstruct- drug_drug), (self.drug_drug_reconstruct-drug_drug)))

        self.virus_virus_reconstruct = self.bi_layer(self.virus_representation,self.virus_representation, sym=True, dim_pred=512)
        self.virus_virus_reconstruct_loss = torch.sum(torch.multiply((self.virus_virus_reconstruct- virus_virus), (self.virus_virus_reconstruct-virus_virus)))

        self.human_human_reconstruct = self.bi_layer(self.human_representation,self.human_representation, sym=True, dim_pred=512)
        self.human_human_reconstruct_loss = torch.sum(torch.multiply((self.human_human_reconstruct-human_human),(self.human_human_reconstruct-human_human)))

        self.human_human_in_reconstruct = self.bi_layer(self.human_representation,self.human_representation, sym=True, dim_pred=512)
        self.human_human_in_reconstruct_loss = torch.sum(torch.multiply((self.human_human_in_reconstruct-human_human_integration),(self.human_human_in_reconstruct-human_human_integration)))

        self.virus_human_reconstruct = self.bi_layer(self.virus_representation,self.human_representation,sym=False,dim_pred=512)
        self.virus_human_reconstruct_loss = torch.sum(torch.multiply((self.virus_human_reconstruct - virus_human),(self.virus_human_reconstruct - virus_human)))

        self.drug_human_reconstruct = self.bi_layer(self.drug_representation,self.human_representation,sym=False,dim_pred=512)
        self.drug_human_reconstruct_loss = torch.sum(torch.multiply((self.drug_human_reconstruct - drug_human),(self.drug_human_reconstruct - drug_human)))

        self.drug_protein_reconstruct = self.bi_layer(self.drug_representation,self.virus_representation, sym=False, dim_pred=512)
        tmp = torch.multiply(drug_protein_mask, (self.drug_protein_reconstruct-drug_protein))
        self.drug_protein_reconstruct_loss = torch.sum(torch.multiply(tmp, tmp))

        loss = self.drug_protein_reconstruct_loss + 0.1* (self.virus_virus_reconstruct_loss / virus_virus.shape[0] + \
            self.human_human_reconstruct_loss / human_human.shape[0] + self.drug_drug_reconstruct_loss / drug_drug.shape[0] + \
            self.human_human_in_reconstruct_loss / human_human.shape[0] + self.virus_human_reconstruct_loss / virus_human.shape[0]+\
            self.drug_human_reconstruct_loss / drug_human.shape[0])
        # print("Total Loss:")
        # print(loss.item())
        return loss,self.drug_protein_reconstruct