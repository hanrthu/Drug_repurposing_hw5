import torch
import torch.nn as nn

class HNM(nn.Module):
    def __init__(self, steps,alpha):
        super(HNM, self).__init__()
        self.steps = steps
        self.alpha = alpha

    def forward(self,drug_protein,drug_protein_norm,drug_human,drug_human_norm,drug_drug,drug_drug_norm,human_human,human_human_norm,
    human_human_integration,human_human_integration_norm,human_drug,human_drug_norm,human_virus,human_virus_norm,virus_virus,virus_virus_norm,
    virus_human,virus_human_norm,protein_drug,protein_drug_norm,drug_protein_mask,W_hd,W_dv):
    
        W_hd = W_hd.clone()
        W_dd = drug_drug_norm
        W_dv = W_dv.clone()
        W_vv = virus_virus_norm
        W_hh = human_human_norm

        W_hd_new = self.alpha * torch.matmul(torch.matmul(torch.matmul(torch.matmul(W_hd,W_dd),W_dv),W_vv),W_dv.T) + (1-self.alpha) * human_drug_norm
        W_dv_new = self.alpha * torch.matmul(torch.matmul(torch.matmul(torch.matmul(W_hd.T,W_hh),W_hd),W_dd),W_dv) + (1-self.alpha) * drug_protein_norm * drug_protein_mask
        tmp = torch.multiply(drug_protein_mask, (W_dv-drug_protein_norm))
        train_loss = torch.sum(torch.multiply(tmp, tmp))
        return train_loss,W_hd_new,W_dv_new
            
