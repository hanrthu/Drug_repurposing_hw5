import numpy as np
import torch
import argparse
import time
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
# from sklearn.cross_validation import train_test_split,StratifiedKFold
from NeoDTI import NeoDTI
from HNM import HNM

def row_normalize(a_matrix, substract_self_loop):
    # b_matrix = np.zeros((a_matrix.shape[0],a_matrix.shape[1]))
    b_matrix = np.copy(a_matrix)
    if substract_self_loop == True:
        np.fill_diagonal(b_matrix,0)
    b_matrix = b_matrix.astype(float)
    row_sums = b_matrix.sum(axis=1)+1e-12
    new_matrix = b_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix

def dataset_split(drug_virus):
    whole_positive_index = []
    whole_negative_index = []
    for i in range(drug_virus.shape[0]):
        for j in range(drug_virus.shape[1]):
            if int(drug_virus[i][j]) == 1:
                whole_positive_index.append([i,j])
            elif int(drug_virus[i][j]) == 0:
                whole_negative_index.append([i,j])
    # print(len(whole_positive_index))
    # print(len(whole_negative_index))
    index = np.arange(len(whole_positive_index))
    np.random.shuffle(index)
    train_index = index[:int(0.7*len(whole_positive_index))]
    valid_index = index[int(0.7*len(whole_positive_index)):int(0.8*len(whole_positive_index))]
    test_index = index[int(0.8*len(whole_positive_index)):]
    train_set = []
    valid_set = []
    test_set = []
    for i in train_index:
        train_set.append([whole_positive_index[i][0],whole_positive_index[i][1],1])
    for i in valid_index:
        valid_set.append([whole_positive_index[i][0],whole_positive_index[i][1],1])
    for i in test_index:
        test_set.append([whole_positive_index[i][0],whole_positive_index[i][1],1])
    if args.ratio == 1:
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=len(whole_positive_index),replace=False)
    elif args.ratio == 10:
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),size=10*len(whole_positive_index),replace=False)
    train_index = negative_sample_index[:int(0.7*len(negative_sample_index))]
    valid_index = negative_sample_index[int(0.7*len(negative_sample_index)):int(0.8*len(negative_sample_index))]
    test_index = negative_sample_index[int(0.8*len(negative_sample_index)):]
    for i in train_index:
        train_set.append([whole_negative_index[i][0],whole_negative_index[i][1],0])
    for i in valid_index:
        valid_set.append([whole_negative_index[i][0],whole_negative_index[i][1],0])
    for i in test_index:
        test_set.append([whole_negative_index[i][0],whole_negative_index[i][1],0])
    # print(negative_sample_index)
    # np.random.shuffle(negative_sample_index)
    np.random.shuffle(train_set)
    np.random.shuffle(valid_set)
    np.random.shuffle(test_set)
    # print(train_set)
    # print(valid_set)
    # print(test_set)
    return train_set,valid_set,test_set

def train_step(model,drug_protein,drug_protein_norm,drug_human,drug_human_norm,drug_drug,drug_drug_norm,human_human,human_human_norm,
    human_human_integration,human_human_integration_norm,human_drug,human_drug_norm,human_virus,human_virus_norm,virus_virus,virus_virus_norm,
    virus_human,virus_human_norm,protein_drug,protein_drug_norm,drug_protein_mask,optimizer): # Training Process
	
    model.train()
    loss, acc, auc, aupr = 0.0, 0.0, 0.0, 0.0
    optimizer.zero_grad()
    # print(drug_drug)
    loss_, acc_, auc_,aupr_,results = model(torch.from_numpy(drug_protein).to(device),torch.from_numpy(drug_protein_norm).to(device),
                                    torch.from_numpy(drug_human).to(device),torch.from_numpy(drug_human_norm).to(device),
                                    torch.from_numpy(drug_drug).to(device),torch.from_numpy(drug_drug_norm).to(device),
                                    torch.from_numpy(human_human).to(device),torch.from_numpy(human_human_norm).to(device),
                                    torch.from_numpy(human_human_integration).to(device),torch.from_numpy(human_human_integration_norm).to(device),
                                    torch.from_numpy(human_drug).to(device),torch.from_numpy(human_drug_norm).to(device),
                                    torch.from_numpy(human_virus).to(device),torch.from_numpy(human_virus_norm).to(device),
                                    torch.from_numpy(virus_virus).to(device),torch.from_numpy(virus_virus_norm).to(device),
                                    torch.from_numpy(virus_human).to(device),torch.from_numpy(virus_human_norm).to(device),
                                    torch.from_numpy(protein_drug).to(device),torch.from_numpy(protein_drug_norm).to(device),
                                    torch.from_numpy(drug_protein_mask).to(device))

    loss_.backward()
    optimizer.step()

    loss = loss_.cpu().data.numpy()
    # acc = acc_.cpu().data.numpy()
    # auc = auc_.cpu().data.numpy()
    # aupr = aupr_.cpu().data.numpy()
    acc = 0
    auc = 0
    aupr = 0
    return loss, acc, auc, aupr,results

def inference(model, X): # Test Process
    model.eval()
    pred_ = model(torch.from_numpy(X).to(device))
    return pred_.cpu().data.numpy()

if __name__ == '__main__':
    #参数输入和设备选取
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim',type=int,default=1024,help='Dim of embeddings')	
    parser.add_argument('--numsteps',type=int,default=2000,help='Num of epochs')
    parser.add_argument('--learning_rate',type=int,default=0.01,help='Learning rate')
    parser.add_argument('--ratio',type=int,default=1,help='Negative-Positive ratio')
    parser.add_argument('--model',type=str,default='NeoDTI',help='Choose a model')
    parser.add_argument('--alpha',type=float,default=0.9,help='HNM alpha')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: %s" %(device))
    # shape: Drug=6255, Virus Protein=404, Human Protein=2567
    # 读取网络并取对称
    network_path = "../DTI-data/"
    drug_virus = np.load(network_path + "VDTI_net.npy")
    # print(drug_virus.shape)
    drug_human = np.load(network_path +"HDTI_net.npy")
    # print(drug_human.shape)
    drug_drug = np.load(network_path +"Drug_simi_net.npy")
    # print(drug_drug.shape)
    human_human = np.load(network_path +"human.npy")
    # print(human_human.shape)
    virus_virus = np.load(network_path +"virusseq_add_ncov.npy")
    # print(virus_virus.shape)
    human_human_integration = np.load(network_path +"PPI_net.npy")
    # print(human_human.shape)
    virus_human = np.load(network_path +"VHI_net.npy")
    # print(virus_human.shape)

    human_virus = virus_human.T
    virus_drug = drug_virus.T
    human_drug = drug_human.T
    # print(drug_drug[0][0])
    #各个维度的设置
    num_drug = drug_virus.shape[0]
    num_virus = virus_human.shape[0]
    num_human = human_human.shape[0]

    #分别Normalize不同类型网络的权重
    #Drug
    drug_virus_norm = row_normalize(drug_virus,False)
    drug_human_norm = row_normalize(drug_human,False)
    drug_drug_norm = row_normalize(drug_drug,True)
    #Human
    human_human_norm = row_normalize(human_human,True)
    human_human_integration_norm = row_normalize(human_human_integration,True)
    human_drug_norm = row_normalize(human_drug,False)
    human_virus_norm = row_normalize(human_virus,False)
    #Virus
    virus_virus_norm = row_normalize(virus_virus,True)
    virus_human_norm = row_normalize(virus_human,False)
    virus_drug_norm = row_normalize(virus_drug,False)
    #Model的定义
    if args.model == 'NeoDTI':
        print('Using Model:\tNeoDTI')
        dim = args.dim
        model = NeoDTI(num_drug,num_human,num_virus,dim)
        model = model.double()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.model == 'HNM':
        print('Using Model:\tHNM')
        model = HNM(args.numsteps,args.alpha)
        model = model.double()
        model.to(device)

    #训练集和测试集的划分
    train_set,valid_set,test_set = dataset_split(drug_virus)

    #训练并测试
    best_valid_aupr = 0
    best_valid_auc = 0
    drug_protein = np.zeros((num_drug,num_virus))
    drug_protein_mask = np.zeros((num_drug,num_virus))
    for ele in train_set:
        drug_protein[ele[0],ele[1]] = ele[2]
        drug_protein_mask[ele[0],ele[1]] = 1
    protein_drug = drug_protein.T
    drug_protein_norm = row_normalize(drug_protein,False)
    protein_drug_norm = row_normalize(protein_drug,False)
    if args.model == 'NeoDTI':
        for step in range(args.numsteps):
            start = time.time()
            # print("Hello World!")
            # print(drug_drug)
            train_loss,train_acc,train_auc,train_aupr,results= train_step(
                                                        model,
                                                        drug_protein,drug_protein_norm,
                                                        drug_human,drug_human_norm,
                                                        drug_drug,drug_drug_norm,
                                                        human_human,human_human_norm,
                                                        human_human_integration,human_human_integration_norm,
                                                        human_drug,human_drug_norm,
                                                        human_virus,human_virus_norm,
                                                        virus_virus,virus_virus_norm,
                                                        virus_human,virus_human_norm,
                                                        protein_drug,protein_drug_norm,
                                                        drug_protein_mask,
                                                        optimizer)
            train_list = []
            train_truth = []
            for ele in train_set:
                train_list.append(results[ele[0],ele[1]])
                train_truth.append(ele[2])
            train_auc = roc_auc_score(train_truth,train_list)
            train_aupr = average_precision_score(train_truth, train_list)
            print('Train auc aupr', train_auc,train_aupr)

            if step % 1 == 0:
                print('Step',step,'Total loss',train_loss)
                pred_list = []
                ground_truth = []
                for ele in valid_set:
                    pred_list.append(results[ele[0],ele[1]])
                    ground_truth.append(ele[2])
                valid_auc = roc_auc_score(ground_truth, pred_list)
                valid_aupr = average_precision_score(ground_truth, pred_list)
                # test_auc = 0
                # test_aupr = 0
                if valid_aupr >= best_valid_aupr:
                    best_valid_aupr = valid_aupr
                    best_valid_auc = valid_auc
                    pred_list = []
                    ground_truth = []
                    for ele in test_set:
                        pred_list.append(results[ele[0],ele[1]])
                        ground_truth.append(ele[2])
                    test_auc = roc_auc_score(ground_truth, pred_list)
                    test_aupr = average_precision_score(ground_truth, pred_list)
                print('Valid auc aupr,', valid_auc, valid_aupr)
                print('Test auc aupr', test_auc, test_aupr)
    if args.model == 'HNM':
        results = model(torch.from_numpy(drug_protein).to(device),torch.from_numpy(drug_protein_norm).to(device),
                                torch.from_numpy(drug_human).to(device),torch.from_numpy(drug_human_norm).to(device),
                                torch.from_numpy(drug_drug).to(device),torch.from_numpy(drug_drug_norm).to(device),
                                torch.from_numpy(human_human).to(device),torch.from_numpy(human_human_norm).to(device),
                                torch.from_numpy(human_human_integration).to(device),torch.from_numpy(human_human_integration_norm).to(device),
                                torch.from_numpy(human_drug).to(device),torch.from_numpy(human_drug_norm).to(device),
                                torch.from_numpy(human_virus).to(device),torch.from_numpy(human_virus_norm).to(device),
                                torch.from_numpy(virus_virus).to(device),torch.from_numpy(virus_virus_norm).to(device),
                                torch.from_numpy(virus_human).to(device),torch.from_numpy(virus_human_norm).to(device),
                                torch.from_numpy(protein_drug).to(device),torch.from_numpy(protein_drug_norm).to(device),
                                torch.from_numpy(drug_protein_mask).to(device))
        train_list = []
        train_truth = []
        for ele in train_set:
            train_list.append(results[ele[0],ele[1]])
            train_truth.append(ele[2])
        train_auc = roc_auc_score(train_truth,train_list)
        train_aupr = average_precision_score(train_truth, train_list)
        print('Train auc aupr', train_auc,train_aupr)
        pred_list = []
        ground_truth = []
        for ele in valid_set:
            pred_list.append(results[ele[0],ele[1]])
            ground_truth.append(ele[2])
        valid_auc = roc_auc_score(ground_truth, pred_list)
        valid_aupr = average_precision_score(ground_truth, pred_list)
        # test_auc = 0
        # test_aupr = 0
        if valid_aupr >= best_valid_aupr:
            best_valid_aupr = valid_aupr
            best_valid_auc = valid_auc
            pred_list = []
            ground_truth = []
            for ele in test_set:
                pred_list.append(results[ele[0],ele[1]])
                ground_truth.append(ele[2])
            test_auc = roc_auc_score(ground_truth, pred_list)
            test_aupr = average_precision_score(ground_truth, pred_list)
        print('Valid auc aupr,', valid_auc, valid_aupr)
        print('Test auc aupr', test_auc, test_aupr)


