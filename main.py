import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
# from utils import load_data, _collate
# from utils import EarlyStopping, set_random_seed
from utils import *
from models.model import Model
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import fitlog
import sys
import warnings
from collections import Counter
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
warnings.filterwarnings("ignore")

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=134,help='seed')
    parser.add_argument('--batch_size', type=int, default=1024,help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--dropout_ratio', type=float, default=0.2, help='mlp dropout ratio')
    parser.add_argument('--epochs', type=int, default=300,help='maximum number of epochs')
    parser.add_argument('--mode', type=str, default='train',help='train or test mode (default: train)') 
    parser.add_argument('--num_workers', type=int, default=0,help='num_workers of dataloader')  
    parser.add_argument('--patience', type=int, default=10, help='patience for earlystopping')
    parser.add_argument("--model_path",type=str,default=None,help="load trained model params ")
    parser.add_argument('--setup', type=str, default='known', help='experimental setup (default:known) ')
    parser.add_argument("--classify", action='store_true', 
                        default=False, help='classify expreiment (default: regression)')
    # drug
    parser.add_argument("--drug_gnn",type=str,default="CMPNN")
    parser.add_argument('--drug_layer', type=int, default=5, help='number of layers for drug gnn')
    parser.add_argument('--drug_hidden_dim', type=int, default=150, help='hidden dim for drug gnn')
    parser.add_argument('--drug_output_dim', type=int, default=256, help='output dim for drug gnn')
    # cell
    parser.add_argument('--cell_output_dim', type=int, default=256, help='output dim for vnn')
    parser.add_argument('--vnn_dropout_ratio', type=float, default=0.1, help='vnn dropout ratio')
    parser.add_argument('--vnn_mode',type=str, default='ref',help='ref or random or full mode (default: ref)')
    parser.add_argument('--use_average_neuron_n',action='store_true', default=False,
                        help='full connection use average neuron number')
    parser.add_argument('--child_neuron_ratio',type=int, default=1,help='')
    parser.add_argument("--only_combine_child_gene_group", action='store_true', 
                        default=False, help='only combine child gene group')
    return parser.parse_args()


def main():
    args = arg_parse()
    set_random_seed(args.seed)
    print(args)
    torch.multiprocessing.set_start_method('spawn')
    '''
    Load data
    '''
    with open("./data/Drug/drug_feat_idx_full.pkl",'rb') as f:
        drug_dict = pickle.load(f)  
    with open('./data/CCLE/cell_feat_full_clean.pkl', 'rb') as f:
        cell_dict = pickle.load(f)  
    drug_names, cell_cosmic = list(drug_dict.keys()),list(int(cell) for cell in cell_dict.keys()) 
    IC50_thred_file="./data/IC50_thred.txt"
    thred_name = [item.strip().split('(')[0].strip() for item in open(IC50_thred_file).readlines()[0].split('\t')]
    IC50_thred = [float(item.strip()) for item in open(IC50_thred_file).readlines()[1].split('\t')]
    name2thred=dict()
    for a,b in zip(thred_name,IC50_thred):                                                                                                                                 
        name2thred[a]=b
    '''
    Preprocess IC50
    '''
    IC = pd.read_csv('./data/GDSC2/GDSC2_fitted_dose_response_24Jul22.csv')   
    Del=[]
    if args.classify:
        processed_IC50_file="./data/GDSC2/processed_IC50_classify.csv"
        for index,row in IC.iterrows():
            if row['DRUG_ID'] not in drug_names or row['COSMIC_ID'] not in cell_cosmic or row["DRUG_NAME"] not in thred_name: 
                Del.append(index)
    else:
        processed_IC50_file="./data/GDSC2/processed_IC50.csv"
        for index,row in IC.iterrows():
            if row['DRUG_ID'] not in drug_names or row['COSMIC_ID'] not in cell_cosmic: 
                Del.append(index)  
    IC=IC.drop(Del)
    IC.reset_index(drop=True, inplace=True)
    IC.to_csv(processed_IC50_file)

    id2name=dict()
    for index,row in IC.iterrows():
        id2name[row["DRUG_ID"]]=row["DRUG_NAME"]
    
    drugid2thred={}
    for id,name in id2name.items():
        if name in name2thred.keys():
            drugid2thred[id]=name2thred[name]
    '''
    Prepare 
    '''    
    train_loader, val_loader, test_loader = load_data(IC, drug_dict, cell_dict,drugid2thred,args) 
    print('all IC50: %d, train: %d, val: %d, test: %d' % (len(IC), len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
    args.num_feature = cell_dict[683667].shape[1] 
    '''
    Train or Test
    '''
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = Model(args).to(device)  
    norm_factor = [0, 1]
    if args.mode == 'train':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5) 
        log_folder = os.path.join(os.getcwd(), "logs", model._get_name())
       
        if not os.path.exists(log_folder):
                os.makedirs(log_folder)
        fitlog.set_log_dir(log_folder)
        fitlog.add_hyper(args)
        fitlog.add_hyper_in_file(__file__)

        if args.classify:
            criterion = nn.BCELoss()
            stopper = EarlyStopping(mode="higher", patience=args.patience)

            for epoch in range(1, args.epochs + 1):
                print("=====Epoch {}".format(epoch))
                print("Training...")
                train_loss = train_classify(model, train_loader, criterion, opt, scheduler, norm_factor, device)
                fitlog.add_loss(train_loss.item(), name='Train bce', step=epoch)
                print("Evaluating...")
                val_metric,val_pred,val_true = validate_classify(model, val_loader, norm_factor, device)
                test_metric,test_pred,test_true = validate_classify(model, test_loader, norm_factor, device)#去掉?
                print("Train BCE Loss: %.4f  Validation AUROC:%.4f AUPR:%.4f Test AUROC:%.4f AUPR:%.4f " % (train_loss,val_metric["AUROC"],val_metric["AUPR"],test_metric["AUROC"],test_metric["AUPR"]))
                fitlog.add_metric({'Validate': {'AUROC': val_metric["AUROC"], 'AUPR': val_metric["AUPR"]}}, step=epoch)
                early_stop = stopper.step(val_metric["AUROC"], model)
                if early_stop:
                    break
            print('EarlyStopping! Finish training!')
            print('Testing...')
            stopper.load_checkpoint(model)
            train_metric,train_pred,train_true = validate_classify(model, train_loader, norm_factor, device)
            val_metric,val_pred,val_true = validate_classify(model, val_loader, norm_factor, device)
            test_metric,test_pred,test_true = validate_classify(model, test_loader, norm_factor, device)
        else:
            criterion = nn.MSELoss()
            stopper = EarlyStopping(mode="lower", patience=args.patience)

            for epoch in range(1, args.epochs + 1):
                print("=====Epoch {}".format(epoch))
                print("Training...")
                train_loss = train(model, train_loader, criterion, opt, scheduler, norm_factor, device)
                fitlog.add_loss(train_loss.item(), name='Train mse', step=epoch)
                print("Evaluating...")
                val_metric,val_pred,val_true = validate(model, val_loader, norm_factor, device)
                test_metric,test_pred,test_true = validate(model, test_loader, norm_factor, device)#去掉?
                print("Train MSE Loss: %.4f  Validation RMSE:%.4f MAE:%.4f PCC:%.4f Test RMSE:%.4f MAE:%.4f PCC:%.4f" % (train_loss, val_metric["RMSE"], val_metric["MAE"], val_metric["PCC"],
                                                                                                                        test_metric["RMSE"], test_metric["MAE"], test_metric["PCC"]))
                fitlog.add_metric({'Validate': {'RMSE':val_metric["RMSE"], 'MAE': val_metric["MAE"], 'PCC': val_metric["PCC"]}}, step=epoch)
                early_stop = stopper.step(val_metric["RMSE"], model)
                if early_stop:
                    break
            print('EarlyStopping! Finish training!')
            print('Testing...')
            stopper.load_checkpoint(model)
            train_metric,train_pred,train_true = validate(model, train_loader, norm_factor, device)
            val_metric,val_pred,val_true = validate(model, val_loader, norm_factor, device)
            test_metric,test_pred,test_true = validate(model, test_loader, norm_factor, device)

        print('Test reslut:',test_metric)
        fitlog.add_best_metric(
            {'epoch': epoch - args.patience,
             "train": train_metric,
             "valid": val_metric,
             "test": test_metric})
    
    elif args.mode == 'test':
        if args.model_path:
            print("loading trained model params...")
            model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        else:
            print("Please set --model_path")
            return 
        test_metric,test_pred,test_true = validate(model, test_loader, norm_factor, device)
        print('Test RMSE: {}, MAE: {}, R2: {}, PCC: {} SPCC: {} '.format(round(test_metric["RMSE"], 4), round(test_metric["MAE"], 4),
                                                             round(test_metric["R2"], 4), round(test_metric["PCC"], 4),round(test_metric["SPCC"],4)))
            
if __name__ == "__main__":
    main()
