import os
import time
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import torch

from utils import newloss_generate_batch_data, get_loc2cat, get_unseen_user, get_adj
from Mymodel.hgat_newloss import Model

def settings(param=[]):
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('--path_in', type=str, default='./data/', help="input data path")
    parser.add_argument('--path_out', type=str, default='./results/', help="output data path")
    parser.add_argument('--data_name', type=str, default='NYC', help="data name")
    parser.add_argument('--cat_contained', action='store_false', default=True, help="whether contain category")
    parser.add_argument('--out_filename', type=str, default='', help="output data filename")
    # train params
    parser.add_argument('--gpu', type=str, default='1', help="GPU index to choose")
    parser.add_argument('--setting', type=int, default=0, help="0: full, 1: transductive, 2: inductive")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--run_num', type=int, default=1, help="run number")
    parser.add_argument('--epoch_num', type=int, default=80, help="epoch number")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--evaluate_step', type=int, default=1, help="evaluate step")
    parser.add_argument('--lam_c', type=float, default=1, help="category loss term factor")
    parser.add_argument('--lam_l', type=float, default=1, help="location loss term factor")
    parser.add_argument('--gt_w', type=float, default=0.8, help="gt weight")
    parser.add_argument('--sum_loss', action='store_true', default=False, help="sum loss, average loss else-wise")
    parser.add_argument('--schedule_w', action='store_true', default=False, help="schedule weight")
    # model params
    parser.add_argument('--res_factor', type=float, default=0.1, help="residual factor")
    parser.add_argument('--self_loop', action='store_true', default=False, help="add self loop")
    # embedding
    parser.add_argument('--u_dim', type=int, default=20, help="user embedding dimension")
    parser.add_argument('--l_dim', type=int, default=200, help="loc embedding dimension")
    parser.add_argument('--th_dim', type=int, default=20, help="time hour embedding dimension")
    parser.add_argument('--tw_dim', type=int, default=10, help="time week embedding dimension")
    parser.add_argument('--c_dim', type=int, default=200, help="category embedding dimension")
    # adj
    parser.add_argument('--dist', type=float, default=0.1, help="distance for l_adj, km")
    parser.add_argument('--weighted', action='store_true', default=False, help="whether use weighted adj")
    # l_gat
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of head attentions.')
    parser.add_argument('--nhid', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--gat_outdim', type=int, default=50, help="gat_outdim dimension")
    # rnn
    parser.add_argument('--rnn_type', type=str, default='gru', help="rnn type")
    parser.add_argument('--rnn_dim', type=int, default=600, help="rnn hidden dimension")
    parser.add_argument('--rnn_layer', type=int, default=1, help="rnn layer number")
    parser.add_argument('--dropout', type=float, default=0.1, help="drop out for rnn")
    # optimizer
    parser.add_argument('--step_size', type=int, default=1, help="drop out for rnn")
    parser.add_argument('--gamma', type=float, default=0.8, help="drop out for rnn")

    if __name__ == '__main__' and param == []:
        params = parser.parse_args()
    else:
        params = parser.parse_args(param)

    if not os.path.exists(params.path_out):
        os.mkdir(params.path_out)

    return params


def train(params, dataset):
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    # dataset info
    params.uid_size = len(dataset['uid_list'])
    params.pid_size = len(dataset['pid_dict'])
    params.cid_size = len(dataset['cid_dict']) if params.cat_contained else 0

    print(f'============Our Model============')
    print(f'params: \n{params}\n')

    pid_cid_dict = dataset['pid_cid_dict']

    # generate input data
    data_train, train_id = dataset['train_data'], dataset['train_id']
    data_test, test_id = dataset['test_data'], dataset['test_id']
    pid_lat_lon_radians = torch.tensor([[0, 0]] + list(dataset['pid_lat_lon_radians'].values()))
    pid_lat_lon = torch.tensor([[0, 0]] + list(dataset['pid_lat_lon'].values()))


    params.loc2cat, params.cat2loc, unseen_loc, test_mask_uid_sid, train_mask_uid_sid = \
        get_loc2cat(data_train, train_id, data_test, test_id, params.batch_size, params.pid_size, params.setting)
    mask_uid_sid = get_unseen_user(data_test, test_id, params.device, params.batch_size, params.cat_contained, loc_unseen=unseen_loc)
    # model and optimizer
    model = Model(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.8)
    scheduler = StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)

    # iterate epoch
    best_info_test = {'epoch': 0,
                      'R_Cat@1': 0, 'R_Cat@5': 0, 'R_Cat@10': 0, 'R_Cat@20': 0,
                      'R_Loc@1': 0, 'R_Loc@5': 0, 'R_Loc@10': 0, 'R_Loc@20': 0,
                      'N_Cat@1': 0, 'N_Cat@5': 0, 'N_Cat@10': 0, 'N_Cat@20': 0,
                      'N_Loc@1': 0, 'N_Loc@5': 0, 'N_Loc@10': 0, 'N_Loc@20': 0
                      }  # best metrics
    print('=' * 10, ' Training')


    l_adj, c_adj, lc_adj = get_adj(data_train, train_id, pid_lat_lon, pid_cid_dict, params)

    l_adj = torch.tensor(l_adj)
    l_adj = l_adj.to(params.device)
    c_adj = torch.tensor(c_adj)
    c_adj = c_adj.to(params.device)
    lc_adj = torch.tensor(lc_adj)
    lc_adj = lc_adj.to(params.device)

    l_adj_1c = torch.cat((l_adj, lc_adj), dim=1)
    l_adj_1c_down = F.pad(lc_adj.T, (0, c_adj.shape[0]))
    l_adj_1c = torch.cat((l_adj_1c, l_adj_1c_down), dim=0)

    l_pad = torch.zeros(l_adj.shape).to(l_adj.device)
    c_pad = torch.zeros(c_adj.shape).to(c_adj.device)
    c_h_adj_top = torch.cat((l_pad, lc_adj), dim=1)
    c_h_adj_down = torch.cat((lc_adj.T, c_pad), dim=1)
    c_h_adj = torch.cat((c_h_adj_top, c_h_adj_down), dim=0)

    cc_adj = torch.eye(c_adj.shape[0]).to(c_adj.device)
    cc_pad = torch.zeros(c_adj.shape).to(c_adj.device)
    cc_adj_top = torch.cat((c_adj, cc_adj), dim=1)
    cc_adj_down = torch.cat((cc_adj.T, cc_pad), dim=1)
    cc_adj = torch.cat((cc_adj_top, cc_adj_down), dim=0).to(c_adj.device)

    model = model.float()

    if params.schedule_w:
        th_epoch = (1 - params.gt_w) / params.epoch_num

    for i in range(params.epoch_num):

        if params.schedule_w:
            params.gt_w += th_epoch
            # print(f'gt_w: {params.gt_w}')
        epoch_start_time = time.time()
        loss_epoch, l_train_epoch_recall, l_test_epoch_recall, c_train_epoch_recall, c_test_epoch_recall, num_test_batch = \
            [], np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), 0
        l_train_epoch_ndcg, l_test_epoch_ndcg, c_train_epoch_ndcg, c_test_epoch_ndcg = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
        num_train_batch = 0

        model.train()
        for mask_batch, target_batch, data_batch, info_batch in newloss_generate_batch_data(
                data_train, train_id, params.device, params.batch_size,
                params.cat_contained, 'train', train_mask_uid_sid, test_mask_uid_sid, mask_uid_sid, params.setting):

            l_pred, c_pred = model(data_batch, mask_batch, l_adj, c_h_adj, cc_adj)

            l_loss = model.calculate_loss(l_pred, torch.squeeze(target_batch[0]), info_batch[0], info_batch[1], info_batch[2], params.gt_w)
            c_loss = model.calculate_loss(c_pred, torch.squeeze(target_batch[2]), info_batch[3], info_batch[4], info_batch[5], params.gt_w)
            loss = params.lam_l * l_loss + params.lam_c * c_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.item())
            l_train_batch_ndcg = model.calculate_ndcg(l_pred, torch.squeeze(target_batch[0]) - 1)
            l_train_batch_recall = model.calculate_recall(l_pred, torch.squeeze(target_batch[0]) - 1)
            c_train_batch_ndcg = model.calculate_ndcg(c_pred, torch.squeeze(target_batch[2]) - 1)
            c_train_batch_recall = model.calculate_recall(c_pred, torch.squeeze(target_batch[2]) - 1)
            l_train_epoch_recall += l_train_batch_recall
            l_train_epoch_ndcg += l_train_batch_ndcg
            c_train_epoch_recall += c_train_batch_recall
            c_train_epoch_ndcg += c_train_batch_ndcg

            num_train_batch += 1

        with torch.no_grad():
            model.eval()

            for mask_batch, target_batch, data_batch, unique_loc_batch in newloss_generate_batch_data(
                    data_test, test_id, params.device, params.batch_size,
                    params.cat_contained, 'test', train_mask_uid_sid, test_mask_uid_sid, mask_uid_sid, params.setting):
                # l_pred, c_pred, c_l_pred = model(data_batch, mask_batch, target_batch[1])
                l_pred, c_pred = model(data_batch, mask_batch, l_adj, c_h_adj, cc_adj)
                l_test_batch_ndcg  = model.calculate_ndcg(l_pred, torch.squeeze(target_batch[0])-1)
                l_test_batch_recall = model.calculate_recall(l_pred, torch.squeeze(target_batch[0])-1)
                c_test_batch_ndcg = model.calculate_ndcg(c_pred, torch.squeeze(target_batch[2])-1)
                c_test_batch_recall = model.calculate_recall(c_pred, torch.squeeze(target_batch[2])-1)
                l_test_epoch_recall += l_test_batch_recall
                l_test_epoch_ndcg += l_test_batch_ndcg
                c_test_epoch_recall += c_test_batch_recall
                c_test_epoch_ndcg += c_test_batch_ndcg

                num_test_batch += 1

        l_train_epoch_recall /= num_train_batch
        l_train_epoch_ndcg /= num_train_batch
        c_train_epoch_recall /= num_train_batch
        c_train_epoch_ndcg /= num_train_batch
        l_test_epoch_recall /= num_test_batch
        l_test_epoch_ndcg /= num_test_batch
        c_test_epoch_recall /= num_test_batch
        c_test_epoch_ndcg /= num_test_batch
        scheduler.step()

        print(f'Epoch {i+1} | loss: {np.mean(loss_epoch):.3f} | TRAIN |==| Recall | Cat: ['
              f'{c_train_epoch_recall[0]:.3f}; {c_train_epoch_recall[1]:.3f}; {c_train_epoch_recall[2]:.3f}; {c_train_epoch_recall[3]:.3f}] | Loc: ['
              f'{l_train_epoch_recall[0]:.3f}; {l_train_epoch_recall[1]:.3f}; {l_train_epoch_recall[2]:.3f}; {l_train_epoch_recall[3]:.3f}] |==| NDCG | Cat: ['
              f'{c_train_epoch_ndcg[0]:.3f}; {c_train_epoch_ndcg[1]:.3f}; {c_train_epoch_ndcg[2]:.3f}; {c_train_epoch_ndcg[3]:.3f}] | Loc: ['
              f'{l_train_epoch_ndcg[0]:.3f}; {l_train_epoch_ndcg[1]:.3f}; {l_train_epoch_ndcg[2]:.3f}; {l_train_epoch_ndcg[3]:.3f}]'
              f'\nEpoch {i+1} | loss: {np.mean(loss_epoch):.3f} |  TEST |==| Recall | Cat: ['
              f'{c_test_epoch_recall[0]:.3f}; {c_test_epoch_recall[1]:.3f}; {c_test_epoch_recall[2]:.3f}; {c_test_epoch_recall[3]:.3f}] | Loc: ['
              f'{l_test_epoch_recall[0]:.3f}; {l_test_epoch_recall[1]:.3f}; {l_test_epoch_recall[2]:.3f}; {l_test_epoch_recall[3]:.3f}] |==| NDCG | Cat: ['
              f'{c_test_epoch_ndcg[0]:.3f}; {c_test_epoch_ndcg[1]:.3f}; {c_test_epoch_ndcg[2]:.3f}; {c_test_epoch_ndcg[3]:.3f}] | Loc: ['
              f'{l_test_epoch_ndcg[0]:.3f}; {l_test_epoch_ndcg[1]:.3f}; {l_test_epoch_ndcg[2]:.3f}; {l_test_epoch_ndcg[3]:.3f}]'
              f' | Time: '
              f'{time.time()-epoch_start_time:.3f}\n')


        if params.setting == 2:
            if best_info_test["R_Loc@5"] < l_test_epoch_recall[1]:
                best_info_test["epoch"] = i + 1

                best_info_test["R_Cat@1"] = c_test_epoch_recall[0]
                best_info_test["R_Cat@5"] = c_test_epoch_recall[1]
                best_info_test["R_Cat@10"] = c_test_epoch_recall[2]
                best_info_test["R_Cat@20"] = c_test_epoch_recall[3]
                best_info_test["R_Loc@1"] = l_test_epoch_recall[0]
                best_info_test["R_Loc@5"] = l_test_epoch_recall[1]
                best_info_test["R_Loc@10"] = l_test_epoch_recall[2]
                best_info_test["R_Loc@20"] = l_test_epoch_recall[3]

                best_info_test["N_Cat@1"] = c_test_epoch_ndcg[0]
                best_info_test["N_Cat@5"] = c_test_epoch_ndcg[1]
                best_info_test["N_Cat@10"] = c_test_epoch_ndcg[2]
                best_info_test["N_Cat@20"] = c_test_epoch_ndcg[3]
                best_info_test["N_Loc@1"] = l_test_epoch_ndcg[0]
                best_info_test["N_Loc@5"] = l_test_epoch_ndcg[1]
                best_info_test["N_Loc@10"] = l_test_epoch_ndcg[2]
                best_info_test["N_Loc@20"] = l_test_epoch_ndcg[3]

        else:

            if best_info_test["R_Loc@1"] < l_test_epoch_recall[0]:
                best_info_test["epoch"] = i+1

                best_info_test["R_Cat@1"] = c_test_epoch_recall[0]
                best_info_test["R_Cat@5"] = c_test_epoch_recall[1]
                best_info_test["R_Cat@10"] = c_test_epoch_recall[2]
                best_info_test["R_Cat@20"] = c_test_epoch_recall[3]
                best_info_test["R_Loc@1"] = l_test_epoch_recall[0]
                best_info_test["R_Loc@5"] = l_test_epoch_recall[1]
                best_info_test["R_Loc@10"] = l_test_epoch_recall[2]
                best_info_test["R_Loc@20"] = l_test_epoch_recall[3]

                best_info_test["N_Cat@1"] = c_test_epoch_ndcg[0]
                best_info_test["N_Cat@5"] = c_test_epoch_ndcg[1]
                best_info_test["N_Cat@10"] = c_test_epoch_ndcg[2]
                best_info_test["N_Cat@20"] = c_test_epoch_ndcg[3]
                best_info_test["N_Loc@1"] = l_test_epoch_ndcg[0]
                best_info_test["N_Loc@5"] = l_test_epoch_ndcg[1]
                best_info_test["N_Loc@10"] = l_test_epoch_ndcg[2]
                best_info_test["N_Loc@20"] = l_test_epoch_ndcg[3]

    print(f'\nBest test: \nEpoch {best_info_test["epoch"]} | '
          f'Recall: \n'
          f'Cat: [{best_info_test["R_Cat@1"]:.3f}; {best_info_test["R_Cat@5"]:.3f}; {best_info_test["R_Cat@10"]:.3f}; {best_info_test["R_Cat@20"]:.3f}] | '
          f'Loc: [{best_info_test["R_Loc@1"]:.3f}; {best_info_test["R_Loc@5"]:.3f}; {best_info_test["R_Loc@10"]:.3f}; {best_info_test["R_Loc@20"]:.3f}]'
          f'\nNDCG: \n'
          f'Cat: [{best_info_test["N_Cat@1"]:.3f}; {best_info_test["N_Cat@5"]:.3f}; {best_info_test["N_Cat@10"]:.3f}; {best_info_test["N_Cat@20"]:.3f}] | '
          f'Loc: [{best_info_test["N_Loc@1"]:.3f}; {best_info_test["N_Loc@5"]:.3f}; {best_info_test["N_Loc@10"]:.3f}; {best_info_test["N_Loc@20"]:.3f}]')

    return best_info_test

if __name__ == '__main__':

    # print('=' * 20, ' Program Start')
    params = settings()
    params.device = torch.device(f"cuda:{params.gpu}")
    # print('Parameter is\n', params.__dict__)

    # file name to store
    FILE_NAME = [params.path_out, f'{time.strftime("%Y%m%d")}_{params.data_name}_']
    FILE_NAME[1] += f'{params.out_filename}'

    # Load data
    print('=' * 20, ' Loading data')
    start_time = time.time()
    if params.cat_contained:
        dataset = pickle.load(open(f'{params.path_in}{params.data_name}_cat.pkl', 'rb'))
    else:
        dataset = pickle.load(open(f'{params.path_in}{params.data_name}.pkl', 'rb'))
    print(f'Finished, time cost is {time.time() - start_time:.1f}')

    best_info_test = train(params, dataset)