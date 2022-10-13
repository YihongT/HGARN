import os
import mpu
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics.pairwise import haversine_distances
from math import radians, cos, sin, asin, sqrt

def get_adj(data_input, data_id, pid_lat_lon, pid_cid_dict, params):

    cur_dir = os.getcwd()
    if not os.path.exists(cur_dir + f'/data/dis_matrix_{params.data_name}.npy'):
        print(f'Distance matrix for {params.data_name} not exist, start computing...')
        start = time.time()
        dist_matrix = []
        for i in range(pid_lat_lon.shape[0]):
            dist_row = []
            for j in range(pid_lat_lon.shape[0]):
                # dist_row.append(haversine(pid_lat_lon[i][1], pid_lat_lon[i][0], pid_lat_lon[j][1], pid_lat_lon[j][0]))
                dist_row.append(mpu.haversine_distance((pid_lat_lon[i][0], pid_lat_lon[i][1]), (pid_lat_lon[j][0], pid_lat_lon[j][1])))
            dist_matrix.append(dist_row)
        dist_matrix = np.array(dist_matrix)
        np.save(cur_dir + f'/data/dis_matrix_{params.data_name}.npy', dist_matrix)
        end = time.time()
        print(f'time to compute distance matrix: {end-start}')
    else:
        print(f'Distance matrix for {params.data_name} exist, start Loading...')
        dist_matrix = np.load(cur_dir + f'/data/dis_matrix_{params.data_name}.npy')

    dist_matrix = dist_matrix[1:, 1:]

    print(f'dist_matrix: {dist_matrix.shape}')

    dist_matrix += np.eye(dist_matrix.shape[0]) * 11

    l_adj = (dist_matrix < params.dist) * (dist_matrix > 0.001) * 1

    data = {}
    data_queue = list()
    uid_list = data_id.keys()
    for uid in uid_list:
        data_queue.append((uid, max(data_id[uid])))

    count_unique = []
    c_pastexist, l_pastexist = 0, 0

    for i in range(len(data_queue)):
        uid, sid = data_queue[i]
        curr_input = copy.deepcopy(data_input[uid][sid])

        curr_input['loc'][0].extend(curr_input['loc'][1][:])
        curr_input['tim'][0].extend(curr_input['tim'][1][:])
        curr_input['cat'][0].extend(curr_input['cat'][1][:])
        curr_input['loc'] = [curr_input['loc'][0]]
        curr_input['tim'] = [curr_input['tim'][0]]
        curr_input['cat'] = [curr_input['cat'][0]]
        curr_input['target_l'] = [curr_input['target_l'][-1]]
        curr_input['target_c'] = [curr_input['target_c'][-1]]
        curr_input['target_th'] = [curr_input['target_th'][-1]]


        uid_data = []
        uid_l_data = []
        uid_t_data = []
        uid_c_data = []

        uid_l_data.extend(curr_input["loc"][0])
        uid_l_data.extend(curr_input["target_l"])
        uid_c_data.extend(curr_input["cat"][0])
        uid_c_data.extend(curr_input["target_c"])

        for t in curr_input["tim"][0]:
            uid_t_data.append(t[1])

        uid_t_data.extend(curr_input["target_th"])

        uid_data.append(uid_t_data)
        uid_data.append(uid_c_data)
        uid_data.append(uid_l_data)


        data[uid] = uid_data


    c_adj = np.zeros((params.cid_size, params.cid_size))
    lc_adj = np.zeros((params.pid_size, params.cid_size))
    if params.weighted:
        lc_adj_weight = np.zeros((params.pid_size, params.cid_size))

    for key in pid_cid_dict.keys():
        lc_adj[key-1][pid_cid_dict[key]-1] = 1


    for uid in range(1, len(data)+1):
        u_data = data[uid]

        # print(f'u_data: \n{u_data}')

        u_t_data = np.array(u_data[0])
        u_c_data = np.array(u_data[1]) - 1
        u_l_data = np.array(u_data[2]) - 1

        if params.weighted:
            for i in range(len(u_c_data)):
                lc_adj_weight[u_l_data[i]][u_c_data[i]] += 1

        for t in np.unique(u_t_data):

            cats = np.unique(u_c_data[np.where(u_t_data == t)[0]])

            for i in range(len(cats)):
                for j in range(i+1, len(cats)):
                    # print(f'i, j: {i, j}')
                    c_adj[cats[i]][cats[j]] += 1
                    c_adj[cats[j]][cats[i]] += 1

    if params.weighted:
        lc_adj_weight = np.squeeze(np.array([[lc_adj_weight[i] == max(np.max(lc_adj_weight, axis=1)[i], 1e-5)] for i in range(lc_adj_weight.shape[0])]), axis=1)
        for i in range(lc_adj_weight.shape[0]):
            first_true = np.where(lc_adj_weight[i] == True)[0]
            # print(f'first: {first_true}')
            if len(first_true) > 1:
                # print(f'first: {first_true}')
                # print(f'lc: {lc_adj[i]}')
                lc_adj_weight[i][first_true[-1]] = False

        lc_adj_weight = lc_adj_weight * 1

    c_adj = (c_adj >= np.mean(c_adj)) * 1


    return l_adj, c_adj, lc_adj


def get_unseen_user(data_input, data_id, device, batch_size, cat_contained, loc_unseen):
    data_queue = list()
    uid_list = data_id.keys()
    for uid in uid_list:
        for sid in data_id[uid]:
            data_queue.append((uid, sid))

    # generate batch data
    data_len = len(data_queue)
    batch_num = int(data_len / batch_size)
    mask_uid = set()

    for i in range(batch_num):
        batch_idx_list = np.random.choice(data_len, batch_size, replace=False)
        for batch_idx in batch_idx_list:
            uid, sid = data_queue[batch_idx]
            curr_input = copy.deepcopy(data_input[uid][sid])
            curr_input['loc'][0].extend(curr_input['loc'][1][:])
            curr_input['tim'][0].extend(curr_input['tim'][1][:])
            curr_input['cat'][0].extend(curr_input['cat'][1][:])
            curr_input['loc'] = [curr_input['loc'][0]]
            curr_input['tim'] = [curr_input['tim'][0]]
            curr_input['cat'] = [curr_input['cat'][0]]
            curr_input['target_l'] = [curr_input['target_l'][-1]]
            curr_input['target_c'] = [curr_input['target_c'][-1]]
            curr_input['target_th'] = [curr_input['target_th'][-1]]


            for loc in list(loc_unseen):
                if loc in curr_input['loc'][0] or loc in curr_input['target_l']:
                    mask_uid.add((uid, sid))


    return list(mask_uid)

def get_loc2cat(data_train, train_id, data_test, test_id, batch_size, pid_size, setting):
    '''generate batch data'''

    data_queue = list()
    uid_list = train_id.keys()
    for uid in uid_list:
        for sid in train_id[uid]:
            data_queue.append((uid, sid))

    # generate batch data
    data_len = len(data_queue)
    batch_num = int(data_len / batch_size)
    loc2cat, cat2loc = {}, {}
    train_mask_list = set()

    for i in range(batch_num):
        batch_idx_list = np.random.choice(data_len, batch_size, replace=False)
        for batch_idx in batch_idx_list:
            uid, sid = data_queue[batch_idx]
            curr_input = copy.deepcopy(data_train[uid][sid])
            curr_input['loc'][0].extend(curr_input['loc'][1][:])
            curr_input['tim'][0].extend(curr_input['tim'][1][:])
            curr_input['cat'][0].extend(curr_input['cat'][1][:])
            curr_input['loc'] = [curr_input['loc'][0]]
            curr_input['tim'] = [curr_input['tim'][0]]
            curr_input['cat'] = [curr_input['cat'][0]]
            curr_input['target_l'] = [curr_input['target_l'][-1]]
            curr_input['target_c'] = [curr_input['target_c'][-1]]
            curr_input['target_th'] = [curr_input['target_th'][-1]]

            for idx in range(len(curr_input['loc'][0])):
                if curr_input['loc'][0][idx] not in loc2cat:
                    loc2cat[curr_input['loc'][0][idx]] = curr_input['cat'][0][idx]
                if curr_input['cat'][0][idx] not in cat2loc:
                    cat2loc[curr_input['cat'][0][idx]] = curr_input['loc'][0][idx]


            if setting == 1:
                if curr_input['target_l'][0] not in curr_input['loc'][0] or curr_input['target_c'][0] not in curr_input['cat'][0]:
                    train_mask_list.add((uid, sid))
            elif setting == 2:
                if curr_input['target_l'][0] not in curr_input['loc'][0] and curr_input['target_c'][0] not in curr_input['cat'][0]:
                    train_mask_list.add((uid, sid))


    num = set([i for i in range(1, pid_size+1)])

    unseen_loc = set.difference(num, set(loc2cat.keys()))

    data_queue = list()
    uid_list = test_id.keys()
    for uid in uid_list:
        for sid in test_id[uid]:
            data_queue.append((uid, sid))

    # generate batch data
    data_len = len(data_queue)
    batch_num = int(data_len / batch_size)
    test_mask_list = set()

    for i in range(batch_num):
        batch_idx_list = np.random.choice(data_len, batch_size, replace=False)
        for batch_idx in batch_idx_list:
            uid, sid = data_queue[batch_idx]
            curr_input = copy.deepcopy(data_test[uid][sid])
            curr_input['loc'][0].extend(curr_input['loc'][1][:])
            curr_input['tim'][0].extend(curr_input['tim'][1][:])
            curr_input['cat'][0].extend(curr_input['cat'][1][:])
            curr_input['loc'] = [curr_input['loc'][0]]
            curr_input['tim'] = [curr_input['tim'][0]]
            curr_input['cat'] = [curr_input['cat'][0]]
            curr_input['target_l'] = [curr_input['target_l'][-1]]
            curr_input['target_c'] = [curr_input['target_c'][-1]]
            curr_input['target_th'] = [curr_input['target_th'][-1]]

            # print(f'curr: \n{curr_input}')
            if setting == 1:
                if curr_input['target_l'][0] not in curr_input['loc'][0] or curr_input['target_c'][0] not in curr_input['cat'][0]:
                    test_mask_list.add((uid, sid))
            elif setting == 2:
                if curr_input['target_l'][0] not in curr_input['loc'][0] and curr_input['target_c'][0] not in curr_input['cat'][0]:
                    test_mask_list.add((uid, sid))


    return loc2cat, cat2loc, unseen_loc, list(test_mask_list), list(train_mask_list)



def new_generate_batch_data(data_input, data_id, device, batch_size, cat_contained, type,
                            train_mask_uid_sid, test_mask_uid_sid, mask_uid_sid, setting):
    '''generate batch data'''

    # data_id: {uid: [sid, ...], uid: [sid, ...], ...}

    # generate (uid, sid) queue
    data_queue = list()
    uid_list = data_id.keys()

    for uid in uid_list:
        for sid in data_id[uid]:
            if (uid, sid) not in mask_uid_sid:
                if setting == 0:
                    data_queue.append((uid, sid))
                elif setting == 1:
                    if type == 'test':
                        if (uid, sid) in test_mask_uid_sid:
                            continue
                        else:
                            data_queue.append((uid, sid))
                    elif type == 'train':
                        if (uid, sid) in train_mask_uid_sid:
                            continue
                        else:
                            data_queue.append((uid, sid))
                elif setting == 2:
                    if type == 'test':
                        if (uid, sid) in test_mask_uid_sid:
                            data_queue.append((uid, sid))
                        else:
                            continue
                    elif type == 'train':
                        if (uid, sid) in train_mask_uid_sid:
                            data_queue.append((uid, sid))
                        else:
                            continue
                else:
                    assert 1 == 2
            else:
                continue

    # generate batch data
    data_len = len(data_queue)
    batch_num = int(data_len / batch_size)


    # iterate batch number times
    for i in range(batch_num):
        # print(f'i: {i}')
        # batch data
        uid_batch = []
        loc_his_batch = []
        tim_w_his_batch = []
        tim_h_his_batch = []
        target_l_batch = []
        target_c_batch = []
        target_th_batch = []
        target_len_batch = []
        history_len_batch = []
        current_len_batch = []
        if cat_contained:
            cat_his_batch = []


        batch_idx_list = np.random.choice(data_len, batch_size, replace=False)
        # iterate batch index

        # count = 0
        for batch_idx in batch_idx_list:

            uid, sid = data_queue[batch_idx]
            uid_batch.append([uid])
            curr_input = copy.deepcopy(data_input[uid][sid])

            curr_input['loc'][0].extend(curr_input['loc'][1][:])
            curr_input['tim'][0].extend(curr_input['tim'][1][:])
            curr_input['cat'][0].extend(curr_input['cat'][1][:])
            curr_input['loc'] = [curr_input['loc'][0]]
            curr_input['tim'] = [curr_input['tim'][0]]
            curr_input['cat'] = [curr_input['cat'][0]]
            curr_input['target_l'] = [curr_input['target_l'][-1]]
            curr_input['target_c'] = [curr_input['target_c'][-1]]
            curr_input['target_th'] = [curr_input['target_th'][-1]]

            # history
            loc_his_batch.append(torch.LongTensor(curr_input['loc'][0]))
            tim_his_ts = torch.LongTensor(curr_input['tim'][0])
            tim_w_his_batch.append(tim_his_ts[:, 0])
            tim_h_his_batch.append(tim_his_ts[:, 1])
            history_len_batch.append(tim_his_ts.shape[0])
            # target

            target_l = torch.LongTensor(curr_input['target_l'])
            target_l_batch.append(target_l)
            target_len_batch.append(target_l.shape[0])
            target_th_batch.append(torch.LongTensor(curr_input['target_th']))
            # catrgory
            if cat_contained:
                cat_his_batch.append(torch.LongTensor(curr_input['cat'][0]))
                target_c_batch.append(torch.LongTensor(curr_input['target_c']))

        # padding
        uid_batch_tensor = torch.LongTensor(uid_batch).to(device)
        # history
        loc_his_batch_pad = pad_sequence(loc_his_batch, batch_first=True).to(device)
        tim_w_his_batch_pad = pad_sequence(tim_w_his_batch, batch_first=True).to(device)
        tim_h_his_batch_pad = pad_sequence(tim_h_his_batch, batch_first=True).to(device)
        # target
        target_l_batch_pad = pad_sequence(target_l_batch, batch_first=True).to(device)
        target_th_batch_pad = pad_sequence(target_th_batch, batch_first=True).to(device)

        if cat_contained:
            cat_his_batch_pad = pad_sequence(cat_his_batch, batch_first=True).to(device)
            target_c_batch_pad = pad_sequence(target_c_batch, batch_first=True).to(device)
            yield (target_len_batch, history_len_batch), \
                  (target_l_batch_pad, target_th_batch_pad, target_c_batch_pad), \
                  (uid_batch_tensor, loc_his_batch_pad, tim_w_his_batch_pad, tim_h_his_batch_pad, cat_his_batch_pad)
        else:
            yield (target_len_batch, history_len_batch), \
                  (target_l_batch_pad, target_th_batch_pad), \
                  (uid_batch_tensor, loc_his_batch_pad, tim_w_his_batch_pad, tim_h_his_batch_pad)

def generate_batch_data(data_input, data_id, device, batch_size, cat_contained, type,
                        train_mask_uid_sid, test_mask_uid_sid, mask_uid_sid, setting):
    '''generate batch data'''

    # generate (uid, sid) queue
    data_queue = list()
    uid_list = data_id.keys()

    for uid in uid_list:
        for sid in data_id[uid]:
            if (uid, sid) not in mask_uid_sid:
                if setting == 0:
                    data_queue.append((uid, sid))
                elif setting == 1:
                    if type == 'test':
                        if (uid, sid) in test_mask_uid_sid:
                            continue
                        else:
                            data_queue.append((uid, sid))
                    elif type == 'train':
                        if (uid, sid) in train_mask_uid_sid:
                            continue
                        else:
                            data_queue.append((uid, sid))
                elif setting == 2:
                    if type == 'test':
                        if (uid, sid) in test_mask_uid_sid:
                            data_queue.append((uid, sid))
                        else:
                            continue
                    elif type == 'train':
                        if (uid, sid) in train_mask_uid_sid:
                            data_queue.append((uid, sid))
                        else:
                            continue
                else:
                    assert 1 == 2
            else:
                continue


    # generate batch data
    data_len = len(data_queue)
    batch_num = int(data_len/batch_size)

    # iterate batch number times
    for i in range(batch_num):
        # print(f'i: {i}')
        # batch data
        uid_batch = []
        loc_cur_batch = []
        tim_w_cur_batch = []
        tim_h_cur_batch = []
        loc_his_batch = []
        tim_w_his_batch = []
        tim_h_his_batch = []
        target_l_batch = []
        target_c_batch = []
        target_th_batch = []
        target_len_batch = []
        history_len_batch = []
        current_len_batch = []
        if cat_contained:
            cat_cur_batch = []
            cat_his_batch = []


        batch_idx_list = np.random.choice(data_len, batch_size, replace=False)
        # iterate batch index
        for batch_idx in batch_idx_list:
            uid, sid = data_queue[batch_idx]
            uid_batch.append([uid])

            data_input[uid][sid]['loc'][0].extend(data_input[uid][sid]['loc'][1][:-1])
            data_input[uid][sid]['tim'][0].extend(data_input[uid][sid]['tim'][1][:-1])
            data_input[uid][sid]['cat'][0].extend(data_input[uid][sid]['cat'][1][:-1])
            data_input[uid][sid]['loc'][1] = [data_input[uid][sid]['loc'][1][-1]]
            data_input[uid][sid]['tim'][1] = [data_input[uid][sid]['tim'][1][-1]]
            data_input[uid][sid]['cat'][1] = [data_input[uid][sid]['cat'][1][-1]]
            data_input[uid][sid]['target_l'] = [data_input[uid][sid]['target_l'][-1]]
            data_input[uid][sid]['target_c'] = [data_input[uid][sid]['target_c'][-1]]
            data_input[uid][sid]['target_th'] = [data_input[uid][sid]['target_th'][-1]]

            # current
            loc_cur_batch.append(torch.LongTensor(data_input[uid][sid]['loc'][1]))  
            tim_cur_ts = torch.LongTensor(data_input[uid][sid]['tim'][1])
            tim_w_cur_batch.append(tim_cur_ts[:, 0])      
            tim_h_cur_batch.append(tim_cur_ts[:, 1]) 
            current_len_batch.append(tim_cur_ts.shape[0])
            # history
            loc_his_batch.append(torch.LongTensor(data_input[uid][sid]['loc'][0]))
            tim_his_ts = torch.LongTensor(data_input[uid][sid]['tim'][0])
            tim_w_his_batch.append(tim_his_ts[:, 0])      
            tim_h_his_batch.append(tim_his_ts[:, 1])   
            history_len_batch.append(tim_his_ts.shape[0])
            # target



            target_l = torch.LongTensor(data_input[uid][sid]['target_l']) 
            target_l_batch.append(target_l) 
            target_len_batch.append(target_l.shape[0])
            target_th_batch.append(torch.LongTensor(data_input[uid][sid]['target_th']))   
            # catrgory
            if cat_contained:
                cat_his_batch.append(torch.LongTensor(data_input[uid][sid]['cat'][0])) 
                cat_cur_batch.append(torch.LongTensor(data_input[uid][sid]['cat'][1])) 
                target_c_batch.append(torch.LongTensor(data_input[uid][sid]['target_c']))

            # if i >= 226:
            #     print(f'current_len_batch: {current_len_batch}')
            #     print(f'history_len_batch: {history_len_batch}')

        # padding
        uid_batch_tensor = torch.LongTensor(uid_batch).to(device)
        # current
        loc_cur_batch_pad = pad_sequence(loc_cur_batch, batch_first=True).to(device)
        tim_w_cur_batch_pad = pad_sequence(tim_w_cur_batch, batch_first=True).to(device)
        tim_h_cur_batch_pad = pad_sequence(tim_h_cur_batch, batch_first=True).to(device)
        # history
        loc_his_batch_pad = pad_sequence(loc_his_batch, batch_first=True).to(device)
        tim_w_his_batch_pad = pad_sequence(tim_w_his_batch, batch_first=True).to(device)
        tim_h_his_batch_pad = pad_sequence(tim_h_his_batch, batch_first=True).to(device)
        # target
        target_l_batch_pad = pad_sequence(target_l_batch, batch_first=True).to(device)   
        target_th_batch_pad = pad_sequence(target_th_batch, batch_first=True).to(device)

        if cat_contained:    
            cat_his_batch_pad = pad_sequence(cat_his_batch, batch_first=True).to(device)
            cat_cur_batch_pad = pad_sequence(cat_cur_batch, batch_first=True).to(device)
            target_c_batch_pad = pad_sequence(target_c_batch, batch_first=True).to(device)

            yield  (target_len_batch, history_len_batch, current_len_batch),\
                    (target_l_batch_pad, target_th_batch_pad, target_c_batch_pad),\
                    (uid_batch_tensor,\
                         loc_his_batch_pad, loc_cur_batch_pad,\
                         tim_w_his_batch_pad, tim_w_cur_batch_pad,\
                         tim_h_his_batch_pad, tim_h_cur_batch_pad,\
                         cat_his_batch_pad, cat_cur_batch_pad)
        else:
            yield  (target_len_batch, history_len_batch, current_len_batch),\
                    (target_l_batch_pad, target_th_batch_pad),\
                    (uid_batch_tensor,\
                         loc_his_batch_pad, loc_cur_batch_pad,\
                         tim_w_his_batch_pad, tim_w_cur_batch_pad,\
                         tim_h_his_batch_pad, tim_h_cur_batch_pad)


def c_generate_batch_data(data_input, data_id, device, batch_size, cat_contained, type,
                            train_mask_uid_sid, test_mask_uid_sid, mask_uid_sid, setting):
    '''generate batch data'''

    # data_id: {uid: [sid, ...], uid: [sid, ...], ...}

    # generate (uid, sid) queue
    data_queue = list()
    uid_list = data_id.keys()

    for uid in uid_list:
        for sid in data_id[uid]:
            if (uid, sid) not in mask_uid_sid:
                if setting == 0:
                    data_queue.append((uid, sid))
                elif setting == 1:
                    if type == 'test':
                        if (uid, sid) in test_mask_uid_sid:
                            continue
                        else:
                            data_queue.append((uid, sid))
                    elif type == 'train':
                        if (uid, sid) in train_mask_uid_sid:
                            continue
                        else:
                            data_queue.append((uid, sid))
                elif setting == 2:
                    if type == 'test':
                        if (uid, sid) in test_mask_uid_sid:
                            data_queue.append((uid, sid))
                        else:
                            continue
                    elif type == 'train':
                        if (uid, sid) in train_mask_uid_sid:
                            data_queue.append((uid, sid))
                        else:
                            continue
                else:
                    assert 1 == 2
            else:
                continue

    # generate batch data
    data_len = len(data_queue)
    batch_num = int(data_len / batch_size)

    # iterate batch number times
    for i in range(batch_num):
        # print(f'i: {i}')
        # batch data
        uid_batch = []
        loc_his_batch = []
        tim_w_his_batch = []
        tim_h_his_batch = []
        target_l_batch = []
        target_c_batch = []
        target_th_batch = []
        target_len_batch = []
        history_len_batch = []
        current_len_batch = []
        if cat_contained:
            cat_his_batch = []


        batch_idx_list = np.random.choice(data_len, batch_size, replace=False)
        # iterate batch index

        for batch_idx in batch_idx_list:

            uid, sid = data_queue[batch_idx]
            uid_batch.append([uid])
            curr_input = copy.deepcopy(data_input[uid][sid])

            curr_input['loc'][0].extend(curr_input['loc'][1][:])
            curr_input['tim'][0].extend(curr_input['tim'][1][:])
            curr_input['cat'][0].extend(curr_input['cat'][1][:])
            curr_input['loc'] = [curr_input['loc'][0]]
            curr_input['tim'] = [curr_input['tim'][0]]
            curr_input['cat'] = [curr_input['cat'][0]]
            curr_input['target_l'] = [curr_input['target_l'][-1]]
            curr_input['target_c'] = [curr_input['target_c'][-1]]
            curr_input['target_th'] = [curr_input['target_th'][-1]]

            # history
            loc_his_batch.append(torch.LongTensor(curr_input['loc'][0]))
            tim_his_ts = torch.LongTensor(curr_input['tim'][0])
            tim_w_his_batch.append(tim_his_ts[:, 0])
            tim_h_his_batch.append(tim_his_ts[:, 1])
            history_len_batch.append(tim_his_ts.shape[0])
            # target

            target_l = torch.LongTensor(curr_input['target_l'])
            target_l_batch.append(target_l)
            target_len_batch.append(target_l.shape[0])
            target_th_batch.append(torch.LongTensor(curr_input['target_th']))
            # catrgory
            if cat_contained:
                cat_his_batch.append(torch.LongTensor(curr_input['cat'][0]))
                target_c_batch.append(torch.LongTensor(curr_input['target_c']))


        # padding
        uid_batch_tensor = torch.LongTensor(uid_batch).to(device)
        # history
        loc_his_batch_pad = pad_sequence(loc_his_batch, batch_first=True).to(device)
        tim_w_his_batch_pad = pad_sequence(tim_w_his_batch, batch_first=True).to(device)
        tim_h_his_batch_pad = pad_sequence(tim_h_his_batch, batch_first=True).to(device)
        # target
        target_l_batch_pad = pad_sequence(target_l_batch, batch_first=True).to(device)
        target_th_batch_pad = pad_sequence(target_th_batch, batch_first=True).to(device)


        if cat_contained:
            cat_his_batch_pad = pad_sequence(cat_his_batch, batch_first=True).to(device)
            target_c_batch_pad = pad_sequence(target_c_batch, batch_first=True).to(device)

            unique_loc_batch = []
            unique_cat_batch = []
            for i in range(batch_size):
                l = torch.unique(loc_his_batch_pad[i])[1:]
                l = l[l != target_l_batch_pad[i, 0]]
                unique_loc_batch.append(l)
                c = torch.unique(cat_his_batch_pad[i])[1:]
                c = c[c != target_c_batch_pad[i, 0]]
                unique_cat_batch.append(c)

            num_unique_loc_batch = [len(unique_loc_batch[i]) for i in range(batch_size)]
            num_unique_cat_batch = [len(unique_cat_batch[i]) for i in range(batch_size)]
            # print(f'unique_loc_batch: {len(unique_loc_batch)}\n{unique_loc_batch}')
            # print(f'num_unique_loc_batch: \n{num_unique_loc_batch}')

            # assert 1==2


            yield (target_len_batch, history_len_batch), \
                  (target_l_batch_pad, target_th_batch_pad, target_c_batch_pad), \
                  (uid_batch_tensor, loc_his_batch_pad, tim_w_his_batch_pad, tim_h_his_batch_pad, cat_his_batch_pad), \
                  (unique_loc_batch, num_unique_loc_batch, unique_cat_batch, num_unique_cat_batch)
        else:
            yield (target_len_batch, history_len_batch), \
                  (target_l_batch_pad, target_th_batch_pad), \
                  (uid_batch_tensor, loc_his_batch_pad, tim_w_his_batch_pad, tim_h_his_batch_pad)

    # print('Batch Finished')
    # assert 1==2


def newloss_generate_batch_data(data_input, data_id, device, batch_size, cat_contained, type,
                            train_mask_uid_sid, test_mask_uid_sid, mask_uid_sid, setting):
    '''generate batch data'''

    # generate (uid, sid) queue
    data_queue = list()
    uid_list = data_id.keys()

    for uid in uid_list:
        for sid in data_id[uid]:
            if (uid, sid) not in mask_uid_sid:
                if setting == 0:
                    data_queue.append((uid, sid))
                elif setting == 1:
                    if type == 'test':
                        if (uid, sid) in test_mask_uid_sid:
                            continue
                        else:
                            data_queue.append((uid, sid))
                    elif type == 'train':
                        if (uid, sid) in train_mask_uid_sid:
                            continue
                        else:
                            data_queue.append((uid, sid))
                elif setting == 2:
                    if type == 'test':
                        if (uid, sid) in test_mask_uid_sid:
                            data_queue.append((uid, sid))
                        else:
                            continue
                    elif type == 'train':
                        if (uid, sid) in train_mask_uid_sid:
                            data_queue.append((uid, sid))
                        else:
                            continue
                else:
                    assert 1 == 2
            else:
                continue

    # generate batch data
    data_len = len(data_queue)
    batch_num = int(data_len / batch_size)

    # iterate batch number times
    for i in range(batch_num):
        uid_batch = []
        loc_his_batch = []
        tim_w_his_batch = []
        tim_h_his_batch = []
        target_l_batch = []
        target_c_batch = []
        target_th_batch = []
        target_len_batch = []
        history_len_batch = []
        current_len_batch = []
        if cat_contained:
            cat_his_batch = []

        batch_idx_list = np.random.choice(data_len, batch_size, replace=False)

        for batch_idx in batch_idx_list:
            uid, sid = data_queue[batch_idx]
            uid_batch.append([uid])
            curr_input = copy.deepcopy(data_input[uid][sid])

            curr_input['loc'][0].extend(curr_input['loc'][1][:])
            curr_input['tim'][0].extend(curr_input['tim'][1][:])
            curr_input['cat'][0].extend(curr_input['cat'][1][:])
            curr_input['loc'] = [curr_input['loc'][0]]
            curr_input['tim'] = [curr_input['tim'][0]]
            curr_input['cat'] = [curr_input['cat'][0]]
            curr_input['target_l'] = [curr_input['target_l'][-1]]
            curr_input['target_c'] = [curr_input['target_c'][-1]]
            curr_input['target_th'] = [curr_input['target_th'][-1]]

            # history
            loc_his_batch.append(torch.LongTensor(curr_input['loc'][0]))
            tim_his_ts = torch.LongTensor(curr_input['tim'][0])
            tim_w_his_batch.append(tim_his_ts[:, 0])
            tim_h_his_batch.append(tim_his_ts[:, 1])
            history_len_batch.append(tim_his_ts.shape[0])
            # target

            target_l = torch.LongTensor(curr_input['target_l'])
            target_l_batch.append(target_l)
            target_len_batch.append(target_l.shape[0])
            target_th_batch.append(torch.LongTensor(curr_input['target_th']))
            # catrgory
            if cat_contained:
                cat_his_batch.append(torch.LongTensor(curr_input['cat'][0]))
                target_c_batch.append(torch.LongTensor(curr_input['target_c']))

        # padding
        uid_batch_tensor = torch.LongTensor(uid_batch).to(device)
        # history
        loc_his_batch_pad = pad_sequence(loc_his_batch, batch_first=True).to(device)
        tim_w_his_batch_pad = pad_sequence(tim_w_his_batch, batch_first=True).to(device)
        tim_h_his_batch_pad = pad_sequence(tim_h_his_batch, batch_first=True).to(device)
        # target
        target_l_batch_pad = pad_sequence(target_l_batch, batch_first=True).to(device)
        target_th_batch_pad = pad_sequence(target_th_batch, batch_first=True).to(device)

        if cat_contained:
            cat_his_batch_pad = pad_sequence(cat_his_batch, batch_first=True).to(device)
            target_c_batch_pad = pad_sequence(target_c_batch, batch_first=True).to(device)

            unique_loc_batch, unique_cat_batch, count_loc_batch, count_cat_batch = [], [], [], []

            for i in range(batch_size):
                l, l_counts = torch.unique(loc_his_batch_pad[i], return_counts=True)
                l = l[1:]
                l_counts = l_counts[1:]
                l_counts = l_counts[l != target_l_batch_pad[i, 0]]
                l = l[l != target_l_batch_pad[i, 0]]
                unique_loc_batch.append(l)
                count_loc_batch.append(l_counts / torch.sum(l_counts))

                c, c_counts = torch.unique(cat_his_batch_pad[i], return_counts=True)
                c = c[1:]
                c_counts = c_counts[1:]
                c_counts = c_counts[c != target_c_batch_pad[i, 0]]
                c = c[c != target_c_batch_pad[i, 0]]
                unique_cat_batch.append(c)
                count_cat_batch.append(c_counts / torch.sum(c_counts))

            num_unique_loc_batch = [len(unique_loc_batch[i]) for i in range(batch_size)]
            num_unique_cat_batch = [len(unique_cat_batch[i]) for i in range(batch_size)]

            # print(f'num_unique_loc_batch: {0 in num_unique_loc_batch}')
            # print(f'num_unique_cat_batch: {0 in num_unique_cat_batch}')

            # if 0 in num_unique_loc_batch:
            #     print(f'loc_his_batch_pad: {loc_his_batch_pad}')
            #     print(f'num_unique_loc_batch: {num_unique_loc_batch}')
            #     print(f'num_unique_loc_batch: {num_unique_loc_batch.index(0)}')
            #     torch.set_printoptions(threshold=99999999)
            #     print(f'xx: {loc_his_batch_pad[num_unique_loc_batch.index(0)]}')
            #     assert 1==2
            # if 0 in num_unique_cat_batch:
            #     print(f'num_unique_cat_batch: {num_unique_cat_batch}')

            yield (target_len_batch, history_len_batch), \
                  (target_l_batch_pad, target_th_batch_pad, target_c_batch_pad), \
                  (uid_batch_tensor, loc_his_batch_pad, tim_w_his_batch_pad, tim_h_his_batch_pad, cat_his_batch_pad), \
                  (unique_loc_batch, num_unique_loc_batch, count_loc_batch, unique_cat_batch, num_unique_cat_batch, count_cat_batch)
        else:
            yield (target_len_batch, history_len_batch), \
                  (target_l_batch_pad, target_th_batch_pad), \
                  (uid_batch_tensor, loc_his_batch_pad, tim_w_his_batch_pad, tim_h_his_batch_pad)


def newloss_smooth_generate_batch_data(data_input, data_id, device, batch_size, cat_contained, type,
                            train_mask_uid_sid, test_mask_uid_sid, mask_uid_sid, setting):
    '''generate batch data'''

    # generate (uid, sid) queue
    data_queue = list()
    uid_list = data_id.keys()

    for uid in uid_list:
        for sid in data_id[uid]:
            if (uid, sid) not in mask_uid_sid:
                if setting == 0:
                    data_queue.append((uid, sid))
                elif setting == 1:
                    if type == 'test':
                        if (uid, sid) in test_mask_uid_sid:
                            continue
                        else:
                            data_queue.append((uid, sid))
                    elif type == 'train':
                        if (uid, sid) in train_mask_uid_sid:
                            continue
                        else:
                            data_queue.append((uid, sid))
                elif setting == 2:
                    if type == 'test':
                        if (uid, sid) in test_mask_uid_sid:
                            data_queue.append((uid, sid))
                        else:
                            continue
                    elif type == 'train':
                        if (uid, sid) in train_mask_uid_sid:
                            data_queue.append((uid, sid))
                        else:
                            continue
                else:
                    assert 1 == 2
            else:
                continue

    # generate batch data
    data_len = len(data_queue)
    batch_num = int(data_len / batch_size)

    # iterate batch number times
    for i in range(batch_num):
        uid_batch = []
        loc_his_batch = []
        tim_w_his_batch = []
        tim_h_his_batch = []
        target_l_batch = []
        target_c_batch = []
        target_th_batch = []
        target_len_batch = []
        history_len_batch = []
        current_len_batch = []
        cat_his_batch = []

        batch_idx_list = np.random.choice(data_len, batch_size, replace=False)

        for batch_idx in batch_idx_list:
            uid, sid = data_queue[batch_idx]
            uid_batch.append([uid])
            curr_input = copy.deepcopy(data_input[uid][sid])

            curr_input['loc'][0].extend(curr_input['loc'][1][:])
            curr_input['tim'][0].extend(curr_input['tim'][1][:])
            curr_input['cat'][0].extend(curr_input['cat'][1][:])
            curr_input['loc'] = [curr_input['loc'][0]]
            curr_input['tim'] = [curr_input['tim'][0]]
            curr_input['cat'] = [curr_input['cat'][0]]
            curr_input['target_l'] = [curr_input['target_l'][-1]]
            curr_input['target_c'] = [curr_input['target_c'][-1]]
            curr_input['target_th'] = [curr_input['target_th'][-1]]

            # history
            loc_his_batch.append(torch.LongTensor(curr_input['loc'][0]))
            tim_his_ts = torch.LongTensor(curr_input['tim'][0])
            tim_w_his_batch.append(tim_his_ts[:, 0])
            tim_h_his_batch.append(tim_his_ts[:, 1])
            history_len_batch.append(tim_his_ts.shape[0])
            # target

            target_l = torch.LongTensor(curr_input['target_l'])
            target_l_batch.append(target_l)
            target_len_batch.append(target_l.shape[0])
            target_th_batch.append(torch.LongTensor(curr_input['target_th']))
            # catrgory
            cat_his_batch.append(torch.LongTensor(curr_input['cat'][0]))
            target_c_batch.append(torch.LongTensor(curr_input['target_c']))

        # padding
        uid_batch_tensor = torch.LongTensor(uid_batch).to(device)
        # history
        loc_his_batch_pad = pad_sequence(loc_his_batch, batch_first=True).to(device)
        tim_w_his_batch_pad = pad_sequence(tim_w_his_batch, batch_first=True).to(device)
        tim_h_his_batch_pad = pad_sequence(tim_h_his_batch, batch_first=True).to(device)
        # target
        target_l_batch_pad = pad_sequence(target_l_batch, batch_first=True).to(device)
        target_th_batch_pad = pad_sequence(target_th_batch, batch_first=True).to(device)

        cat_his_batch_pad = pad_sequence(cat_his_batch, batch_first=True).to(device)
        target_c_batch_pad = pad_sequence(target_c_batch, batch_first=True).to(device)

        unique_loc_batch, unique_cat_batch, count_loc_batch, count_cat_batch = [], [], [], []

        for i in range(batch_size):
            l, l_counts = torch.unique(loc_his_batch_pad[i], return_counts=True)
            l = l[1:]
            l_counts = l_counts[1:]
            l_counts = l_counts[l != target_l_batch_pad[i, 0]]
            l = l[l != target_l_batch_pad[i, 0]]
            unique_loc_batch.append(l)
            count_loc_batch.append(F.softmax(l_counts / torch.sum(l_counts)))

            c, c_counts = torch.unique(cat_his_batch_pad[i], return_counts=True)
            c = c[1:]
            c_counts = c_counts[1:]
            c_counts = c_counts[c != target_c_batch_pad[i, 0]]
            c = c[c != target_c_batch_pad[i, 0]]
            unique_cat_batch.append(c)
            count_cat_batch.append(F.softmax(c_counts / torch.sum(c_counts)))

        num_unique_loc_batch = [len(unique_loc_batch[i]) for i in range(batch_size)]
        num_unique_cat_batch = [len(unique_cat_batch[i]) for i in range(batch_size)]

        yield (target_len_batch, history_len_batch), \
              (target_l_batch_pad, target_th_batch_pad, target_c_batch_pad), \
              (uid_batch_tensor, loc_his_batch_pad, tim_w_his_batch_pad, tim_h_his_batch_pad, cat_his_batch_pad), \
              (unique_loc_batch, num_unique_loc_batch, count_loc_batch, unique_cat_batch, num_unique_cat_batch, count_cat_batch)

def generate_mask(data_len):
    '''Generate mask
    Args:
        data_len : one dimension list, reflect sequence length
    '''


    mask = []
    for i_len in data_len:
        mask.append(torch.ones(i_len).bool())


    return ~pad_sequence(mask, batch_first=True)
        

def calculate_recall(target_pad, pred_pad, time=False):
    '''Calculate recall
    Args:
        target: (batch, max_seq_len), padded target
        pred: (batch, max_seq_len, pred_scores), padded
    '''
    # variable
    acc = np.zeros(3)    # 1, 5, 10
    
    # reshape and to numpy
    target_list = target_pad.data.reshape(-1).cpu().numpy()
    # topK
    pid_size = pred_pad.shape[-1]
    if time:
        pred_list = pred_pad.data.reshape(-1, pid_size)
    else:
        _, pred_list = pred_pad.data.reshape(-1, pid_size).topk(20)
    pred_list = pred_list.cpu().numpy()

    for idx, pred in enumerate(pred_list):
        target = target_list[idx]
        if target == 0:  # pad
            continue
        if target in pred[:1]:
            acc += 1
        elif target in pred[:5]:
            acc[1:] += 1
        elif target in pred[:10]:
            acc[2:] += 1

    
    return acc


def get_model_params(model):
    total_num = sum(param.numel() for param in model.parameters())
    trainable_num = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'==== Parameter numbers:\n total={total_num}, trainable={trainable_num}')
    
    