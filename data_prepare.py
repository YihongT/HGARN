import time
import datetime
import argparse
import numpy as np
import pickle
from math import pi
from collections import Counter



class DataGeneration(object):
    def __init__(self, params):
        
        self.__dict__.update(params.__dict__)

        self.raw_data = {}     # raw user's trajectory. {uid: [[pid, tim], ...]}
        self.poi_count = {}   # raw location counts. {pid: count}
        self.data_filtered = {}   
        self.uid_list = []   # filtered user id
        self.pid_dict = {}   # filtered location id map
        self.train_data = {} # train data with history,   {'uid': {'sid': {'loc': [], 'tim': [], 'target': [] (, 'cat': [])}}}
        self.train_id = {}   # train data session id list  
        self.test_data = {}
        self.test_id = {}
        self.tim_w = set()
        self.tim_h = set()
        
        self.raw_lat_lon = {} # count for latitude and longitude
        self.new_lat_lon = {}
        self.lat_lon_radians = {}
        
        if self.cat_contained:
            self.cid_dict = {}      # cid_dict
            self.cid_count_dict = {}  # cid count
            self.raw_cat_dict = {}  #  cid-cat
            self.new_cat_dict = {}
            self.pid_cid_dict = {}    # pid-cid dict

    # 1. read trajectory data
    def load_trajectory(self):
        with open(self.path_in + self.data_name + '_merged.txt', 'r', encoding='latin-1') as fid:
            for i, line in enumerate(fid):
                if self.data_name in ['NYC', 'TKY']:
                    uid, pid, cid, cat, lat, lon, _, tim = line.strip().split('\t')
                elif self.data_name == 'Dallas':
                    uid, tim, pid, lon, lat = line.strip().split('\t')
                else:
                    uid, tim, lat, lon, pid = line.strip().split('\t')
                    
                # Note: user and location is id
                if self.cat_contained:
                    # count uid records
                    if uid not in self.raw_data:
                        self.raw_data[uid] = [[pid, tim, cid]]
                    else:
                        self.raw_data[uid].append([pid, tim, cid]) 
                    # count raw_cid-cat 
                    if cid not in self.raw_cat_dict:
                        self.raw_cat_dict[cid] = cat                        
                else:
                    if uid not in self.raw_data:
                        self.raw_data[uid] = [[pid, tim]]
                    else:
                        self.raw_data[uid].append([pid, tim]) 
                if pid not in self.poi_count:
                    self.poi_count[pid] = 1
                else:
                    self.poi_count[pid] += 1
                    
                # count poi latitude and longitude
                if pid not in self.raw_lat_lon:
                    self.raw_lat_lon[pid] = [eval(lat), eval(lon)]

    # 2. filter users and locations, and then split trajectory into sessions
    def filter_and_divide_sessions(self):
        POI_MIN_RECORD_FOR_USER = 1  # keep same setting with DeepMove and LSTPM
        
        # filter user and location
        uid_list = [x for x in self.raw_data if len(self.raw_data[x]) > self.user_record_min]      # uid list
        pid_list = [x for x in self.poi_count if self.poi_count[x] > self.poi_record_min]  # pid list

        # print(f'uid_list: {uid_list}')

        # iterate each user
        for uid in uid_list:
            # print(f'uid: {uid}')
            user_records = self.raw_data[uid]   # user_records is [[pid, tim (, cid)]]
            topk = Counter([x[0] for x in user_records]).most_common()  # most common poi, [(poi, count), ...]
            topk_filter = [x[0] for x in topk if x[1] > POI_MIN_RECORD_FOR_USER]   # the poi that the user go more than one time 
            sessions = {}   # sessions is {'sid' : [[pid, [week, hour] (, cid)], ...]}

            # topk: [('988', 46), ('1', 40), ('985', 36), ('492', 26), ('384', 13), ('81', 1), ('74', 1), ('1689', 1), ('1442', 1)]
            # topk_filter: ['988', '1', '985', '492', '384']
            # print(f'user_records: {user_records}')

            # iterate each record
            for i, record in enumerate(user_records):
                # print(f'record: {record}')
                if self.cat_contained:
                    poi, tim, cid = record
                else:
                    poi, tim = record
                try:
                    # time processing
                    # print(f'tim: {tim}')
                    # time_struct = time.strptime(tim, "%Y-%m-%dT%H:%M:%SZ")
                    time_struct = time.strptime(tim, "%a %b %d %H:%M:%S +0000 %Y")
                    # print(f'time_struct: {time_struct}')
                    calendar_date = datetime.date(time_struct.tm_year, time_struct.tm_mon, time_struct.tm_mday).isocalendar()
                    # print(f'calendar_date: {calendar_date}')
                    # current_week = f'{calendar_date.year}-{calendar_date.week}'
                    current_week = f'{calendar_date[0]}-{calendar_date[1]}'
                    # Encode time
#                     tim_code = [time_struct.tm_wday+1, time_struct.tm_hour*2+int(time_struct.tm_min/30)+1 ]    # week(1~7), hours(1~48) 
                    tim_code = [time_struct.tm_wday+1, int(time_struct.tm_hour/2)+1]    # week(1~7), hours(1~12)
                    # revise record
                    record[1] = tim_code
                except Exception as e:
                    print('error:{}'.format(e))
                    raise Exception

                # divide session. Rule is: same week
                sid = len(sessions)  # session id
                if poi not in pid_list and poi not in topk_filter:
                    # filter the poi if poi not in topk_filter:
                    continue
                if i == 0 or len(sessions) == 0:
                    sessions[sid] = [record]
                else:
                    if last_week != current_week:    # new session
                        sessions[sid] = [record]    
                    else:
                        sessions[sid - 1].append(record)   # Note: data is already merged
                last_week = current_week

            sessions_filtered = {}
            # filter session with session_min
            for s in sessions:
                if len(sessions[s]) >= self.session_min:
                    sessions_filtered[len(sessions_filtered)] = sessions[s]
            # filter user with sessions_min, that is, the user must have sessions_min's sessions.
            if len(sessions_filtered) < self.sessions_min:
                continue

            # print(f'sessions_filtered: \n{sessions_filtered}')
            # ReEncode location index (may encode category)
            for sid in sessions_filtered:    # sessions is {'sid' : [[pid, [week, hour]], ...]}
                # print(f'sid: {sid}')
                # print(f'sessions_filtered[sid]: {sessions_filtered[sid]}')
                for idx, record in enumerate(sessions_filtered[sid]):
                    # print(f'idx: {idx}, record: {record}')
                    # reEncode location
                    if record[0] not in self.pid_dict: 
                        self.pid_dict[record[0]] = len(self.pid_dict) + 1    # the id start from 1
                    new_pid = self.pid_dict[record[0]]
                    # new pid for latitude and longitude
                    self.new_lat_lon[new_pid] = self.raw_lat_lon[record[0]] 
                    self.lat_lon_radians[new_pid] = list(np.array(self.raw_lat_lon[record[0]]) * pi / 180)
                    # assign new pid
                    record[0] = new_pid

                    # time
                    self.tim_w.add(record[1][0])
                    self.tim_h.add(record[1][1])
                    # category
                    if self.cat_contained:
                        # encode cid
                        if record[2] not in self.cid_dict:
                            new_cid = len(self.cid_dict) + 1  # the id start from 1
                            self.cid_dict[record[2]] = new_cid
                            self.new_cat_dict[new_cid] = self.raw_cat_dict[record[2]]  # raw_cid-cat to new_cid-cat
                            self.cid_count_dict[new_cid] = 1
                        # assign cid
                        record[2] = self.cid_dict[record[2]]
                        # count cid
                        self.cid_count_dict[record[2]] += 1
                        # pid-cid dict
                        if new_pid not in self.pid_cid_dict:
                            self.pid_cid_dict[new_pid] = record[2]
                    # reassign record        
                    sessions_filtered[sid][idx] = record
            # print(f'\n\n after, sessions_filtered: \n{sessions_filtered}')
            # divide train and test
            sessions_id = list(sessions_filtered.keys())
            split_id = int(np.floor(self.train_split * len(sessions_id)))
            train_id = sessions_id[:split_id]
            test_id = sessions_id[split_id:]
            # print(f'sessions_id: {sessions_id}')
            # print(f'split_id: {split_id}')
            # print(f'train_id: {train_id}')
            # print(f'test_id: {test_id}')
            # assert 1==2
            assert len(train_id) > 0, 'train sessions have error'
            assert len(test_id) > 0, 'test sessions have error'
            # preprare final data. (ReEncode user index), he id start from 1
            self.data_filtered[len(self.data_filtered)+1] = {'sessions_count': len(sessions_filtered), 'sessions': sessions_filtered, 'train': train_id, 'test': test_id}
            

        # final uid list
        self.uid_list = list(self.data_filtered.keys())
        print(f'Final user is {len(self.uid_list)}, location is {len(self.pid_dict)}')
        print(f'data_filtered: {len(self.data_filtered)}')
        # print(f'final uid list: {self.uid_list}')
        
    # 3. generate data with history sessions
    def generate_history_sessions(self, mode):
        print('='*4, f'generate history sessions in mode={mode}')
        data_input = self.data_filtered
        data = {}
        data_id = {}
        user_list = data_input.keys()

        # print(f'user_list: {user_list}')
        # assert 1==2

        for uid in user_list:
            print(f'uid: {uid}')
            data[uid] = {}
            user_sid_list = data_input[uid][mode]
            print(f'user_sid_list: {user_sid_list}')
            data_id[uid] = user_sid_list.copy()
            print(f'data_id: {data_id}')
            for idx, sid in enumerate(user_sid_list):
                # require at least one session as history;  one <- history_session_min
                if mode == 'train' and idx < self.history_session_min:
                    data_id[uid].pop(idx)
                    continue
                data[uid][sid] = {}
                loc_seq_cur = []
                loc_seq_his = []
                tim_seq_cur = []
                tim_seq_his = []
                
                if self.cat_contained:
                    cat_seq_cur = []
                    cat_seq_his = []

                if mode == 'test':  # in test mode, append all train data as history
                    train_sid = data_input[uid]['train']
                    for tmp_sid in train_sid:
                        loc_seq_his.extend([record[0] for record in data_input[uid]['sessions'][tmp_sid]])
                        tim_seq_his.extend([record[1] for record in data_input[uid]['sessions'][tmp_sid]])
                        if self.cat_contained:
                            cat_seq_his.extend([record[2] for record in data_input[uid]['sessions'][tmp_sid]])

                # append past sessions
                for past_idx in range(idx):
                    tmp_sid = user_sid_list[past_idx]
                    loc_seq_his.extend([record[0] for record in data_input[uid]['sessions'][tmp_sid]])
                    tim_seq_his.extend([record[1] for record in data_input[uid]['sessions'][tmp_sid]])
                    if self.cat_contained:
                        cat_seq_his.extend([record[2] for record in data_input[uid]['sessions'][tmp_sid]])

                # current session
                loc_seq_cur.extend([record[0] for record in data_input[uid]['sessions'][sid][:-1]])  # [[pid1], [pid2], ...]
                tim_seq_cur.extend([record[1] for record in data_input[uid]['sessions'][sid][:-1]])
                if self.cat_contained:
                    cat_seq_cur.extend([record[2]  for record in data_input[uid]['sessions'][sid][:-1]])
                # print(f'data_input[uid]["sessions"][sid]: {len(data_input[uid]["sessions"][sid])}\n{data_input[uid]["sessions"][sid]}')
                # print(f'loc_seq_cur: {len(loc_seq_cur)} {loc_seq_cur}')
                # print(f'tim_seq_cur: {len(loc_seq_cur)} {tim_seq_cur}')
                # print(f'cat_seq_cur: {len(loc_seq_cur)} {cat_seq_cur}')
                # assert 1==2
                    
                # store sequence
                data[uid][sid]['target_l'] = [record[0] for record in data_input[uid]['sessions'][sid][1:]]    
                data[uid][sid]['target_th'] = [record[1][1] for record in data_input[uid]['sessions'][sid][1:]]
                data[uid][sid]['loc'] = [loc_seq_his, loc_seq_cur]   # list
                data[uid][sid]['tim'] = [tim_seq_his, tim_seq_cur]   # list
                if self.cat_contained:
                    data[uid][sid]['cat'] = [cat_seq_his, cat_seq_cur] # list
                    data[uid][sid]['target_c'] = [record[2] for record in data_input[uid]['sessions'][sid][1:]]

            # print(f'data: \n{data}\n\n')
            # print(f'data_id: {data_id}')
            # assert 1==2

        # train/test_data is {'uid': {'sid': {'loc': [pid_seq], 'tim': [[week, hour], ...] (, 'cat': [cid_seq])}}}
        # 
        if mode == 'train':
            self.train_data = data
            self.train_id = data_id
        elif mode == 'test':
            self.test_data = data
            self.test_id = data_id
        print('Finish')
    

    # 4. save variables
    def save_variables(self):
        dataset = {'train_data': self.train_data, 'train_id': self.train_id,
                   'test_data': self.test_data, 'test_id': self.test_id,
                   'pid_dict': self.pid_dict, 'uid_list': self.uid_list,
                   'pid_lat_lon': self.new_lat_lon,
                   'pid_lat_lon_radians' : self.lat_lon_radians,
                   'parameters': self.get_parameters()}
        if self.cat_contained:
            dataset['cid_dict'] = self.cid_dict
            dataset['cid_count_dict'] = self.cid_count_dict
            dataset['cid_cat_dict'] = self.new_cat_dict
            dataset['pid_cid_dict'] = self.pid_cid_dict
            pickle.dump(dataset, open(self.path_out + self.data_name + '_cat.pkl', 'wb'))
        else:
            pickle.dump(dataset, open(self.path_out + self.data_name + '.pkl', 'wb'))

    def get_parameters(self):
        parameters = self.__dict__.copy()
        del parameters['raw_data']
        del parameters['poi_count']
        del parameters['data_filtered']
        del parameters['uid_list']
        del parameters['pid_dict']
        del parameters['train_data']
        del parameters['train_id']
        del parameters['test_data']
        del parameters['test_id']
        if self.cat_contained:
            del parameters['cid_dict']
        
        return parameters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', type=str, default='raw_data/dataset_tsmc2014/', help="input data path")
    parser.add_argument('--path_out', type=str, default='raw_data/dataset_tsmc2014/', help="output data path")
    parser.add_argument('--data_name', type=str, default='NYC', help="data name")
    parser.add_argument('--user_record_min', type=int, default=10, help="user record length filter threshold")
    parser.add_argument('--poi_record_min', type=int, default=10, help="location record length filter threshold")
    parser.add_argument('--session_min', type=int, default=2, help="control the length of session not too short")
    parser.add_argument('--sessions_min', type=int, default=5, help="the minimum amount of the user's sessions")
    parser.add_argument('--train_split', type=float, default=0.8, help="train/test ratio")
    parser.add_argument('--cat_contained', action='store_false', default=True, help="whether contain category")
    parser.add_argument('--history_session_min', type=int, default=1, help="minimun number of history session")
    
    if __name__ == '__main__':
        return parser.parse_args()
    else:
        return parser.parse_args([])


if __name__ == '__main__':
    
    start_time = time.time()

    params = parse_args()
    data_generator = DataGeneration(params)
    parameters = data_generator.get_parameters()
    print('='*20 + ' Parameter settings')
    print(',  '.join([p + '=' + str(parameters[p]) for p in parameters]))
    print('='*20 + ' Start processing')
    print('==== Load trajectory from {}'.format(data_generator.path_in))
    data_generator.load_trajectory()
    
    print('==== filter users')
    data_generator.filter_and_divide_sessions()
    
    
    print('==== generate history sessions')
    data_generator.generate_history_sessions('train')
    data_generator.generate_history_sessions('test')
    
    print('==== save prepared data')
    data_generator.save_variables()
    
    print('==== Preparetion Finished')
    print('Raw users:{} raw locations:{}'.format(
        len(data_generator.raw_data), len(data_generator.poi_count)))
    print(f'Final users:{len(data_generator.uid_list)}, min_id:{np.min(data_generator.uid_list)}, max_id:{np.max(data_generator.uid_list)}')
    pid_list = list(data_generator.pid_dict.values())
    print(f'Final locations:{len(pid_list)}, min_id:{np.min(pid_list)}, max_id:{np.max(pid_list)}')
    print(f'Final time-week:{len(data_generator.tim_w)}, min_id:{np.min(list(data_generator.tim_w))}, max_id:{np.max(list(data_generator.tim_w))}')
    print(f'Final time-hour:{len(data_generator.tim_h)}, min_id:{np.min(list(data_generator.tim_h))}, max_id:{np.max(list(data_generator.tim_h))}')
    if params.cat_contained:
        cid_list = list(data_generator.cid_dict.values())
        print(f'Final categories:{len(cid_list)}, min_id:{np.min(cid_list)}, max_id:{np.max(cid_list)}')
    print(f'Time cost is {time.time()-start_time:.0f}s')
     
