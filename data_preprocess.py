import os
import csv
import time
import argparse


def settings():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./raw_data/dataset_tsmc2014/', type=str,
                       help='data path')
    parser.add_argument('--dataname', default='NYC', type=str,
                       help='data name')
    parser.add_argument('--filetype', default='txt', type=str,
                       help='file type')
    parser.add_argument('--user_po', default=0, type=int,
                       help='Position in record for user')
    parser.add_argument('--loc_po', default=1, type=int,
                       help='Position in record for location')
    parser.add_argument('--tim_po', default=2, type=int,
                       help='Position in record for time')
    
    parser.add_argument('--user_record_min', default=10, type=int,
                       help='Minimun record number for user')
    parser.add_argument('--loc_record_min', default=10, type=int,
                       help='Minimun record number for location')
    return parser.parse_args()


def preprocessing(params):
    '''Preprocessing data
    Note:
        1. Raw data is sorted by user(1) and time(2)
        2. Filter sparse data with minimun record numbers.
        3. Encode user and location id.
        4. Merge data with same user and location in a day.
        
    '''
    
    # Loading and Filtering sparse data
    print('='*20, 'Loading and preprocessing sparse data')
    filepath = f'{params.path}dataset_TSMC2014_{params.dataname}.{params.filetype}'
    print(f'Path is {filepath}')
    loc_count = {}  # store location info with loc-num
    user_count = {} #  store user info with user-num 
    user_id = {}
    loc_id = {}
    
    # load file and count numbers
    print('='*20, 'Loading and Counting')
    if params.filetype == 'txt':
        with open(filepath, 'r', encoding='latin-1') as f:
            reader = csv.reader(f, delimiter='\t')
            for record in reader:
                if '' in record:
                    continue
                # print(f'record: \n{record}')
                user = record[params.user_po]
                loc = record[params.loc_po]
                if user not in user_count:
                    user_count[user] = 1
                else:
                    user_count[user] += 1
                if loc not in loc_count:
                    loc_count[loc] = 1
                else:
                    loc_count[loc] += 1
                # print(f'uesr: {user}, loc: {loc}')
                # assert 1==2
    
    record_num = os.popen(f'wc -l {filepath}').readlines()[0].split()[0]
    print(f'Finished, records is {record_num}, all user is {len(user_count)}, all location is {len(loc_count)}')
    
    # Filter and encode user and location
    print('='*20, 'Filtering and encoding')
    for i in user_count:
        if user_count[i] > params.user_record_min:
            user_id[i] = len(user_id)
    for i in loc_count:
        if loc_count[i] > params.loc_record_min:
            loc_id[i] = len(loc_id)

    # print(f'user_id: {user_id}')
    # print(f'loc_id: {loc_id}')
    # assert 1==2

    # store
    filter_path = f'{params.path}{params.dataname}_filtered.txt'
    print(f'Filter path is {filter_path}')
    with open(filter_path, 'w', encoding='latin-1') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        with open(filepath, 'r', encoding='latin-1') as f_in:
            reader = csv.reader(f_in, delimiter='\t')
            for record in reader:
                if '' in record:
                    continue
                user = record[params.user_po]
                loc = record[params.loc_po]
                if user in user_id and loc in loc_id:
                    record[params.user_po] = user_id[user]
                    record[params.loc_po] = loc_id[loc]
                    writer.writerow(record)
    
    record_num = os.popen(f'wc -l {filter_path}').readlines()[0].split()[0]
    print(f'Finished, records is {record_num}, user is {len(user_id)}, location is {len(loc_id)}')
    
    
    # Merge data 
    print('='*20, 'Merging')
    merge_path = f'{params.path}{params.dataname}_merged.txt'
    print(f'Merge path is {filter_path}')
    with open(merge_path, 'w', encoding='latin-1') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        # get first record
        with open(filter_path, 'r', encoding='latin-1') as f_in:
            pre_record = f_in.readlines()[0].split('\t')
        # all record
        with open(filter_path, 'r', encoding='latin-1') as f_in:
            reader = csv.reader(f_in, delimiter='\t')
            for record in reader:
                # same person, same location, same day
                if record[params.user_po] == pre_record[params.user_po] and \
                    record[params.loc_po] == pre_record[params.loc_po] and \
                    record[params.tim_po].split('T')[0] == pre_record[params.tim_po].split('T')[0]:   
                    continue
                writer.writerow(record)
                pre_record = record
     
    record_num = os.popen(f'wc -l {merge_path}').readlines()[0].split()[0]
    print(f'Finished, records is {record_num}')             
   

    
if __name__ == '__main__':
    
    start_time = time.time()
    params = settings()
    preprocessing(params)
    print('Time cost:', f'{time.time()-start_time:.0f}')