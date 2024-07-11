import numpy as np
import pandas as pd

def save_result(ll, elbo, iter, theta, beta, eps):
    import json
    result_dict = {}
    result_dict['ll'] = ll
    result_dict['elbo'] = elbo
    result_dict['iter'] = iter
    result_dict['theta'] = theta.tolist()
    result_dict['beta'] = beta.tolist()
    result_dict['eps'] = eps.tolist()
    with open('result.json', 'w') as outfile:
        json.dump(result_dict, outfile)


def load_data(num_train_user, num_train_live, path='data/data(1).csv'):
    data = pd.read_csv(path, header=0)
    # data['Count'] = data['Count'].apply(lambda x: 0 if x<=1 else x)
    user_unique_id = data['UserId'].unique()
    live_unique_id = data['RoomId'].unique()
    # get train set
    num_train_user = int(len(user_unique_id) * num_train_user)
    num_train_live = int(len(live_unique_id) * num_train_live)
    # train_user = np.random.choice(user_unique_id, size=num_train_user, replace=False)
    # train_livestream = np.random.choice(live_unique_id, size=num_train_live, replace=False)
    train_user = user_unique_id
    train_livestream = live_unique_id
    return data.values, train_user, train_livestream


def pre_process(click_train, idx_time, idx_user, idx_livestream, idx_cnt):
    # number of users
    U = len(np.unique(click_train[:, idx_user])) # user col
    # number of livestream
    L = len(np.unique(click_train[:, idx_livestream])) # livestream col
    Time = 3
    #### the rows that have positive number of clicks
    index_click_pos = np.where(click_train[:, idx_cnt] > 0);

    index_perUser = []
    for u in range(U):
        index_perUser.append(np.where(click_train[:, idx_user]==u))  # Record the index in click_train according to userID

    index_perLivestream = []
    for l in range(L):
        index_perLivestream.append(np.where(click_train[:, idx_livestream]==l))

    index_perTime = []
    for t in range(Time):
        index_perTime.append(np.where(click_train[:, idx_time]==t))

    return U, L, Time, index_click_pos, index_perUser, index_perLivestream, index_perTime, np.shape(click_train)[0]

