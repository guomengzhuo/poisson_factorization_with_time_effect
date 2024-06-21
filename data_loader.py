import numpy as np

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

