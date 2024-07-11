import numpy as np
from scipy.special import digamma, loggamma


def aux_click(idx_time, idx_user, idx_livestream, idx_cnt,
              N_pair, K, index_click_pos, click_nonCens,
              theta_shp, theta_rte, beta_shp, beta_rte,
              epsilon_shp, epsilon_rte):

    mu = np.zeros(shape=(N_pair, 2 * K))
    theta = digamma(theta_shp) - np.log(theta_rte)
    beta = digamma(beta_shp) - np.log(beta_rte)
    epsilon = digamma(epsilon_shp) - np.log(epsilon_rte)

    userId = click_nonCens[:, idx_user][index_click_pos]
    livestreamId = click_nonCens[:, idx_livestream][index_click_pos]
    timeId = click_nonCens[:, idx_time][index_click_pos]

    p1 = np.exp(theta[userId,] + beta[livestreamId,])
    p2 = np.exp(theta[userId,] + epsilon[timeId,])

    temp = np.hstack((p1, p2))
    aa = np.tile(np.sum(temp, axis=1).reshape((-1, 1)), np.shape(temp)[1])
    mu[index_click_pos,] = temp/(aa + 1e-30)
    return mu

def expected_aux(K, click_nonCens, idx_cnt, mu):
    Z = click_nonCens[:, idx_cnt].reshape(-1, 1) * mu
    Za = Z[:, 0:K]
    Zb = Z[:, K:]
    return Za, Zb


def get_theta(U, train_user, Time, K, a, click_nonCens,
              beta_shp, beta_rte, eps_shp,
              eps_rte, index_perUser,
              phi_shp, phi_rte,
              Za, Zb,
              idx_time, idx_livestream):
    _theta_shp = np.ones(shape=(U, K)) * a
    _theta_rte = (phi_shp/phi_rte).reshape(-1, 1) * np.ones(shape=(U, K))
    temp_theta1 = beta_shp/beta_rte
    temp_theta2 = eps_shp/eps_rte
    for u in train_user:
        ind_user = index_perUser[u][0]
        N = len(ind_user)
        mat1 = np.array(Za[ind_user,] + Zb[ind_user,])
        mat2 = np.array(temp_theta1[click_nonCens[:, idx_livestream][ind_user], ]).reshape((N, K)) + np.array(temp_theta2[click_nonCens[:, idx_time][ind_user], ]).reshape((N, K))
        _theta_shp[u, ] = _theta_shp[u, ] + np.sum(mat1, axis=0)
        _theta_rte[u, ] = _theta_rte[u, ] + np.sum(mat2, axis=0)
    return _theta_shp, _theta_rte

def get_beta(L, train_livestream, Time, K, b, click_nonCens,
              theta_shp, theta_rte, index_perLivestream,
              eta_shp, eta_rte,
              Za, idx_time, idx_user):
    _beta_shp = np.ones(shape=(L, K)) * b
    _beta_rte = (eta_shp / eta_rte).reshape(-1, 1) * np.ones(shape=(L, K))
    temp_beta = theta_shp / theta_rte
    for l in train_livestream:
        ind_user = index_perLivestream[l][0]
        N = len(ind_user)
        mat1 = np.array(Za[ind_user,])
        mat2 = np.array(temp_beta[click_nonCens[:, idx_user][ind_user],]).reshape((N, K))
        _beta_shp[l,] = _beta_shp[l,] + np.sum(mat1, axis=0)
        _beta_rte[l,] = _beta_rte[l,] + np.sum(mat2, axis=0)
    return _beta_shp, _beta_rte

def get_eps(T, train_time, Time, K, e1, e2, click_nonCens,
              theta_shp, theta_rte, index_perTime,
            Zb, idx_livestream, idx_user):
    _eps_shp = np.ones(shape=(T, K)) * e1
    _eps_rte = np.ones(shape=(T, K)) * e2
    temp_eps = theta_shp / theta_rte
    for t in train_time:
        ind_user = index_perTime[t][0]
        N = len(ind_user)
        mat1 = np.array(Zb[ind_user,])
        mat2 = np.array(temp_eps[click_nonCens[:, idx_user][ind_user],]).reshape((N, K))
        _eps_shp[t,] = _eps_shp[t,] + np.sum(mat1, axis=0)
        _eps_rte[t,] = _eps_rte[t,] + np.sum(mat2, axis=0)
    return _eps_shp, _eps_rte

def get_phi(U, train_user, K, a, c1, c2, theta_shp, theta_rte, click_nonCens, index_perUser, idx_user):
    _phi_shp = np.ones((U, 1)) * (K*a+c1)
    _phi_rte = np.ones((U, 1)) * c2
    temp_phi = theta_shp / theta_rte
    for u in train_user:
        # ind_user = index_perUser[u][0]
        # N = len(ind_user)
        mat2 = np.array(temp_phi[u, ])
        # mat2 = np.array(temp_phi[click_nonCens[:, idx_user][ind_user],]).reshape((N, K))
        # sum over K
        # mat2 = np.sum(mat2, axis=1)
        _phi_rte[u, ] = _phi_rte[u, ]+np.sum(mat2)
    return _phi_shp, _phi_rte

def get_eta(L, train_livestream, K, b, d1, d2, beta_shp, beta_rte, click_nonCens, index_perLivestream, idx_livestream):
    _eta_shp = np.ones((L, 1)) * (K*b+d1)
    _eta_rte = np.ones((L, 1)) * d2
    temp_eta = beta_shp / beta_rte
    for u in train_livestream:
        # ind_user = index_perLivestream[u][0]
        # N = len(ind_user)
        mat2 = np.array(temp_eta[u, ])
        # mat2 = np.array(temp_eta[click_nonCens[:, idx_livestream][ind_user],]).reshape((N, K))
        _eta_rte[u, ] = _eta_rte[u, ]+np.sum(mat2)
    return _eta_shp, _eta_rte
