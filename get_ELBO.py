import numpy as np
from scipy.special import digamma, loggamma
from scipy.stats import poisson


def scale_diff(a, b, shp, rte):  # epsilon, eta, phi
    temp = (a-shp) * digamma(shp) - a * np.log(rte) + shp - b * shp/rte + loggamma(shp)
    return np.sum(temp)

def gamma_diff(a, shp, rte, b_shp, b_rte):  # theta, beta
    b = b_shp/b_rte
    temp = (a - shp) * digamma(shp) - a * np.log(rte) - b * shp/rte + shp + loggamma(shp) + a*(digamma(b_shp) - np.log(b_rte))
    return np.sum(temp)

def count_diff(K, Za, Zb, mu_z, index_click_pos, click_nonCens, idx_time, idx_user, idx_livestream,
               theta_shp, theta_rte,
               beta_shp, beta_rte,
               eps_shp, eps_rte):
    user = click_nonCens[:, idx_user][index_click_pos]
    livestream = click_nonCens[:, idx_livestream][index_click_pos]
    time = click_nonCens[:, idx_time][index_click_pos]
    temp1 = beta_shp[livestream, ]/beta_rte[livestream, ] * theta_shp[user, ]/theta_rte[user, ]
    temp2 = eps_shp[time, ]/eps_rte[time, ] * theta_shp[user, ]/theta_rte[user, ]
    temp3 = Za[index_click_pos, ] * (digamma(theta_shp[user, ]) - np.log(theta_rte[user, ]) + digamma(beta_shp[livestream, ]) - np.log(beta_rte[livestream, ]) - np.log(mu_z[index_click_pos,0:K]+1e-30))
    temp4 = Zb[index_click_pos, ] * (digamma(theta_shp[user, ]) - np.log(theta_rte[user, ]) + digamma(eps_shp[time, ]) - np.log(eps_rte[time, ]) - np.log(mu_z[index_click_pos,K:]+1e-30))
    temp = np.sum(temp3) + np.sum(temp4) -np.sum(temp1)-np.sum(temp2)
    # temp = -np.sum(temp3) - np.sum(temp4) -np.sum(temp1)-np.sum(temp2)
    return temp



def ELBO(K, Za, Zb, mu_z, index_click_pos, click_train, idx_time, idx_user, idx_livestream,
                      theta_shp, theta_rte,
                      beta_shp, beta_rte,
                      epsilon_shp, epsilon_rte,
         a, b, c1, c2, d1, d2, e1, e2,
         phi_shp, phi_rte,
         eta_shp, eta_rte,
         ):
    temp1 = count_diff(K, Za, Zb, mu_z, index_click_pos, click_train, idx_time, idx_user, idx_livestream,
                      theta_shp, theta_rte,
                      beta_shp, beta_rte,
                      epsilon_shp, epsilon_rte)
    temp2 = scale_diff(c1, c2, phi_shp, phi_rte) + scale_diff(d1, d2, eta_shp, eta_rte) + scale_diff(e1, e2, epsilon_shp, epsilon_rte)
    temp3 = gamma_diff(a, theta_shp, theta_rte, phi_shp, phi_rte) + gamma_diff(b, beta_shp, beta_rte, eta_shp, eta_rte)

    return temp1 + temp2 + temp3

def data_loglike(theta, beta, eps,
                 idx_time, idx_user, idx_livestream, idx_cnt,
                 click_nonCens):
    # a = theta[click_nonCens[:, idx_user],:]
    # b = beta[click_nonCens[:, idx_livestream], :]
    # c = eps[click_nonCens[:, idx_time],:]
    mat1 = theta[click_nonCens[:, idx_user],:]
    mat2 = beta[click_nonCens[:, idx_livestream], :] + eps[click_nonCens[:, idx_time],:]
    # rate = np.zeros((mat1.shape[0], 1), dtype='float16')
    ll = 0.
    for idx in range(mat1.shape[0]):
        rate = np.sum(mat1[idx, :] * mat2[idx, :])
        ll += poisson.logpmf(click_nonCens[idx, idx_cnt], rate)
    # temp = poisson.logpmf(click_nonCens[:, idx_cnt], rate)
    # rate = np.matmul(theta[click_nonCens[:, idx_user],:], (beta[click_nonCens[:, idx_livestream], :] + eps[click_nonCens[:, idx_time],:]).T)
    # temp = poisson.logpmf(click_nonCens[:, idx_cnt], np.diagonal(rate))
    return ll