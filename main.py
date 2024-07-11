import argparse
import random
import numpy as np
from poisson_factorization_with_time_effect import data_loader, get_ELBO, model


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

parser = argparse.ArgumentParser(description="Poisson Factorization for livestream")
parser.add_argument('--a', type=float, default=0.4, help='shape para for theta')
parser.add_argument('--b', type=float, default=0.4, help='shape para for beta')
parser.add_argument('--c1', type=float, default=0.3, help='shape para for phi')
parser.add_argument('--c2', type=float, default=0.5, help='rate para for phi')
parser.add_argument('--d1', type=float, default=0.3, help='shape para for eta')
parser.add_argument('--d2', type=float, default=0.5, help='rate para for eta')
parser.add_argument('--e1', type=float, default=0.1, help='shape para for eps')
parser.add_argument('--e2', type=float, default=0.3, help='rate para for eps')
parser.add_argument('--K', type=int, default=10, help='number of latent elements')

parser.add_argument('--idx_time', type=int, default=0, help='position of time col')
parser.add_argument('--idx_user', type=int, default=1, help='position of user col')
parser.add_argument('--idx_livestream', type=int, default=2, help='position of livestream col')
parser.add_argument('--idx_cnt', type=int, default=3, help='position of cnt col')

parser.add_argument('--max_iter', type=int, default=2000, help='maximum number of iterations')
parser.add_argument('--eval', type=int, default=10, help='report frequency')
parser.add_argument('--tol', type=float, default=0.01, help='stopping criterion')

parser.add_argument('--num_train_user', type=float, default=0.8, help='percent of train user')
parser.add_argument('--num_train_live', type=float, default=0.8, help='percent of train livestream')

args = parser.parse_args()



if __name__ == '__main__':
    set_seed()
    click_train, train_user, train_livestream = data_loader.load_data(args.num_train_user, args.num_train_live)
    train_time = [0, 1, 2]

    idx_time = args.idx_time
    idx_user = args.idx_user
    idx_livestream = args.idx_livestream
    idx_cnt = args.idx_cnt

    U, L, Time, index_click_pos, index_perUser, index_perLivestream, index_perTime, N_pair = data_loader.pre_process(click_train,
                                                                                                                     idx_time,
                                                                                                                     idx_user,
                                                                                                                     idx_livestream,
                                                                                                                     idx_cnt)
    # hyperparmeters
    a = args.a
    b = args.b
    c1 = args.c1
    c2 = args.c2
    d1= args.d1
    d2 = args.d2
    e1 = args.e1
    e2 = args.e2
    K = args.K

    # initialization
    phi_init = np.random.gamma(c1, c2, size=U) * 5
    eta_init = np.random.gamma(d1, d2, size=L) * 5
    epsilon_init = np.random.gamma(e1, e2, size=(Time, K)) * 5
    # generate theta/beta
    theta_init = np.random.gamma(shape=a * np.ones_like(np.reshape(phi_init, (1, -1))),
                                 scale=np.reshape(phi_init, (1, -1)), size=(K, U)).T *5
    beta_init = np.random.gamma(shape=b * np.ones_like(np.reshape(eta_init, (1, -1))),
                                scale=np.reshape(eta_init, (1, -1)), size=(K, L)).T * 5

    theta_shp = theta_init
    theta_rte = theta_init / theta_init

    beta_shp = beta_init
    beta_rte = beta_init / beta_init

    epsilon_shp = epsilon_init
    epsilon_rte = epsilon_init / epsilon_init

    phi_shp = phi_init
    phi_rte = phi_init / phi_init

    eta_shp = eta_init
    eta_rte = eta_init / eta_init

    # aux click and expectation
    mu_z = model.aux_click(idx_time, idx_user, idx_livestream, idx_cnt,
                           N_pair, K, index_click_pos, click_train,
                           theta_shp, theta_rte, beta_shp, beta_rte,
                           epsilon_shp, epsilon_rte)

    Za, Zb = model.expected_aux(K, click_train, idx_cnt, mu_z)

    # start calucate ELBO
    ELBO_pre = 0
    ll_pre = 0
    iter = 1
    diff = 10.0
    tol = args.tol
    max = args.max_iter
    eval = args.eval

    # LL and ELBO save
    ll_result = []
    elbo_result = []
    iter_result = []
    while (abs(diff) > tol and iter <= max):

        # update parameter
        theta_shp, theta_rte = model.get_theta(U, train_user, Time, K, a, click_train,
                                               beta_shp, beta_rte, epsilon_shp, epsilon_rte,
                                               index_perUser, phi_shp, phi_rte, Za, Zb,
                                               idx_time, idx_livestream)

        beta_shp, beta_rte = model.get_beta(L, train_livestream, Time, K, b, click_train, theta_shp, theta_rte,
                                            index_perLivestream, eta_shp,
                                            eta_rte, Za, idx_time, idx_user)
        epsilon_shp, epsilon_rte = model.get_eps(Time, train_time, Time, K, e1, e2, click_train, theta_shp, theta_rte,
                                                 index_perTime, Zb, idx_livestream,
                                                 idx_user)
        phi_shp, phi_rte = model.get_phi(U, train_user, K, a, c1, c2, theta_shp, theta_rte, click_train, index_perUser,
                                         idx_user)
        eta_shp, eta_rte = model.get_eta(L, train_livestream, K, b, d1, d2, beta_shp, beta_rte, click_train,
                                         index_perLivestream, idx_livestream)
        # aux variable
        mu_z = model.aux_click(idx_time, idx_user, idx_livestream, idx_cnt,
                               N_pair, K, index_click_pos, click_train,
                               theta_shp, theta_rte, beta_shp, beta_rte,
                               epsilon_shp, epsilon_rte)
        Za, Zb = model.expected_aux(K, click_train, idx_cnt, mu_z)

        # iteration evaluation
        if (iter % eval == 0 or iter==1):
            ELBO_new = get_ELBO.ELBO(K, Za, Zb, mu_z, index_click_pos, click_train, idx_time, idx_user, idx_livestream,
                                     theta_shp, theta_rte,
                                     beta_shp, beta_rte,
                                     epsilon_shp, epsilon_rte,
                                     a, b, c1, c2, d1, d2, e1, e2,
                                     phi_shp, phi_rte,
                                     eta_shp, eta_rte,
                                     )
            diff = (ELBO_new - ELBO_pre)/ELBO_pre
            ELBO_pre = ELBO_new
            data_ll = get_ELBO.data_loglike(theta_shp / theta_rte, beta_shp / beta_rte, epsilon_shp / epsilon_rte,
                                            idx_time, idx_user, idx_livestream, idx_cnt, click_train)
            ll_diff = data_ll - ll_pre
            ll_pre = data_ll
            print("At iteration {}, diff(ELBO) = {}, new ELBO={}, data_ll={}, diff_ll={}".format(iter, diff, ELBO_new,
                                                                                                 data_ll, ll_diff))
            ll_result.append(data_ll)
            elbo_result.append(ELBO_new)
            iter_result.append(iter)
        iter += 1
    # save results
    theta = theta_shp / theta_rte
    beta = beta_shp / beta_rte
    epsilon = epsilon_shp / epsilon_rte
    data_loader.save_result(ll_result, elbo_result, iter_result, theta, beta, epsilon)
