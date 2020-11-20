import numpy as np
import numpy.random as rd
from numpy import linalg as la
from keras.callbacks import EarlyStopping
import keras.backend as Kback
from sklearn.metrics import log_loss
from scipy.sparse import coo_matrix
import multiprocessing as mp
from scripts.verboseprinter import VerbosePrinter as VP


class GCM:
    """
    Static class for fitting Generalized click model_definitions.
    """
    # Minimum probabilistic signal
    MIN_SIGNAL = 10**(-5)

    @staticmethod
    def runEM(click_mat, var_dic, var_models, item_order, model_def,
              n_jobs=mp.cpu_count()-1, max_iter=100, seed=0, tol=10**(-3),
              earlystop_patience=5, verbose=False, keras_epochs=250, store_best=False):
        '''
        Parameters
        ----------
        :param click_mat: Sparse click matrix.
        :param var_dic: Dictionary of feature matrices, one for each variable in the model.
        :param var_models: Keras neural networks with a Tensorflow backend. One for each latent variable in the model.
        :param item_order: A session x list_size matrix, where each entry indicates the item in position k of session i.
        :param model_def: Instance of gcm_definition object, which defines the activation and transition matrix/matrices.
        :param n_jobs: Number of parallel jobs during the E-step. Note that parallelization of the M-step is regulated
        in the Keras model.
        :param max_iter: Maximum EM iterations
        :param seed: The random seed
        :param tol: Tolerance with regards to the likelihood of the model
        :param verbose: Print status of EM
        :return: A list of the fitted variable model_definitions, current parameter estimates, and the a list of entropies computed
        at each iteration.
        '''

        # Step 1: randomly initialize the parameters
        rd.seed(seed)

        no_sessions = click_mat.shape[0]
        all_click_prob = []

        all_pred = []
        early_stop_counter = 0

        param_norm = 100
        cond_entropy = []
        best_entropy = 2

        it = 0

        # Step 2: EM algorithm:
        while param_norm > tol and it < max_iter and early_stop_counter < earlystop_patience:
            VP.print("Iteration: " + str(it), verbose)

            pred = GCM._get_prediction(var_models, var_dic)

            all_pred.append(pred)

            if it > 0:
                param_norm = 0
                for var_name, param_ests in pred.items():
                    #  Compute the norm
                    param_norm += la.norm((param_ests.flatten() - all_pred[len(all_pred) - 2][var_name]))

                VP.print("Current norm: " + str(round(param_norm, 5)), verbose)

            # Step 2.1: E-step:

            VP.print("Running E-step ...", verbose)

            param_dic_list = [{'click_vec': click_mat[i, :],
                               'cur_list_pos': item_order[i, :],
                               'var_dic': pred,
                               'item_order': item_order[i, :],
                               'model_def': model_def,
                               'i': i,
                               'session_count': no_sessions}
                              for i in range(no_sessions)]

            # Used for debugging
            if n_jobs == 1:
                all_marginal_dat = [GCM._compute_marginals_IO_HMM(param_dic_list[i])
                                    for i in np.arange(no_sessions)]
            else:
                pool = mp.Pool(mp.cpu_count()-1)
                all_marginal_dat = list(pool.map(GCM._compute_marginals_IO_HMM, param_dic_list))

                pool.close()
                pool.join()

            # Format the marginal probabilities as weights
            weight_dic, click_prob = GCM._format_weights_and_covariates(all_marginal_dat, model_def)

            # # Compute entropy using the zeta vector (state probability vector)
            cur_entropy = log_loss(click_mat.flatten(), click_prob.flatten())
            cond_entropy.append(cur_entropy)
            all_click_prob.append(click_prob.flatten())

            VP.print("Current conditional entropy:" + str(round(cond_entropy[it], 5)), verbose)
            if cond_entropy[it] < best_entropy:
                # Store the best
                best_entropy = cond_entropy[it]
                early_stop_counter = 0
                if store_best:
                    for model_name, model in var_models.items():
                        model_yaml = model.to_yaml()
                        with open("./models/" + model_name +"_model.yaml", "w") as yaml_file:
                            yaml_file.write(model_yaml)
                        # serialize weights to HDF5
                        model.save_weights("./models/" + model_name + "_weights.h5")
            else:
                early_stop_counter += 1

            VP.print("Running M-step ...", verbose)
            #
            # # M-step (since keras already paralizes, I do not):
            var_models = GCM._optimize_params(var_models, weight_dic, var_dic, epochs=keras_epochs)

            it += 1

        VP.print("Finished", verbose)
        return var_models, all_pred, cond_entropy, all_click_prob

    @staticmethod
    def pos_log_loss(y_true, y_pred):
        """
        A custom loss function, should be used in the declaration of your Keras model as loss function
        :param y_true: The true y-values
        :param y_pred: The predicted y-values
        :return:
        """
        smooth = 10**(-5)

        # Ensure value is not 0 or 1, and flatten (to 1-d array)

        # Don't clip true values! we need the sign! Plus, we do not compute the log over true values
        y_true_f = Kback.flatten(y_true)
        y_pred_f = Kback.clip(Kback.flatten(y_pred), smooth, 1 - smooth)

        # Positive, as keras minimizes
        return -Kback.sum((Kback.sign(y_true_f) + 1.) * y_true_f * Kback.log(y_pred_f) / 2.
                         + (1. - (Kback.sign(y_true_f) + 1.) / 2.)
                         * Kback.abs(y_true_f) * Kback.log(1. - y_pred_f))

    @staticmethod
    def _format_weights_and_covariates(all_marginal_dat, model_def):
        weight_dic = {}
        zeta_lst = []
        i = 0

        for ses_marginal in all_marginal_dat:
            zeta_lst.append((ses_marginal['zeta_vec'] * model_def.click_states).sum(axis=0))
            for var_key in model_def.var_type.keys():
                if var_key not in weight_dic:
                    weight_dic[var_key] = ses_marginal['reg_out'][var_key]
                else:
                    weight_dic[var_key] += ses_marginal['reg_out'][var_key]

            i += 1

        # Center (sign should remain the same!):
        for key, value in weight_dic.items():
            weight_dic[key] = value / np.sum(np.abs(value))

        zeta_mat = np.vstack(zeta_lst)
        return weight_dic, zeta_mat[:, 1:]  # Remove time 0

    @staticmethod
    def _get_weight_index_matrix(list_size, no_states):
        # Helper function to indicate the indices for which we should sum the weights
        index_weight_mat = None
        for i in range(list_size):
            cur_mat = np.zeros(list_size)
            cur_mat[i] = 1
            cur_mat = np.repeat(cur_mat, no_states).reshape(list_size*no_states, 1)
            if index_weight_mat is None:
                index_weight_mat = cur_mat
            else:
                index_weight_mat = np.hstack((index_weight_mat, cur_mat))

        return index_weight_mat

    @staticmethod
    def _get_prediction(var_models, var_dic):
        # Procedure that computes the current variable predictions, using the current model parameters
        pred = {}

        for var_name, k_model in var_models.items():
            X = var_dic[var_name].astype("float32")
            model = var_models[var_name]

            # Note that since we first double the number of rows, division by 2 to always results in a natural number

            pred[var_name] = model.predict(X, batch_size=X.shape[0]).flatten()

            non_zero_pred = np.where(pred[var_name] != 0)
            pred[var_name][non_zero_pred] = np.clip(pred[var_name][non_zero_pred], 10**(-3), 1-10**(-3))

            # Replace possible NaN-values by 0s
            # pred[var_name] = np.nan_to_num(pred[var_name], nan=0)

        return pred

    @staticmethod
    def _optimize_params(var_models, weight_dic, var_dic, verbose=0, epochs=250):
        # Procedure that finds the next parameters, based on the current E-step
        callback = EarlyStopping(monitor='loss', patience=5)

        for var_name, k_model in var_models.items():
            X = np.vstack((var_dic[var_name], var_dic[var_name]))
            model = var_models[var_name]
            output_dim = int(weight_dic[var_name].shape[0]/X.shape[0]*2)
            trainable = len(model.trainable_weights) > 0

            if trainable:
                if output_dim == 1:
                    Y = weight_dic[var_name].flatten(order='F')  # column-wise flatten (row-wise is the default)
                else:
                    # output doesn't have to be equal to the number of nodes in the layer. Determine empirically:
                    output_dim = int(weight_dic[var_name].shape[0]/X.shape[0]*2)
                    Y = weight_dic[var_name].T.reshape((X.shape[0], output_dim))

                model.fit(X, Y, batch_size=min(Y.shape[0], 8192), epochs=epochs, verbose=verbose, callbacks=[callback])
            var_models[var_name] = model

        return var_models

    @staticmethod
    def _compute_marginals_IO_HMM(param_dic):
        click_vec = param_dic['click_vec']
        var_dic = param_dic['var_dic']
        item_order = param_dic['item_order']
        model_def = param_dic['model_def']
        i = param_dic['i']
        session_count = param_dic['session_count']

        # Determine the sessions weights for clicks and skips
        H_mat, zeta_vec = GCM._compute_IO_HMM_est(click_vec, var_dic, item_order, model_def, i)

        # ncs = model_def.skip_state
        # cs = model_def.click_state
        marginals = {}

        for var_key, act_value in model_def.act_matrices.items():
            # Negative H weights are just to indicate that we have 1-theta instead of theta.
            # Easy way of transmitting that information
            W_plus = H_mat * np.tile(act_value['pos_mat'], (model_def.list_size, 1, 1)).transpose((2, 1, 0))
            W_minus = H_mat * -np.tile(act_value['neg_mat'], (model_def.list_size, 1, 1)).transpose((2, 1, 0))

            if model_def.var_type[var_key] == "item":
                # First column are the positives items, second are the negatives
                cur_vec_plus = np.zeros(model_def.no_items)
                cur_vec_minus = np.zeros(model_def.no_items)
                np.put(cur_vec_plus, item_order, np.sum(W_plus, axis=(0, 1)))
                np.put(cur_vec_minus, item_order, np.sum(W_minus, axis=(0, 1)))

                marginals[var_key] = np.hstack((cur_vec_plus.reshape(-1, 1), cur_vec_minus.reshape(-1, 1)))

            elif model_def.var_type[var_key] == "session":
                cur_vec_plus = np.zeros(session_count)
                cur_vec_minus = np.zeros(session_count)
                np.put(cur_vec_plus, i, np.sum(W_plus))
                np.put(cur_vec_minus, i, np.sum(W_minus))

                marginals[var_key] = np.hstack((cur_vec_plus.reshape(-1, 1), cur_vec_minus.reshape(-1, 1)))

            elif model_def.var_type[var_key] == "state":
                marginals[var_key] = np.hstack((np.sum(W_plus, axis=2).flatten().reshape(-1, 1),
                                                np.sum(W_minus, axis=2).flatten().reshape(-1, 1)))

            elif model_def.var_type[var_key] == "pos":
                marginals[var_key] = \
                    np.hstack((np.repeat(np.sum(W_plus, axis=(0, 1)), model_def.no_states**2).reshape(-1, 1),
                               np.repeat(np.sum(W_minus, axis=(0, 1)), model_def.no_states**2).reshape(-1, 1)))

            # Standardize:
            # if np.sum(np.abs(marginals[var_key])) == 0:
            #     print("wef")
            #
            # marginals[var_key] = marginals[var_key] / np.sum(np.abs(marginals[var_key]))

        return {'reg_out': marginals, 'zeta_vec': zeta_vec}

    @staticmethod
    def _compute_IO_HMM_est(click_vec, var_dic, item_order, model_def, i):
        # Determine all sessions weights and the state probabilities (zeta vector)
        #print(str(i))
        trans_matrices = GCM._get_trans_mat(model_def, var_dic, item_order, i)
        x_init_state = model_def.init_state
        list_size = model_def.list_size

        # Please note: In the paper I've defined H, y and M for t=1 to T, here I shift it to t=0 to T-1.

        # The forward-backward algorithm:
        # at t=0:
        B = np.zeros((model_def.no_states, list_size + 1))
        A = np.zeros((model_def.no_states, list_size + 1))

        # Just like the transition matrix, L is only on the transitions, so if the paper says t, we take t-1
        L = np.zeros((model_def.no_states, list_size))
        H = np.zeros((model_def.no_states, model_def.no_states, list_size))
        zeta = np.zeros((model_def.no_states, list_size + 1))

        click_states = model_def.click_states

        B[:, list_size] = 1

        A[x_init_state, 0] = 1  #
        zeta[x_init_state, 0] = 1
        for t in range(1, list_size+1):
            # Note that the click vector itself does not have the 0 state, so the index is one behind
            zeta[:, t] = trans_matrices[t - 1] @  zeta[:, t - 1]

            # should it be A transpose?
            A[:, t] = click_vec[t-1] * click_states[:, t] * np.dot(trans_matrices[t - 1], A[:, t - 1]) + \
                      (1 - click_vec[t-1]) * (1 - click_states[:, t]) * np.dot(trans_matrices[t - 1], A[:, t - 1])
        for t in reversed(range(1, list_size+1)):
            # Just like the transition matrix, L is only on the transitions, so if the paper says t, we take t-1!
            L[:, t-1] = click_vec[t - 1] * click_states[:, t] * B[:, t] + \
                      (1 - click_vec[t - 1]) * (1 - click_states[:, t]) * B[:, t]
            B[:, t - 1] = np.dot(trans_matrices[t-1].T, L[:, t-1])
            H[:, :, t - 1] = np.outer(L[:, t-1], A[:, t - 1]) / np.sum(A[:, list_size]) * trans_matrices[t - 1]

            # Eq. 13.33 in Bishop: this would be conditioned on all data,
            # zeta[:, t] = B[:, t] * A[:, t] / np.sum(A[:, list_size])

        return H, zeta

    @staticmethod
    def _get_trans_mat(md, vars_dic, item_order, i):
        # Initialize the M matrices:
        trans_matrices = []
        trans_prob_corr = dict(zip(list(vars_dic.keys()), np.zeros(len(vars_dic), dtype='int').tolist()))

        # Just to ensure it is a transition matrix, only the click state (= initial state) is omitted
        for t in range(md.list_size):
            trans_mat = np.ones((md.no_states, md.no_states))
            for var_name, act_mat in md.act_matrices.items():
                if var_name in md.t0_fixed and t == 0:
                    cur_param = md.t0_fixed[var_name]
                    trans_prob_corr[var_name] = 1
                elif md.var_type[var_name] == 'item':
                    cur_param = vars_dic[var_name][item_order[t - trans_prob_corr[var_name]]]
                elif md.var_type[var_name] == 'session':
                    cur_param = vars_dic[var_name][i]
                elif md.var_type[var_name] == 'state':
                    cur_param = vars_dic[var_name].reshape((md.no_states, md.no_states)).T
                elif md.var_type[var_name] == "pos":
                    size_statespace = md.no_states ** 2
                    cur_param = vars_dic[var_name][size_statespace * (t - trans_prob_corr[var_name]):
                                                   size_statespace * ((t - trans_prob_corr[var_name])+1)]\
                        .reshape((md.no_states, md.no_states))
                else:
                    raise KeyError("Parameter type " + str(md.var_type[var_name]) +
                                   " is not supported. Supported types: 'item', 'session' and 'pos'")

                update = cur_param * (act_mat['pos_mat'] - act_mat['neg_mat']) + \
                         act_mat['neg_mat'] + act_mat['fixed_mat']

                # I.e., overlapping + new updates + old updates
                trans_mat = trans_mat * update

            row_sums = np.sum(trans_mat, axis=1)

            # Check if less than 1
            try:
                np.testing.assert_array_less(row_sums, np.ones(md.no_states)+10**(-13))
            except AssertionError as e:
                raise ValueError("Probabilities in transition matrix at time: " + str(t) + ", session: " + str(i) +
                                 ", exceed one. Assertion error message: " + str(e))

            for j in range(md.no_states):
                trans_mat[md.absorbing_state[j]] = 1 - row_sums[j]

            # The original I/O-HMM paper considers the transition matrix transpose, so from that paper this is
            # a bit easier perspective.
            trans_mat = trans_mat.T
            trans_matrices.append(trans_mat)

        return trans_matrices

    @staticmethod
    def _min_prob_or_zero(val):
        # Helper function to avoid boundary problems
        if val < 10 ** (-10):  # i.e., < 10**(-5) * 10**(-5), which should not be possible
            return 0
        else:
            return max(val, GCM.MIN_SIGNAL)

