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
              n_jobs=mp.cpu_count()-1, max_iter=10, seed=0, tol=10**(-3), verbose=False):
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
        list_size = model_def.list_size
        pred = {}
        var_type_dic = model_def.var_type

        param_norm = 100
        cond_entropy = []

        it = 0

        # Step 2: EM algorithm:
        while param_norm > tol and it < max_iter:
            VP.print("Iteration: " + str(it), verbose)

            pred = GCM._get_prediction(var_models, var_dic)

            if it > 0:
                param_norm = 0
                for var_name, param_ests in pred.items():
                    #  Compute the norm
                    param_norm += la.norm((param_ests.flatten(), pred[var_name]))

                VP.print("Current norm: " + str(round(param_norm, 5)), verbose)
                VP.print("Current perplexity: " + str(round(cur_entropy, 5)), verbose)

            # Step 2.1: E-step:
            # pool = mp.Pool(n_jobs)

            VP.print("Running E-step ...", verbose)

            param_dic_list = [{'click_vec': click_mat[i, :],
                               'cur_list_pos': item_order[i, :],
                               'var_dic': pred,
                               'item_order': item_order[i, :],
                               'model_def': model_def,
                               'i': i,
                               'session_count': no_sessions}
                              for i in range(no_sessions)]

            # Use for debugging (same as above, but not paralized)
            all_marginal_dat = [GCM._compute_marginals_IO_HMM(param_dic_list[i])
                                for i in np.arange(no_sessions)]

            # Compute all the marginal probabilities
            # TODO: USE MAP INSTEAD, OTHERWISE WON'T BE OF MUCH USE...
            # all_marginal_dat = [pool.apply(GCM._compute_marginals_IO_HMM,
            #                                args=(click_mat[i, :],
            #                                 parameter_dic,
            #                                 list_size,
            #                                 item_order[i, :],
            #                                 model_def,
            #                                 i)) for i in np.arange(no_sessions)]

            # pool.close()
            # pool.join()

            # Format the marginal probabilities as weights
            weight_dic, click_prob = GCM._format_weights_and_covariates(all_marginal_dat, model_def)

            # # Compute entropy using the zeta vector (state probability vector)
            cur_entropy = log_loss(click_mat.flatten(), click_prob.flatten())
            cond_entropy.append(cur_entropy)

            VP.print("Current conditional entropy:" + str(round(cond_entropy[it], 5)), verbose)
            VP.print("Running M-step ...", verbose)
            #
            # # M-step (since keras already paralizes, I do not):
            var_models = GCM._optimize_params(var_models, weight_dic, var_dic, verbose)

            it += 1

        return var_models, pred, cond_entropy

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
        y_true_f = Kback.clip(Kback.flatten(y_true), smooth, 1 - smooth)
        y_pred_f = Kback.clip(Kback.flatten(y_pred), smooth, 1 - smooth)

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

        zeta_mat = np.vstack(zeta_lst)
        return weight_dic, zeta_mat[:, 1:]  # Remove time 0

    @staticmethod
    def _get_param_weights_in_session(param_dic):
        cur_list_pos = param_dic['cur_list_pos']
        marginal_dic = param_dic['marginal_dic']
        model_def = param_dic['model_def']
        session_count = param_dic['session_count']
        cur_session = param_dic['cur_session']

        click_prob = (marginal_dic['zeta_vec'] *
                          np.eye(model_def.no_states, model_def.list_size + 1)).sum(-1)

        weight_vectors = {}
        weight_dic = marginal_dic['reg_out']

        for var_key, feat_type in model_def.var_type.items():
            if feat_type == "item":
                # First column are the positives items, second are the negatives
                weight_vectors[var_key] = np.hstack((np.put(np.zeros(model_def.no_items), cur_list_pos,
                                                            weight_dic[var_key][0]).reshape(-1, 1),
                                                     np.put(np.zeros(model_def.no_items), cur_list_pos,
                                                            weight_dic[var_key][1]).reshape(-1, 1)))

            elif feat_type == "session":
                weight_vectors[var_key] = np.hstack((np.put(np.zeros(session_count), cur_session,
                                                            weight_dic[var_key][0].sum()),
                                                     np.put(np.zeros(session_count), cur_session,
                                                            weight_dic[var_key][1].sum())))

            elif feat_type == "state":
                weight_vectors[var_key]

            else:
                raise NotImplementedError("Only 'session' and 'item' variables are currently implemented")

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
            X = np.vstack((var_dic[var_name], var_dic[var_name]))
            model = var_models[var_name]

            # Note that since we first double the number of rows, division by 2 to always results in a natural number
            pred[var_name] = model.predict(X[:(int)(X.shape[0]/2), :]).flatten()

            # Replace possible NaN-values by 0s
            pred[var_name] = np.nan_to_num(pred[var_name], nan=0)

        return pred

    @staticmethod
    def _optimize_params(var_models, weight_dic, var_dic, verbose):
        # Procedure that finds the next parameters, based on the current E-step
        callback = EarlyStopping(monitor='loss', patience=5)

        for var_name, k_model in var_models.items():
            X = np.vstack((var_dic[var_name], var_dic[var_name]))
            model = var_models[var_name]
            output_dim = model.layers[len(model.layers) - 1].output.shape[1]
            trainable = len(model.trainable_weights) > 0

            if trainable:
                if output_dim == 1:
                    Y = weight_dic[var_name].flatten(order='F')  # column-wise flatten (row-wise is the default)
                else:
                    Y = weight_dic[var_name].T.reshape((X.shape[0], output_dim))

                model.fit(X, Y, batch_size=Y.shape[0], epochs=100, verbose=verbose, callbacks=[callback])
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
                np.put(cur_vec_minus, item_order, np.sum(W_plus, axis=(0, 1)))

                marginals[var_key] = np.hstack((cur_vec_plus.reshape(-1, 1), cur_vec_minus.reshape(-1, 1)))

            elif model_def.var_type[var_key] == "session":
                cur_vec_plus = np.zeros(session_count)
                cur_vec_minus = np.zeros(session_count)
                np.put(cur_vec_plus, i, np.sum(W_plus)).reshape(-1, 1)
                np.put(cur_vec_minus, i, np.sum(W_minus)).reshape(-1, 1)

                marginals[var_key] = np.hstack((cur_vec_plus.reshape(-1, 1), cur_vec_minus.reshape(-1, 1)))

            elif model_def.var_type[var_key] == "state": # TODO: Check, is this axis correct???, is the way the matrix is flatten correct?
                marginals[var_key] = np.hstack((np.sum(W_plus, axis=2).flatten().reshape(-1, 1),
                                                np.sum(W_minus, axis=2).flatten().reshape(-1, 1)))
            elif model_def.var_type[var_key] == "pos":
                marginals[var_key] = np.hstack((np.sum(W_plus, axis=(0, 1)).reshape(-1, 1),
                                                np.sum(W_minus, axis=(0, 1)).reshape(-1, 1)))

        return {'reg_out': marginals, 'zeta_vec': zeta_vec}

    @staticmethod
    def _compute_IO_HMM_est(click_vec, var_dic, item_order, model_def, i):
        # Determine all sessions weights and the state probabilities (zeta vector)
        trans_matrices = GCM._get_trans_mat(model_def, var_dic, item_order, i)
        x_init_state = model_def.init_state
        list_size = model_def.list_size

        # The forward-backward algorithm:
        # at t=0:
        B = np.zeros((model_def.no_states, list_size + 1))
        A = np.zeros((model_def.no_states, list_size + 1))
        H = np.zeros((model_def.no_states, model_def.no_states, list_size))
        zeta = np.zeros((model_def.no_states, list_size + 1))

        click_states = model_def.click_states

        # States where there is a click by definition:
        B[:, list_size] = click_vec[list_size - 1] * click_states[:, list_size] + (1 - click_vec[list_size - 1]) * (
                1 - click_states[:, list_size])
        A[x_init_state, 0] = 1  # We assume that at time t=0 there is a click
        zeta[x_init_state, 0] = 1
        for t in range(1, list_size+1):
            zeta[:, t] = np.dot(trans_matrices[t - 1], zeta[:, t - 1].T)
            # Just as due to some numerical unstability the sum might not always be 1:
            zeta[:, t] = zeta[:, t] / np.sum(zeta[:, t])

            # Note that the click vector itself does not have the 0 state, so the index is one behind
            A[:, t] = click_vec[t-1] * click_states[:, t] * np.dot(trans_matrices[t - 1], A[:, t - 1].T) + \
                      (1 - click_vec[t-1]) * (1 - click_states[:, t]) * np.dot(trans_matrices[t - 1], A[:, t - 1].T)
        for t in reversed(range(1, list_size+1)):
            H[:, :, t - 1] = np.outer(B[:, t], A[:, t - 1]) / np.sum(A[:, list_size - 1]) * trans_matrices[t - 1]

            # Just as due to some numerical instability the sum might not always be 1:
            H[:, :, t - 1] = H[:, :, t - 1] / np.sum(H[:, :, t - 1])
            if t > 1:
                B[:, t - 1] = click_vec[t - 2] * click_states[:, t-1] * np.dot(trans_matrices[t - 1].T, B[:, t].T) + \
                              (1 - click_vec[t - 2]) * (1 - click_states[:, t-1]) * np.dot(trans_matrices[t - 1].T, B[:, t].T)
            else:  # We assume state 0 is clicked
                B[:, t-1] = click_states[:, t-1] * np.dot(trans_matrices[t - 1].T, B[:, t].T)

        return H, zeta

    @staticmethod
    def _get_trans_mat(md, vars_dic, item_order, i):
        # Initialize the M matrices:
        trans_matrices = []

        # Just to ensure it is a transition matrix, only the click state (= initial state) is omitted
        for t in range(md.list_size):
            trans_mat = np.ones((md.no_states, md.no_states))
            for var_name, act_mat in md.act_matrices.items():
                if var_name in md.t0_fixed and t == 0:
                    cur_param = md.t0_fixed[var_name]
                elif md.var_type[var_name] == 'item':
                    cur_param = vars_dic[var_name][item_order[t]]
                elif md.var_type[var_name] == 'session':
                    cur_param = vars_dic[var_name][i]
                elif md.var_type[var_name] == 'state':
                    cur_param = vars_dic[var_name].reshape((md.no_states, md.no_states)).T
                elif md.var_type[var_name] == "pos":
                    size_statespace = md.no_states ** 2
                    cur_param = vars_dic[var_name][size_statespace * t: size_statespace * (t+1)]\
                        .reshape((md.no_states, md.no_states))
                else:
                    raise KeyError("Parameter type " + str(md.var_type[var_name]) +
                                   " is not supported. Supported types: 'item', 'session' and 'pos'")

                update = cur_param * (act_mat['pos_mat'] - act_mat['neg_mat']) \
                    + act_mat['neg_mat'] + act_mat['fixed_mat']

                # I.e., overlapping + new updates + old updates
                trans_mat = trans_mat * update

            # try:
            #     np.testing.assert_array_almost_equal(np.sum(trans_mat, axis=1), expected_sums)
            # except AssertionError as e:
            #     raise ValueError("Probabilities in transition matrix at time: " + str(t) + ", session: " + str(i) +
            #                      ", do not sum to one. Assertion error message: " + str(e))

            # As final step, absorb everything in the absorbing state
            # trans_mat[:, md.no_states - 1] = 1 - expected_sums

            row_sums = np.sum(trans_mat, axis=1)

            # Check if less than 1
            try:
                np.testing.assert_array_less(row_sums, np.ones(md.no_states))
            except AssertionError as e:
                raise ValueError("Probabilities in transition matrix at time: " + str(t) + ", session: " + str(i) +
                                 ", exceed one. Assertion error message: " + str(e))

            for i in range(md.no_states):
                trans_mat[md.absorbing_state[i]] = 1 - row_sums[i]

            # Normalize (to avoid small errors):
            # trans_mat = trans_mat * np.tile(1/np.sum(trans_mat, axis=1), (md.no_states, 1)).T
            # trans_mat = np.nan_to_num(trans_mat, nan=0) # Replace nan's by 0

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

