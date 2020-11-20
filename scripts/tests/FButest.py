import unittest
from scripts.clickmodel_fitters.clickdefinitionreader import ClickDefinition
from scripts.clickmodel_fitters.GCM import GCM
import numpy as np
import pickle as pl


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Define the model

        list_size = 3
        no_states = 7
        click_states = np.zeros((no_states, list_size + 1))
        click_states[3, :] = 1
        abs_state = [(i, 6) for i in range(7)]
        init_state = 3  # Equals the click state
        batch_size = 10000
        no_items = 3

        var_dic = {
            'gamma': {
                'var_type': 'session',
                't0_fixed': 1,  # Always continues to evaluate the first item
                'pos_mat': np.vstack((np.zeros((2, no_states)),
                                      np.array([0, 0, 1, 1, 0, 0, 0]),
                                      np.array([0, 0, 1, 1, 0, 0, 0]),
                                      np.zeros((3, no_states)))),
                'neg_mat': np.vstack((np.zeros((2, no_states)),
                                      np.array([1, 1, 0, 0, 0, 0, 0]),
                                      np.array([1, 1, 0, 0, 0, 0, 0]),
                                      np.zeros((3, no_states)))),
                'fixed_mat': np.vstack((np.zeros((3, no_states)),
                                        np.array([0, 0, 0, 0, 1, 1, 0]),
                                        np.zeros((3, no_states))))
            },
            'phi_S': {
                'var_type': 'item',
                't0_fixed': 0,  # Never satisfied in the first state
                'pos_mat': np.vstack((np.zeros((3, no_states)),
                                      np.array([0, 0, 0, 0, 1, 1, 0]),
                                      np.zeros((3, no_states)))),
                'neg_mat': np.vstack((np.zeros((3, no_states)),
                                      np.array([1, 1, 1, 1, 0, 0, 0]),
                                      np.zeros((3, no_states)))),
                'fixed_mat': np.vstack((np.zeros((2, no_states)),
                                        np.array([1, 1, 1, 1, 0, 0, 0]),
                                        np.zeros((4, no_states))))
            },
            'phi_A': {
                'var_type': 'item',
                'pos_mat': np.vstack((np.zeros((2, no_states)),
                                      np.array([0, 1, 0, 1, 0, 0, 0]),
                                      np.array([0, 1, 0, 1, 0, 1, 0]),
                                      np.zeros((3, no_states)))),
                'neg_mat': np.vstack((np.zeros((2, no_states)),
                                      np.array([1, 0, 1, 0, 0, 0, 0]),
                                      np.array([1, 0, 1, 0, 1, 0, 0]),
                                      np.zeros((3, no_states))))
            }
        }

        self._model_def_czm = ClickDefinition(click_states, init_state, list_size, no_states, batch_size, no_items,
                                          abs_state, var_dic)

        # Define the model
        list_size = 3
        no_states = (list_size + 1) * 4 + 1  # +1 is the absorbing state
        ubm_click_states = np.zeros((no_states, list_size + 1))
        ubm_click_states[:-1, :] = np.kron(np.eye(list_size + 1), np.expand_dims(np.array([0, 0, 0, 1]), axis=1))

        # i.e., number of states minus 1, as we have a zero-based index
        abs_state = [(i, (list_size + 1) * 4) for i in range(no_states)]
        init_state = 3  # Corresponding with (0,1,1), i.e., the zero position was evaluated and clicked
        batch_size = 10000
        no_items = 3

        # Positive matrix for A:
        phi_A_diag = np.vstack((np.hstack((np.kron(np.eye(list_size + 1), np.tile(np.array([0, 0, 1, 0]), (4, 1))),
                                           np.zeros((no_states - 1, 1)))),
                                np.zeros(no_states)))

        phi_upper_tr = np.triu(
            np.tile(np.hstack((np.zeros(4), np.tile(np.eye(4)[3], list_size), np.zeros(1))), (no_states, 1)),
            k=4)
        phi_A_pos = phi_A_diag + phi_upper_tr

        # Negative matrix for A:
        phi_A_neg = np.vstack((np.hstack((np.kron(np.eye(list_size + 1), np.tile(np.array([1, 1, 0, 0]), (4, 1))),
                                          np.zeros((no_states - 1, 1)))),
                               np.zeros(no_states)))

        # Positive matrix for gamma:
        gamma_diag = np.vstack((np.hstack((np.kron(np.eye(list_size + 1), np.tile(np.array([0, 1, 0, 0]), (4, 1))),
                                           np.zeros((no_states - 1, 1)))),
                                np.zeros(no_states)))

        gamma_upper_tr = np.triu(
            np.tile(np.hstack((np.zeros(4), np.tile(np.eye(4)[3], list_size), np.zeros(1))), (no_states, 1)), k=4)
        gamma_pos = gamma_diag + gamma_upper_tr

        # Negative matrix for gamma:
        gamma_neg = np.vstack((np.hstack((np.kron(np.eye(list_size + 1), np.tile(np.array([1, 0, 1, 0]), (4, 1))),
                                          np.zeros((no_states - 1, 1)))),
                               np.zeros(no_states)))

        var_dic = {
            'phi_A': {
                'var_type': 'item',
                'pos_mat': phi_A_pos,
                'neg_mat': phi_A_neg
            },
            'gamma': {
                'var_type': 'pos',
                'pos_mat': gamma_pos,
                'neg_mat': gamma_neg
            }
        }

        self._model_def_ubm = ClickDefinition(ubm_click_states, init_state, list_size, no_states, batch_size, no_items,
                                              abs_state, var_dic)

    def test_ubm_transmat_creation(self):
        # Construct the gamma matrix:
        item_order = [0, 1, 2]
        list_size = 3
        no_states = (list_size + 1) * 4 + 1

        np.random.seed(1992)
        A = np.triu(np.random.gamma(1, 1, (list_size + 1) ** 2).reshape(list_size + 1, list_size + 1)) / 10

        # Positive matrix for gamma:
        gamma_diag = np.vstack((np.hstack((np.kron(A, np.tile(np.array([0, 1, 0, 0]), (4, 1))),
                                           np.zeros((no_states - 1, 1)))),
                                np.zeros(no_states)))

        gamma_upper_tr = np.triu(
            np.tile(np.hstack((np.zeros(4), np.tile(A[3], list_size), np.zeros(1))), (no_states, 1)), k=4)

        gamma_pos = gamma_diag + gamma_upper_tr

        # Negative matrix for gamma:
        gamma_neg = np.vstack((np.hstack((np.kron(1 - A, np.tile(np.array([1, 0, 1, 0]), (4, 1))),
                                          np.zeros((no_states - 1, 1)))),
                               np.zeros(no_states)))

        vars_dic = {'phi_A': np.array([0.5, 0.4, 0.3]),
                    'gamma': np.tile((gamma_pos + gamma_neg).flatten(), 3)}

        trans_matrices = GCM._get_trans_mat(self._model_def_ubm, vars_dic, item_order, i=0)

        with open("ubm_trans.pl", "rb") as f:
            trans_matrices_expected = pl.load(f)
        np.testing.assert_allclose(trans_matrices, trans_matrices_expected)

    def test_FBeq(self):
        item_order = np.array([0, 1, 2])
        click_vector = np.array([1, 0, 0])

        vars_dic = {'phi_A': np.array([0.5, 0.4, 0.3]),
                    'phi_S': np.array([0.9, 0.85, 0.9]),
                    'gamma': np.array([0.7, 0.7, 0.7])}

        H, zeta = GCM._compute_IO_HMM_est(click_vector, vars_dic, item_order, self._model_def_czm, 0)

        with open("H_tensor.pl", "rb") as f:
            H_expected = pl.load(f)
        np.testing.assert_allclose(H, H_expected)

        with open("zeta_tensor.pl", "rb") as f:
            zeta_expected = pl.load(f)
        np.testing.assert_allclose(zeta, zeta_expected)




if __name__ == '__main__':
    unittest.main()
