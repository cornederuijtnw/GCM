import unittest
from scripts.clickmodel_fitters.clickdefinitionreader import ClickDefinition
from scripts.clickmodel_fitters.GCM import GCM
import numpy as np


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

        self._model_def = ClickDefinition(click_states, init_state, list_size, no_states, batch_size, no_items,
                                          abs_state, var_dic)

    def test_FBeq(self):
        item_order = np.array([0, 1, 2])
        click_vector = np.array([1, 0, 0])

        vars_dic = {'phi_A': np.array([0.5, 0.4, 0.3]),
                    'phi_S': np.array([0.9, 0.85, 0.9]),
                    'gamma': np.array([0.7, 0.7, 0.7])}

        GCM._compute_IO_HMM_est(click_vector, vars_dic, item_order, self._model_def, 0)


if __name__ == '__main__':
    unittest.main()
