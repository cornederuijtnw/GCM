import yaml
import numpy as np


class ClickDefinition:
    """
    Class used to initialize the click model using the input YAML file
    """
    def __init__(self, loc, trans_matrix_def_func):
        """

        :param loc: Location of the YAML file
        :param trans_matrix_def_func: Reference to the function defining the transition matrix
        """
        self._loc = loc
        self._act_matrices = {}
        self._param_size = {}
        self._var_type = {}

        with open(self._loc) as f:
            definitions = yaml.load(f)

            self._click_state = definitions['click_state']
            self._non_click_state = definitions['non_click_state']
            self._no_items = definitions['no_items']
            self._list_size = definitions['list_size']
            self._no_states = definitions['no_states']
            self._init_state = definitions['init_state']
            self._batch_size = definitions['batch_size']
            self._init_act_matrices(definitions['var'])
            self._trans_matrix_def_func = trans_matrix_def_func

    @property
    def batch_size(self):
        """
        Batch size used during optimization (in # sessions)
        """
        return self._batch_size

    @property
    def param_shapes(self):
        """
        Dictionary of shapes, one for each variable in the click model
        """
        return self._param_size

    @property
    def var_type(self):
        """
        Type of each variable (currently implemented: session and item)
        """
        return self._var_type

    @property
    def click_state(self):
        """
        The state at which an item is clicked
        """
        return self._click_state

    @property
    def non_click_state(self):
        """
        The state at which an item is skiped but evaluated
        """
        return self._non_click_state

    @property
    def init_state(self):
        """
        Initial state of the model during the start of each session
        """
        return self._init_state

    @property
    def no_items(self):
        """
        Total number of items in the model
        """
        return self._no_items

    @property
    def list_size(self):
        """
        SERP size
        """
        return self._list_size

    @property
    def no_states(self):
        """
        Number of states used to define the click model
        """
        return self._no_states

    @property
    def act_matrices(self):
        """
        A dictionary of activation matrices, one for each variable
        """
        return self._act_matrices

    @property
    def trans_matrix_def_func(self):
        """
        Reference to the function defining the transition matrix
        """
        return self._trans_matrix_def_func

    def _init_act_matrices(self, act_mat_dic):
        # Initializes the activation matrices, parameters sizes and variable types
        for key, d in act_mat_dic.items():
            self._act_matrices[key] = [self._init_act_mat(d['plus']), self._init_act_mat(d['minus'])]
            self._param_size[key] = d['param_size']
            self._var_type[key] = d['var_type']

    def _init_act_mat(self, d):
        # Initialized a single activation matrix
        act_mat = np.zeros((self._no_states, self._no_states))
        init_mat = np.zeros((self._no_states, self._no_states))  # Transition from 0 to 1 may be different

        for row_val in d['subsequent']:
            act_mat[row_val[0], :] = np.array(row_val[1])
        if 'init' in d:
            for row_val in d['init']:
                init_mat[row_val[0], :] = np.array(row_val[1])

        act_mat = np.concatenate((init_mat.reshape((self._no_states, self._no_states, 1)),
                                  np.tile(act_mat.T.reshape((self._no_states, self._no_states, 1)),
                                      (1, 1, self._list_size - 1))), axis=2)

        return act_mat






