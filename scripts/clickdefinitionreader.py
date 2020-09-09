import yaml
import numpy as np


class ClickDefinition:
    """
    Class used to initialize the click model using the input YAML file.
    """
    def __init__(self, loc):
        """
        :param loc: Location of the YAML file
        """
        self._loc = loc
        self._act_matrices = {}
        self._param_size = {}
        self._var_type = {}

        with open(self._loc) as f:
            definitions = yaml.load(f)

            self._click_state = definitions['click_state']
            self._non_click_state = definitions['skip_state']
            self._list_size = definitions['list_size']
            self._no_states = definitions['no_states']
            self._batch_size = definitions['batch_size']
            self._no_items = definitions['item_size']

            if 't0_fixed' in definitions:
                self._t0_fixed = definitions['t0_fixed']
            else:
                self._t0_fixed = {}

            if 'fixed_params' in definitions:
                self._fixed_params = definitions['fixed_params']
            else:
                self._fixed_params = {}
            self._init_act_matrices(definitions['var'])

    @property
    def no_items(self):
        """
        Number of items distinguished by the model
        """
        return self._no_items

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
    def skip_state(self):
        """
        The state at which an item is skipped but evaluated
        """
        return self._non_click_state

    @property
    def init_state(self):
        """
        Initial state of the model during the start of each session (always equals the click state)
        """
        return self._click_state

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
    def t0_fixed(self):
        """
        A dictionary of parameters being fixed from transition 0 to 1
        """
        return self._t0_fixed

    @property
    def fixed_params(self):
        """
        A dictionary of parameters being fixed, is overwritten by t0_fixed for transitions between 0 and 1
        """
        return self._fixed_params

    def _init_act_matrices(self, act_mat_dic):
        # Initializes the activation matrices, parameters sizes and variable types
        for key, d in act_mat_dic.items():
            if 'fixed_mat' not in d:
                fixed_mat = np.array([])
            else:
                fixed_mat = d['fixed_mat']

            self._act_matrices[key] = {'pos_mat': self._init_act_mat(d['pos_mat']),
                                       'neg_mat': self._init_act_mat(d['neg_mat']),
                                       'fixed_mat': self._init_act_mat(fixed_mat)}
            self._param_size[key] = d['param_size']
            self._var_type[key] = d['var_type']

    def _init_act_mat(self, d):
        # Initialized a single activation matrix
        act_mat = np.zeros((self._no_states, self._no_states))

        for row_val in d:
            act_mat[row_val[0], :] = np.array(row_val[1])

        return act_mat






