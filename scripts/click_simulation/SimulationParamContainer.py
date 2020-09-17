import yaml


class SimulationParamContainer:
    """
    Containing for simulation parameters
    """
    def __init__(self, items, users, list_size, user_distance_sensitivity, user_lifetime_phases,
                 device_initial_probabilities, device_transition_probabilities, user_arrival_rate,
                 inter_session_pareto_shape, cookie_churn_rates, cookie_churn_rate_probability, salience_att,
                 salience_satis, warm_up_sessions, eval_time_cutoff1, eval_time_cutoff2, cont_param):
        self._items = items
        self._users = users
        self._list_size = list_size
        self._user_distance_sensitivity = user_distance_sensitivity
        self._user_lifetime_phases = user_lifetime_phases
        self._device_initial_probabilities = device_initial_probabilities
        self._device_transition_probabilities = device_transition_probabilities
        self._user_arrival_rate = user_arrival_rate
        self._inter_session_pareto_shape = inter_session_pareto_shape
        self._cookie_churn_rates = cookie_churn_rates
        self._cookie_churn_rate_probability = cookie_churn_rate_probability
        self._salience_att = salience_att
        self._salience_satis = salience_satis
        self._warm_up_sessions = warm_up_sessions
        self._eval_time_cutoff1 = eval_time_cutoff1
        self._eval_time_cutoff2 = eval_time_cutoff2
        self._cont_param = cont_param

    @staticmethod
    def from_yaml(yaml_file_loc):
        with open(yaml_file_loc) as f:
            defi = yaml.load(f)

            # model_lifetime_phases
            if not bool(defi['has_cookie_param']):
                sim_container = SimulationParamContainer(defi['no_items'],
                                                         defi['no_users'],
                                                         defi['list_size'],
                                                         defi['user_dist_sens'],
                                                         defi['model_lifetime_phases'],
                                                         None,  # device_initial_probabilities
                                                         None,  # device_transition_probabilities
                                                         defi['user_arr_rate'],
                                                         None,  # inter_session_pareto_shape
                                                         None,  # cookie_churn_rates
                                                         None,  # cookie_churn_rate_probability
                                                         defi['salience_att'],
                                                         defi['salience_satis'],
                                                         defi['warm_up_sessions'],
                                                         defi['eval_cut_off1'],
                                                         defi['eval_cut_off2'],
                                                         defi['cont_param'])
            else:
                sim_container = SimulationParamContainer(defi['no_items'],
                                                         defi['no_users'],
                                                         defi['list_size'],
                                                         defi['user_dist_sens'],
                                                         defi['user_lifetime_phases'],
                                                         defi['device_init_probab'],
                                                         defi['device_trans_probab'],
                                                         defi['user_arr_rate'],
                                                         defi['inter_session_pareto_shape'],
                                                         defi['cookie_churn_rates'],
                                                         defi['cookie_churn_rate_prob'],
                                                         defi['salience_att'],
                                                         defi['salience_satis'],
                                                         defi['warm_up_sessions'],
                                                         defi['eval_cut_off1'],
                                                         defi['eval_cut_off2'],
                                                         defi['cont_param'])
        return sim_container

    @property
    def items(self):
        """
        Total number of items a user can click on
        """
        return self._items

    @ items.setter
    def items(self, items):
        self._items = items

    @property
    def users(self):
        """
        Total number of simulated users
        """
        return self._users

    @users.setter
    def users(self, users):
        self._users = users

    @property
    def list_size(self):
        """
        Number of items shown to a user
        """
        return self._list_size

    @list_size.setter
    def list_size(self, list_size):
        self._list_size = list_size

    @property
    def user_distance_sensitivity(self):
        """
        Determines the probability of a user clicking on an item that is different from its own location (large =
        more likely)
        """
        return self._user_distance_sensitivity

    @user_distance_sensitivity.setter
    def user_distance_sensitivity(self, user_distance_sensitivity):
        self._user_distance_sensitivity = user_distance_sensitivity

    @property
    def user_lifetime_phases(self):
        """
        Parameter used to determine the lifetime of a user (number of hyper-exponential draws that sum up to the
        user lifetime)
        """
        return self._user_lifetime_phases

    @user_lifetime_phases.setter
    def user_lifetime_phases(self, user_lifetime_phases):
        self._user_lifetime_phases = user_lifetime_phases

    @property
    def device_initial_probabilities(self):
        """
        Vector where entry v represents the probability of starting the first session with device v
        """
        return self._device_initial_probabilities

    @device_initial_probabilities.setter
    def device_initial_probabilities(self, device_initial_probabilities):
        self._device_initial_probabilities = device_initial_probabilities

    @property
    def device_transition_probabilities(self):
        """
        Transition matrix with the probabilities of switching the some device v, given the previous session was with
        device w
        """
        return self._device_transition_probabilities

    @device_transition_probabilities.setter
    def device_transition_probabilities(self, device_transition_probabilities):
        self._device_transition_probabilities = device_transition_probabilities

    @property
    def user_arrival_rate(self):
        """
        Average speed at which users arrive at the website (assume Poisson distributed)
        """
        return self._user_arrival_rate

    @user_arrival_rate.setter
    def user_arrival_rate(self, user_arrival_rate):
        self._user_arrival_rate = user_arrival_rate

    @property
    def inter_session_pareto_shape(self):
        """
        Shape parameter of the distribution modelling inter session times (which we assume to be Pareto distributed)
        """
        return self._inter_session_pareto_shape

    @inter_session_pareto_shape.setter
    def inter_session_pareto_shape(self, inter_session_pareto_shape):
        self._inter_session_pareto_shape = inter_session_pareto_shape

    @property
    def cookie_churn_rates(self):
        """
        Hyper-exponential rates determining the probability of a cookie churn within time t
        """
        return self._cookie_churn_rates

    @cookie_churn_rates.setter
    def cookie_churn_rates(self, cookie_churn_rates):
        self._cookie_churn_rates = cookie_churn_rates

    @property
    def cookie_churn_rate_probabilities(self):
        """
        Hyper-exponential probabilities: probability of the cookie-churn being determined by an exponential
        distribution with rate r
        """
        return self._cookie_churn_rate_probability

    @cookie_churn_rate_probabilities.setter
    def cookie_churn_rate_probabilities(self, cookie_churn_rate_probability):
        self._cookie_churn_rate_probability = cookie_churn_rate_probability

    @property
    def salience_att(self):
        return self._salience_att

    @salience_att.setter
    def salience_att(self, salience_att):
        self._salience_att = salience_att

    @property
    def salience_satis(self):
        return self._salience_satis

    @salience_satis.setter
    def salience_satis(self, salience_satis):
        self._salience_satis = salience_satis

    @property
    def warm_up_sessions(self):
        """Note: this is warm-up session to remove from the training set, not warm-up sessions for the estimation of
        the popularity order"""
        return self._warm_up_sessions

    @warm_up_sessions.setter
    def warm_up_sessions(self, warm_up_sessions):
        self._warm_up_sessions = warm_up_sessions

    @property
    def eval_time_cutoff1(self):
        return self._eval_time_cutoff1

    @eval_time_cutoff1.setter
    def eval_time_cutoff1(self, eval_time_cutoff1):
        self._eval_time_cutoff1 = eval_time_cutoff1

    @property
    def eval_time_cutoff2(self):
        return self._eval_time_cutoff2

    @eval_time_cutoff2.setter
    def eval_time_cutoff2(self, eval_time_cutoff2):
        self._eval_time_cutoff2 = eval_time_cutoff2

    @property
    def cont_param(self):
        return self._cont_param

    @cont_param.setter
    def cont_param(self, cont_param):
        self._cont_param = cont_param


