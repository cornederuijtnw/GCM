import numpy as np
import math
import time
import numpy.random as rand
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scripts.clickmodel_fitters.SDBN import SDBN
import multiprocessing as mp


# Change to DBN simulator? SCCM names doesn't make sense now?
class DBNSimulator:
    MIN_PROB = 10 ** (-5)  # To avoid numeric problems
    PD_RES_NAMES = ['user_id', 'item', 'item_order', 'click', 'attr', 'satis',
                    'eval', 'session_count']
    SIMULATION_RES = []
    COUNTER = 0

    def __init__(self, param_container):
        self._param_container = param_container
        self._user_pref = None
        self._item_loc = None
        self._distance_mat = None
        self._null_ist = None
        self._awareness_mat = None
        self._null_dist = None
        self._attr_mat = None
        self._satis_mat = None
        self._rand_state = None
        self._has_init_sim = False

    def simulate(self, warm_up_frac):
        """
        Starts the simulation
        :param warm_up_frac: fraction of users used to determine the overall item popularity (which is used as item
        order)
        :param init_sim: True if the attraction and satisfaction matrices have not been computed yet
        :param rand_state: random seed
        :return: The result of the simulation, and a dataframe with information about the different cookies
        """
        t0 = time.time()
        print("Starting simulation")
        if not self._has_init_sim:
            raise ValueError("Initialize the simulation first before running the simulation")

        print("Running warm-up")
        warm_up_users = math.ceil(warm_up_frac * self._param_container.users)
        rand.set_state(self._rand_state.get_state())

        init_rel = np.repeat(1 / self._param_container.items, self._param_container.items)

        warm_up_res = self._run_simulation(warm_up_users, init_rel)

        dur = round(time.time() - t0)
        print("Warm-up finished, time: " + str(dur) + " seconds")

        new_relevance = self._get_warmed_up_satisfaction(warm_up_res)

        print("Running simulation:")
        sim_result = self._run_simulation(self._param_container.users - warm_up_users, new_relevance)

        dur = round(time.time() - t0)
        print("Simulation finished, simulation time: " + str(dur) + " seconds")

        return sim_result

    def _run_simulation(self, users, order_relevance):
        """
        Runs the simulation procedure
        :param users: number of users to simulate for
        :param order_relevance: the weights used to determine the item ordering
        :return: a pandas dataframe with the simulated clicks, and a dataframe containing cookie information
        """

        pool = mp.Pool(mp.cpu_count() - 1)

        sim_res = list(pool.map(SCCMSimulation._single_user_simulation, [{'param_cont': self._param_container,
                                                          'order_rel': order_relevance,
                                                          'u': u,
                                                          'attr_mat': self._attr_mat,
                                                          'satis_mat': self._satis_mat}
                                                         for u in range(users)]))

        # for u in range(users):
        #     cur_proc.append(pool.apply_async(SCCMSimulation._single_user_simulation,
        #                                      [self._param_container,
        #                                       order_relevance,
        #                                       u,
        #                                       self._attr_mat,
        #                                       self._satis_mat],
        #                                      callback=SCCMSimulation._collect_results_eval_pairs_logreg,
        #                                      error_callback=SCCMSimulation._error_callback))

        pool.close()

        pd_all_res = pd.concat(sim_res, axis=0)

        return pd_all_res

    @staticmethod
    def _single_user_simulation(param_dic):
        """
        Simulates clicks for one user
        :param order_pref: preferences which determines the item ordering
        :param user: the user-id to simulate for
        :param arrival_time: time at which the user starts its first session
        :return: a pandas dataframe with the simulated clicks, and a dataframe containing cookie information
        """
        param_container = param_dic['param_cont']
        order_pref = param_dic['order_rel']
        user = param_dic['u']
        attr_mat = param_dic['attr_mat']
        satis_mat = param_dic['satis_mat']

        if user % 100 == 0:
            print("user" + str(user))

        # normalize order_preference:
        order_pref = order_pref / np.sum(order_pref)
        sim_res = pd.DataFrame()

        # Make the initial draws:
        session_count = 0

        user_sessions = rand.geometric(param_container.user_lifetime_phases, 1)[0]

        # Simulate the clicks for this session
        while session_count < user_sessions:
            # Draw the attraction and satisfaction parameters for all positions
            item_order = rand.choice(np.arange(param_container.items), param_container.list_size,
                                     replace=False, p=order_pref)
            real_att = np.array([rand.binomial(1, x, 1) for x in attr_mat[user, item_order]]).reshape(-1)
            real_satis = np.array([rand.binomial(1, x, 1) for x in satis_mat[user, item_order]]).reshape(-1)
            cont_params = rand.binomial(1, param_container.cont_param, param_container.list_size)
            eval_vec = np.zeros(param_container.list_size + 1)
            eval_vec[0] = 1
            click_vec = np.zeros(param_container.list_size)

            # If the user is satisfied by the query, simulate clicks (otherwise all zero)
            for k in range(param_container.list_size):
                click_vec[k] = real_att[k] * eval_vec[k]
                real_satis[k] = click_vec[k] * real_satis[k]
                eval_vec[k + 1] = eval_vec[k] * cont_params[k] * (1-real_satis[k])

            # Add results to dictionary
            ses_sim_res = pd.DataFrame.from_dict(dict(zip(DBNSimulator.PD_RES_NAMES,
                                                          [np.repeat(user, param_container.list_size),
                                                           item_order,
                                                           np.arange(param_container.list_size) + 1,
                                                           click_vec,
                                                           real_att,
                                                           real_satis,
                                                           eval_vec[0:param_container.list_size],
                                                           np.repeat(session_count, param_container.list_size)])))
            sim_res = sim_res.append(ses_sim_res)
            session_count += 1
        return sim_res

    def _get_warmed_up_satisfaction(self, warm_up_res):
        """
        Computes the overall item popularity, which is used to determine the item ordering in the simulation
        """
        unique_lists = warm_up_res.loc[:, ['user_id', 'session_count']].drop_duplicates()
        unique_lists['list_res_id'] = np.arange(unique_lists.shape[0])

        warm_up_res = warm_up_res.\
            set_index(['user_id', 'session_count']).\
            join(unique_lists.set_index(['user_id', 'session_count']), on=['user_id', 'session_count']).\
            reset_index().\
            rename(columns={'item_order': 'pos',
                            'click': 'clicked',
                            'item': 'item_id'}).\
            loc[:, ['list_res_id', 'pos', 'item_id', 'clicked']]

        est_params = SDBN.fit_model(warm_up_res)

        all_items = pd.DataFrame.from_dict({'item_id': np.arange(self._param_container.items),
                                            'dummy': np.repeat(1, self._param_container.items)}).\
            set_index('item_id').\
            join(est_params, on='item_id')

        all_items['att_est'] = all_items['att_est'].fillna(0)
        all_items.loc[all_items['att_est'] < self.MIN_PROB, 'att_est'] = self.MIN_PROB

        return all_items['att_est'].to_numpy()

    @property
    def param_container(self):
        return self._param_container

    def get_item_loc(self):
        """
        Location of the different items, used to determine the probability of a user clicking the item
        """
        if self._item_loc is not None:
            item_prop = pd.DataFrame(self._item_loc)
            item_prop["item"] = item_prop.index.to_numpy()
            item_prop = item_prop.rename({0: 'X0', 1: "X1"}, axis=1)
            return item_prop
        else:
            raise ValueError("The item property matrix was not constructed yet. Run the simulation first")

    def get_distance_mat(self):
        """
        Distances between users and items
        """
        if self._distance_mat is not None:
            dist_mat = pd.DataFrame(self._distance_mat)
            return dist_mat
        else:
            raise ValueError("Class has not been initialized yet. Please initialize first")

    def get_user_loc(self):
        """
        Location of the different users, used to determine the probability of a user clicking the item
        """
        if self._user_pref is not None:
            user_loc = pd.DataFrame(self._user_pref)
            user_loc["user_id"] = user_loc.index.to_numpy()
            user_loc = user_loc.rename({0: 'X0', 1: "X1"}, axis=1)
            return user_loc
        else:
            raise ValueError("The user property matrix was not constructed yet. Run the simulation first")

    def get_attr_mat(self):
        """
         probabilities of a user clicking an item, given the item is observed
         """
        if self._attr_mat is not None:
            att_mat = pd.DataFrame(self._attr_mat)
            return att_mat
        else:
            raise ValueError("The user property matrix was not constructed yet. Run the simulation first")

    def get_satis_mat(self):
        """
         probabilities of a user being satisfied with a clicked item
         """
        if self._satis_mat is not None:
            satis_mat = pd.DataFrame(self._satis_mat)
            return satis_mat
        else:
            raise ValueError("The user property matrix was not constructed yet. Run the simulation first")

    def initialize_sim(self, rand_state):
        """
         Computes the attraction and satisfaction matrices (which are the same), which determine whether
         a user will click on an item
         """
        self._rand_state = rand_state
        rand.set_state(self._rand_state.get_state())
        self._user_pref = self._simulate_user_preference()
        self._item_loc = self._simulate_item_loc()
        self._distance_mat = euclidean_distances(self._user_pref, self._item_loc)
        self._null_dist = np.transpose(
            np.repeat(euclidean_distances([[0, 0]], self._item_loc), self._param_container.users).
            reshape(-1, self._param_container.users))
        # self._awareness_mat = self._get_awareness_mat()
        self._attr_mat = self._get_attractiveness_mat()
        self._satis_mat = self._get_satisfaction_mat()
        # self._satis_mat = self._attr_mat  # As in click-chain model, assume attractiveness = satisfaction prob
        self._has_init_sim = True

    def _simulate_item_loc(self):
        """
        Simulate the item locations (bivariate normal )
        """
        v1 = rand.standard_normal(self._param_container.items)
        v2 = rand.standard_normal(self._param_container.items)
        res = np.hstack([v1.reshape(-1, 1), v2.reshape(-1, 1)])

        return res

    def _simulate_user_preference(self):
        """
        Simulate the user locations (bivariate normal )
        """
        v1 = rand.standard_normal(self._param_container.users)
        v2 = rand.standard_normal(self._param_container.users)
        res = np.hstack([v1.reshape(-1, 1), v2.reshape(-1, 1)])

        return res

    def _get_satisfaction_mat(self):
        """
        Computed the satisfaction matrix
        """
        similarity = np.exp(-self._param_container.user_distance_sensitivity * np.log(self._distance_mat))

        row_sums = np.sum(similarity, axis=1)
        att = (similarity * np.exp(self.param_container.salience_satis))/ \
              (np.repeat(row_sums, self._param_container.items).reshape(-1, self._param_container.items) +
               similarity * (np.exp(self.param_container.salience_satis) - 1))

        # Stability
        att[np.where(att > 1 - self.MIN_PROB)] = 1 - self.MIN_PROB
        att[np.where(att < self.MIN_PROB)] = self.MIN_PROB

        return att

    def _get_attractiveness_mat(self):
        """
        Computes the attractiveness matrix
        """
        similarity = np.exp(-self._param_container.user_distance_sensitivity * np.log(self._distance_mat))

        row_sums = np.sum(similarity, axis=1)
        att = (similarity * np.exp(self.param_container.salience_att))/ \
              (np.repeat(row_sums, self._param_container.items).reshape(-1, self._param_container.items) +
               similarity * (np.exp(self.param_container.salience_att) - 1))

        # Stability
        att[np.where(att > 1 - self.MIN_PROB)] = 1 - self.MIN_PROB
        att[np.where(att < self.MIN_PROB)] = self.MIN_PROB

        return att
