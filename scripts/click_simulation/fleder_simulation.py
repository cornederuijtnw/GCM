import numpy as np
from scripts.click_simulation.dbn_simulation import DBNSimulator
from scripts.click_simulation.SimulationParamContainer import SimulationParamContainer
from scripts.click_simulation.feature_eng import FeatureEngineer


# @staticmethod
# def add_list_id(sim_data):
#     unique_qsessions = sim_data. \
#         groupby(['user_id', 'session_count']). \
#         size(). \
#         rename('freq'). \
#         reset_index(). \
#         sort_values(['user_id', 'session_count']). \
#         drop(['freq'], axis=1)
#
#     unique_qsessions['list_id'] = np.arange(unique_qsessions.shape[0])
#
#     sim_data_with_list_id = sim_data. \
#         set_index(['user_id', 'session_count']). \
#         join(unique_qsessions.set_index(['user_id', 'session_count']),
#              on=['user_id', 'session_count']). \
#         reset_index()
#
#     sim_data_with_list_id = sim_data_with_list_id. \
#         sort_values(['user_id', 'session_count', 'item_order'])
#
#     return sim_data_with_list_id

if __name__ == "__main__":
    test_frac = 0.3
    split_seed = 1992

    sim_data_store_loc = "./data/small_example"

    sim_param_file_loc = "./model_definitions/basecase_simulation.yaml"
    sim_param_cont = SimulationParamContainer.from_yaml(sim_param_file_loc)

    rand_state = np.random.RandomState(split_seed)

    simulator = DBNSimulator(sim_param_cont)
    simulator.initialize_sim(rand_state)

    item_prop_matrix = simulator.get_item_loc()
    user_prop_matrix = simulator.get_user_loc()
    dist_prop_matrix = simulator.get_distance_mat()

    att_mat = simulator.get_attr_mat()
    satis_mat = simulator.get_satis_mat()

    item_prop_matrix.to_csv(sim_data_store_loc + "/simulation_item_props.csv", index=False)
    user_prop_matrix.to_csv(sim_data_store_loc + "/simulation_user_props.csv", index=False)
    dist_prop_matrix.to_csv(sim_data_store_loc + "/simulation_dist_prop.csv", index=False)
    att_mat.to_csv(sim_data_store_loc + "/simulation_attr_mat.csv", index=False)
    satis_mat.to_csv(sim_data_store_loc + "/simulation_satis_mat.csv", index=False)

    sim_result = simulator.simulate(warm_up_frac=0.1)

    # Add ids:
    sim_result = FeatureEngineer.add_list_id(sim_result)

    # Otherwise annoying in feature engineering: list-id would not be informative about the number of query-sessions
    # in the training data. Now we just add a new list_id during feature engineering which re-indexes.
    sim_result = sim_result.rename(columns={'list_id': 'orig_list_id'})

    print("Splitting into train, test user-wise: 70/30 split")
    train, test = \
        FeatureEngineer.split_dataset(sim_result, test_frac, rand_state)

    print("storing results")
    sim_result.to_csv(sim_data_store_loc + "/full_data_set.csv", index=False)
    train.to_csv(sim_data_store_loc + "/simulation_res_train.csv", index=False)
    test.to_csv(sim_data_store_loc + "/simulation_res_test.csv", index=False)


