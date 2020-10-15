import pandas as pd
from scripts.clickmodel_fitters.clickdefinitionreader import ClickDefinition
import numpy as np
import pickle as pl


if __name__ == "__main__":
    var_dic_store_loc = "./data/small_example/ubm_var_dics.pl"
    click_def_loc = "./data/small_example/click_def_loc.pl"

    click_data = pd.read_csv("./data/small_example/simulation_res_train.csv", index_col=False)
    prod_position = pd.read_csv("./data/small_example/simulation_item_props.csv", index_col=False)

    # __init__(self, click_states, init_state, list_size, no_states, batch_size, no_items, abs_states, var_dic)
    list_size = 10
    no_states = 11
    ubm_click_states = np.eye(list_size + 1, list_size + 1)
    abs_state = [(i, i) for i in range(no_states)]
    init_state = 0
    batch_size = 500
    no_items = 10

    var_dic = {
        'phi_A': {
            'var_type': 'item',
            'pos_mat': np.triu(np.ones((list_size + 1, list_size + 1)), k=1)
        },
        'gamma': {
            'var_type': 'state',
            'pos_mat': np.triu(np.ones((list_size + 1, list_size + 1)), k=1)
        },
        'tau': {
            'var_type': 'pos',
            'pos_mat': np.triu(np.ones((list_size + 1, list_size + 1)), k=1)
        }
    }

    model_def = ClickDefinition(ubm_click_states, init_state, list_size, no_states, batch_size, no_items, abs_state,
                                var_dic)

    # Make sure order is correct:
    click_data = click_data.sort_values(['user_id', 'session_count', 'item_order'])

    # Add session index:
    session_index = click_data.loc[:, ['user_id', 'session_count']] \
        .drop_duplicates() \
        .reset_index()
    session_index['session'] = session_index.index.to_numpy()

    click_data = click_data \
        .set_index(['user_id', 'session_count']) \
        .join(session_index
              .set_index(['user_id', 'session_count']),
              on=['user_id', 'session_count']) \
        .reset_index()

    # Sample first 500 sessions:
    click_data = click_data.loc[click_data['session'] < model_def.batch_size, :]

    sessions = click_data['session'].nunique()

    # Get click matrix and item-position matrix
    click_mat = click_data.loc[:, ['session', 'item_order', 'click']] \
        .pivot(index='session', columns='item_order', values='click')

    item_pos_mat = click_data.loc[:, ['session', 'item_order', 'item']] \
        .pivot(index='session', columns='item_order', values='item')

    # Ensure the order is correct
    item_feature_mat_A = pd.get_dummies(click_data['item']
                                        .sort_values()
                                        .unique())

    pos_feature_mat_tau_lst = []

    pos_feature_mat_tau = np.vstack([np.hstack((np.zeros((model_def.list_size + 1, k1)),
                np.hstack((np.ones(k1), np.zeros(model_def.list_size - k1 + 1))).reshape(-1, 1),
                np.zeros((model_def.list_size + 1, model_def.list_size - k1)))).flatten()
     for k1 in range(1, model_def.list_size+1)])

    pos_feature_gamma = np.eye(model_def.list_size+1, model_def.list_size + 1)
    pos_feature_gamma[:, 0] = 0

    var_dic = {'phi_A': item_feature_mat_A, 'gamma': pos_feature_gamma, 'tau': pos_feature_mat_tau}

    var_dic_file = open(var_dic_store_loc, "wb")
    pl.dump(var_dic, var_dic_file)
    var_dic_file.close()

    # Convert colum names to strings for parquet format:
    click_mat.columns = click_mat.columns.astype(str)
    item_pos_mat.columns = item_pos_mat.columns.astype(str)

    click_mat.to_parquet("./data/small_example/click_mat.parquet.gzip", compression="gzip")
    item_pos_mat.to_parquet("./data/small_example/item_pos.parquet.gzip", compression="gzip")

    # Store click model definitions:
    pl.dump(model_def, open(click_def_loc, "wb"))
