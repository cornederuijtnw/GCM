import numpy as np
import pickle as pl
import pandas as pd
from scripts.clickmodel_fitters.clickdefinitionreader import ClickDefinition
from scripts.clickmodel_fitters.GCM import GCM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adagrad
from tensorflow.keras.layers import RepeatVector
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import GroupKFold


if __name__ == "__main__":
    click_data = pd.read_csv("./data/small_example/simulation_res_train.csv", index_col=False)
    prod_position = pd.read_csv("./data/small_example/simulation_item_props.csv", index_col=False)

    list_size = 10
    no_states = 7
    click_states = np.zeros((no_states, list_size + 1))
    click_states[3, :] = 1
    abs_state = [(i, no_states-1) for i in range(no_states)]
    init_state = 2  # The click state goes to an absorbing state, so use evaluated but not attracted instead
    batch_size = 10000
    no_items = 100

    var_dic = {
        'phi_A': {
            'var_type': 'item',
            'pos_mat': np.vstack((np.zeros((2, no_states)),
                                  np.array([0, 0, 0, 1, 0, 0, 0]),
                                  np.array([0, 0, 0, 1, 0, 1, 0]),
                                  np.zeros((3, no_states)))),
            'neg_mat': np.vstack((np.zeros((2, no_states)),
                                  np.array([0, 0, 1, 0, 0, 0, 0]),
                                  np.array([0, 0, 1, 0, 1, 0, 0]),
                                  np.zeros((3, no_states)))),
        },
        'gamma': {
            'var_type': 'pos',
            't0_fixed': 1,
            'pos_mat': np.vstack((np.zeros((3, no_states)),
                                  np.array([0, 0, 1, 1, 0, 0, 0]),
                                  np.zeros((3, no_states)))),
            'neg_mat': np.vstack((np.zeros((3, no_states)),
                                  np.array([0, 0, 0, 0, 1, 1, 0]),
                                  np.zeros((3, no_states)))),
            'fixed_mat': np.vstack((np.zeros((2, no_states)),
                                  np.array([0, 0, 1, 1, 0, 0, 0]),
                                  np.zeros((4, no_states))))
        }
    }

    model_def = ClickDefinition(click_states, init_state, list_size, no_states, batch_size, no_items, abs_state,
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

    # Get click matrix and item-position matrix
    click_mat = click_data.loc[:, ['session', 'item_order', 'click']] \
        .pivot(index='session', columns='item_order', values='click') \
        .to_numpy()

    item_pos_mat = click_data.loc[:, ['session', 'item_order', 'item']] \
        .pivot(index='session', columns='item_order', values='item') \
        .to_numpy()

    # Ensure that the order is correct
    item_feature_mat_A = pd.get_dummies(click_data['item']
                                        .sort_values()
                                        .unique())

    item_feature_mat_gamma = np.eye(list_size)

    model_phi_A = Sequential()
    model_phi_A.add(Dense(1, input_dim=item_feature_mat_A.shape[1], activation='sigmoid', use_bias=False))
    model_phi_A.compile(loss=GCM.pos_log_loss, optimizer=Adagrad())

    model_gamma = Sequential()
    # First compute the kernel
    model_gamma.add(Dense(1, input_dim=item_feature_mat_gamma.shape[1], activation='sigmoid', use_bias=False))
    model_gamma.add(RepeatVector(no_states**2))
    model_gamma.compile(loss=GCM.pos_log_loss, optimizer=Adagrad())

    var_dic = {'phi_A': item_feature_mat_A, 'gamma': item_feature_mat_gamma}
    var_models = {'phi_A': model_phi_A, 'gamma': model_gamma}

    res = GCM.runEM(click_mat, var_dic, var_models, item_pos_mat, model_def, verbose=True, earlystop_patience=10,
                    n_jobs=1)

    pl.dump(res[1], open("./data/small_example/state_prob.pl", "wb"))
    pl.dump(res[2], open("./data/small_example/convergence.pl", "wb"))
    pl.dump(res[3], open("./data/small_example/click_probs.pl", "wb"))

    print(res[2])
