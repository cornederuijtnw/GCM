import numpy as np
import pandas as pd
from scripts.clickmodel_fitters.clickdefinitionreader import ClickDefinition
from scripts.clickmodel_fitters.GCM import GCM
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from copy import deepcopy


if __name__ == "__main__":
    # Define the model

    np.random.seed(1992)

    list_size = 10
    no_states = 7
    click_states = np.zeros((no_states, list_size + 1))
    click_states[3, :] = 1
    abs_state = [(i, 6) for i in range(7)]
    init_state = 3  # Equals the click state
    batch_size = 10000
    no_items = 100

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

    model_def = ClickDefinition(click_states, init_state, list_size, no_states, batch_size, no_items, abs_state,
                                var_dic)

    # Load data:
    click_data = pd.read_csv("./data/small_example/simulation_res_train.csv", index_col=False)
    prod_position = pd.read_csv("./data/small_example/simulation_item_props.csv", index_col=False)

    # Ensure the order is correct:
    click_data = click_data.sort_values(['user_id', 'session_count', 'item_order'])

    # Add session index:
    session_index = click_data.loc[:, ['user_id', 'session_count']] \
        .drop_duplicates() \
        .reset_index()
    session_index['session'] = session_index.index.to_numpy()

    click_data = (click_data
                  .set_index(['user_id', 'session_count'])
                  .join(session_index
                        .set_index(['user_id', 'session_count']),
                        on=['user_id', 'session_count'])
                  .reset_index()
                  .set_index('item')
                  .join(prod_position
                        .set_index('item'),
                        on='item')
                  .reset_index()
                  )

    # Create the click matrix and item position matrix
    click_mat = click_data.loc[:, ['session', 'item_order', 'click']] \
        .pivot(index='session', columns='item_order', values='click') \
        .to_numpy()

    item_pos_mat = click_data.loc[:, ['session', 'item_order', 'item']] \
        .pivot(index='session', columns='item_order', values='item') \
        .to_numpy()

    # Create feature matrix for phi_A, phi_S and gamma
    item_feature_mat_A = (click_data.loc[:, ['item', 'X0', 'X1']]
                          .drop_duplicates()
                          .sort_values('item')
                          .to_numpy()
                          )
    item_feature_mat_S = deepcopy(item_feature_mat_A)

    n_sessions = click_data['session'].nunique()
    gamma_feature_mat = np.ones((n_sessions, 1))

    # Define the Keras models
    model_phi_A = Sequential()
    model_phi_A.add(Dense(1, input_dim=item_feature_mat_A.shape[1], activation='sigmoid', use_bias=False))
    model_phi_A.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())

    model_phi_S = Sequential()
    model_phi_S.add(Dense(1, input_dim=item_feature_mat_S.shape[1], activation='sigmoid', use_bias=False))
    model_phi_S.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())

    model_gamma = Sequential()
    model_gamma.add(Dense(1, input_dim=gamma_feature_mat.shape[1], activation='sigmoid', use_bias=False))
    model_gamma.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())

    var_dic = {'phi_A': item_feature_mat_A, 'phi_S': item_feature_mat_S, 'gamma': gamma_feature_mat}
    var_models = {'phi_A': model_phi_A, 'phi_S': model_phi_S, 'gamma': model_gamma}

    # Run model
    res = GCM.runEM(click_mat, var_dic, var_models, item_pos_mat, model_def, verbose=True, n_jobs=1)

    print(res[2])