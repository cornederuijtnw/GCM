import numpy as np
import pandas as pd
from scripts.clickmodel_fitters.clickdefinitionreader import ClickDefinition
from scripts.clickmodel_fitters.GCM import GCM
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


if __name__ == "__main__":
    click_data = pd.read_csv("./data/small_example/simulation_res_train.csv", index_col=False)
    prod_position = pd.read_csv("./data/small_example/simulation_item_props.csv", index_col=False)

    list_size = 10
    no_states = 7
    click_states = np.zeros((no_states + 1, no_states + 1))
    click_states[:, 3] = 1
    abs_state = [(i, 6) for i in range(7)]
    init_state = 3  # Equals the click state
    batch_size = 500
    no_items = 10

    var_dic = {
        'gamma': {
            'var_type': 'session',
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
        .pivot(index='session', columns='item_order', values='click') \
        .to_numpy()

    item_pos_mat = click_data.loc[:, ['session', 'item_order', 'item']] \
        .pivot(index='session', columns='item_order', values='item') \
        .to_numpy()

    # Ensure that the order is correct
    item_feature_mat_A = pd.get_dummies(click_data['item']
                                        .sort_values()
                                        .unique())

    item_feature_mat_S = pd.get_dummies(click_data['item']
                                        .sort_values()
                                        .unique())

    gamma_feature_mat = np.ones((sessions, 1))

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

    res = GCM.runEM(click_mat, var_dic, var_models, item_pos_mat, model_def, verbose=True)

    print(res[2])