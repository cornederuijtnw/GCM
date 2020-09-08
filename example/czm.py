import numpy as np
import pandas as pd
from scripts.clickdefinitionreader import ClickDefinition
from scripts.GCM import GCM
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


class CZMModel:
    @staticmethod
    def get_trans_mat(model_definitions, vars_dic, item_order):
        no_states = model_definitions.no_states
        list_size = model_definitions.list_size

        # Initialize the M matrices:
        trans_matrices = []

        trans_mat = np.zeros((no_states, no_states))

        # Note that we omit state 3, the click state
        trans_mat[0, no_states - 1] = 1
        trans_mat[1, no_states - 1] = 1
        trans_mat[2, no_states - 1] = 1
        trans_mat[4, no_states - 1] = 1
        trans_mat[5, no_states - 1] = 1
        trans_mat[6, no_states - 1] = 1

        # Evaluation states:
        trans_mat[3, 2] = 1 - vars_dic['phi_A'][item_order[0]]
        trans_mat[3, 3] = vars_dic['phi_A'][item_order[0]]

        # Bit of a silly fix, but the matrix should be defined in transpose (i.e, the 'from' is on the columns)
        trans_mat = trans_mat.T
        trans_matrices.append(trans_mat)

        for t in range(1, list_size):
            trans_mat = np.zeros((no_states, no_states))

            # absorbing states (note, state 6 is the absorbing state), the other are for product separability:
            trans_mat[0, no_states - 1] = 1
            trans_mat[1, no_states - 1] = 1
            trans_mat[4, no_states - 1] = 1
            trans_mat[5, no_states - 1] = 1
            trans_mat[6, no_states - 1] = 1

            # Evaluation states:
            trans_mat[2, 0] = (1 - vars_dic['gamma'][t]) * (1 - vars_dic['phi_A'][item_order[t]])
            trans_mat[2, 1] = (1 - vars_dic['gamma'][t]) * vars_dic['phi_A'][item_order[t]]
            trans_mat[2, 2] = vars_dic['gamma'][t] * (1 - vars_dic['phi_A'][item_order[t]])
            trans_mat[2, 3] = vars_dic['gamma'][t] * vars_dic['phi_A'][item_order[t]]

            # Others are 0
            trans_mat[3, 0] = (1 - vars_dic['gamma'][t]) * (1 - vars_dic['phi_A'][item_order[t]]) * (
                    1 - vars_dic['phi_S'][item_order[t - 1]])
            trans_mat[3, 1] = (1 - vars_dic['gamma'][t]) * vars_dic['phi_A'][item_order[t]] * (
                    1 - vars_dic['phi_S'][item_order[t - 1]])
            trans_mat[3, 2] = vars_dic['gamma'][t] * (1 - vars_dic['phi_A'][item_order[t]]) * (
                    1 - vars_dic['phi_S'][item_order[t - 1]])
            trans_mat[3, 3] = vars_dic['gamma'][t] * vars_dic['phi_A'][item_order[t]] * (
                    1 - vars_dic['phi_S'][item_order[t - 1]])
            trans_mat[3, 4] = (1 - vars_dic['phi_A'][item_order[t]]) * vars_dic['phi_S'][item_order[t - 1]]
            trans_mat[3, 5] = vars_dic['phi_A'][item_order[t]] * vars_dic['phi_S'][item_order[t - 1]]

            # Bit of a silly fix, but the matrix should be defined in transpose (i.e, the 'from' is on the columns)
            trans_mat = trans_mat.T
            trans_matrices.append(trans_mat)

        return trans_matrices


if __name__ == "__main__":
    yaml_file_loc = "./models/czmdefinitions.yaml"

    click_data = pd.read_csv("./data/small_example/simulation_res_train.csv", index_col=False)
    prod_position = pd.read_csv("./data/small_example/simulation_item_props.csv", index_col=False)

    model_def = ClickDefinition(yaml_file_loc, CZMModel.get_trans_mat)

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