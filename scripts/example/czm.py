import numpy as np
import pandas as pd
from scripts.clickmodel_fitters.clickdefinitionreader import ClickDefinition
from scripts.clickmodel_fitters.GCM import GCM
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


if __name__ == "__main__":
    yaml_file_loc = "./model_definitions/czmdefinitions.yaml"

    click_data = pd.read_csv("./data/small_example/simulation_res_train.csv", index_col=False)
    prod_position = pd.read_csv("./data/small_example/simulation_item_props.csv", index_col=False)

    model_def = ClickDefinition(yaml_file_loc)

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