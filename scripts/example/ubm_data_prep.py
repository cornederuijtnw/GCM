import pandas as pd
from scripts.clickmodel_fitters.clickdefinitionreader import ClickDefinition
import numpy as np
import pickle as pl


if __name__ == "__main__":
    yaml_file_loc = "./model_definitions/ubm_definitions.yaml"
    var_dic_store_loc = "./data/small_example/ubm_var_dics.pl"

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
        .pivot(index='session', columns='item_order', values='click')

    item_pos_mat = click_data.loc[:, ['session', 'item_order', 'item']] \
        .pivot(index='session', columns='item_order', values='item')

    # Ensure the order is correct
    item_feature_mat_A = pd.get_dummies(click_data['item']
                                        .sort_values()
                                        .unique())

    pospos_feature_mat_gamma = np.vstack([np.tile(np.eye(model_def.list_size, 1, -k), model_def.list_size).T
                                for k in range(model_def.list_size)]).T

    pos_feature_mat_tau = np.eye(model_def.list_size)

    var_dic = {'phi_A': item_feature_mat_A, 'gamma': pospos_feature_mat_gamma, 'tau': pos_feature_mat_tau}

    var_dic_file = open(var_dic_store_loc, "wb")
    pl.dump(var_dic, var_dic_file)
    var_dic_file.close()

    # Convert colum names to strings for parquet format:
    click_mat.columns = click_mat.columns.astype(str)
    item_pos_mat.columns = item_pos_mat.columns.astype(str)

    click_mat.to_parquet("./data/small_example/click_mat.parquet.gzip", compression="gzip")
    item_pos_mat.to_parquet("./data/small_example/item_pos.parquet.gzip", compression="gzip")

    # gamma_feature_mat = np.ones((sessions, 1))
    #
    # model_phi_A = Sequential()
    # model_phi_A.add(Dense(1, input_dim=item_feature_mat_A.shape[1], activation='sigmoid', use_bias=False))
    # model_phi_A.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())
    #
    # model_phi_S = Sequential()
    # model_phi_S.add(Dense(1, input_dim=item_feature_mat_S.shape[1], activation='sigmoid', use_bias=False))
    # model_phi_S.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())
    #
    # model_gamma = Sequential()
    # model_gamma.add(Dense(1, input_dim=gamma_feature_mat.shape[1], activation='sigmoid', use_bias=False))
    # model_gamma.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())
    #
    # var_dic = {'phi_A': item_feature_mat_A, 'phi_S': item_feature_mat_S, 'gamma': gamma_feature_mat}
    # var_models = {'phi_A': model_phi_A, 'phi_S': model_phi_S, 'gamma': model_gamma}