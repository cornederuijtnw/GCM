import numpy as np
import pandas as pd
from scripts.clickmodel_fitters.clickdefinitionreader import ClickDefinition
from scripts.clickmodel_fitters.GCM import GCM
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.initializers import Identity
import pickle as pl

if __name__ == "__main__":
    var_dic_store_loc = "./data/small_example/ubm_var_dics.pl"
    yaml_file_loc = "./model_definitions/ubm_definitions.yaml"

    click_mat = pd.read_parquet("./data/small_example/click_mat.parquet.gzip")
    item_pos_mat = pd.read_parquet("./data/small_example/item_pos.parquet.gzip")

    model_def = ClickDefinition(yaml_file_loc)

    var_dic_file = open(var_dic_store_loc, "rb")
    var_dic = pl.load(var_dic_file)
    var_dic_file.close()

    model_phi_A = Sequential()
    model_phi_A.add(Dense(1, input_dim=var_dic['phi_A'].shape[1], activation='sigmoid', use_bias=False))
    model_phi_A.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())

    model_gamma = Sequential()
    model_gamma.add(Dense(1, input_dim=var_dic['gamma'].shape[1], activation='sigmoid', use_bias=False))
    model_gamma.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())

    model_tau = Sequential()
    model_tau.add(Dense(1, input_dim=var_dic['tau'].shape[1], activation=None, use_bias=False,
                        kernel_initializer=Identity(), trainable=False))
    model_tau.compile('rmsprop', 'binary_crossentropy')  # No trainable weights, so doesn't really matter

    var_models = {'phi_A': model_phi_A, 'gamma': model_gamma, 'tau': model_tau}

    res = GCM.runEM(click_mat.to_numpy(), var_dic, var_models, item_pos_mat.to_numpy(), model_def, verbose=True)

    print(res[2])