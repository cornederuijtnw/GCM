import numpy as np
import pandas as pd
from scripts.clickmodel_fitters.GCM import GCM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import Identity
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import multiply
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.activations import relu
import pickle as pl
import tensorflow.keras.backend as K
import tensorflow.keras.regularizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Activation
import tensorflow as tf


# class LowerDiagWeight(Constraint):
#     """Constrains the weights to be on the lower diagonal.
#     """
#     def __init(self, dummy=1):
#         self._dummy = dummy
#
#     def __call__(self, w):
#         N = K.int_shape(w)[-1]
#         m = K.constant(np.tril(np.ones((N, N))))
#         w = w * m
#
#         return w
#
#     def get_config(self):
#         return {'dummy': self._dummy}


def alt_softmax(x):
    #return x
    # Returns the softmax, but only over the non-zero items
    return tf.exp(x) * tf.cast(tf.math.not_equal(x, 0), tf.float32) / \
           tf.reduce_sum(tf.exp(x) * tf.cast(tf.math.not_equal(x, 0), tf.float32), axis=1, keepdims=True)


def gamma_initializer(shape, dtype=None):
    return K.constant(np.vstack((np.zeros(shape[0]).reshape(1, -1),
                                 np.tril(np.ones((shape - np.array([1, 0])).astype(int))))))


if __name__ == "__main__":
    #tf.compat.v1.disable_eager_execution()

    var_dic_store_loc = "./data/small_example/ubm_var_dics.pl"
    yaml_file_loc = "./model_definitions/ubm_definitions.yaml"
    click_def_loc = "./data/small_example/click_def_loc.pl"

    click_mat = pd.read_parquet("./data/small_example/click_mat.parquet.gzip")
    item_pos_mat = pd.read_parquet("./data/small_example/item_pos.parquet.gzip")

    model_def = pl.load(open(click_def_loc, "rb"))

    var_dic_file = open(var_dic_store_loc, "rb")
    var_dic = pl.load(var_dic_file)
    var_dic_file.close()

    model_phi_A = Sequential()
    model_phi_A.add(Dense(1, input_dim=var_dic['phi_A'].shape[1], activation='sigmoid', use_bias=False))
    model_phi_A.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())

    # Note the large output dimension and the softmax. We want multiple transition probabilities that sum up to 1
    # Its the shape**2, as we flatten the square matrix.
    model_gamma = Sequential()
    # First compute the kernel
    model_gamma.add(Dense(var_dic['gamma'].shape[1], use_bias=False, activation=alt_softmax,
                          kernel_initializer=gamma_initializer, kernel_constraint=LowerDiagWeight()))
    model_gamma.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())

    model_tau = Sequential()
    model_tau.add(Dense(var_dic['tau'].shape[1], input_dim=var_dic['tau'].shape[1], activation=None, use_bias=False,
                        kernel_initializer=Identity(), trainable=False))
    model_tau.compile('rmsprop', 'binary_crossentropy')  # No trainable weights, so doesn't really matter

    var_models = {'phi_A': model_phi_A, 'gamma': model_gamma, 'tau': model_tau}

    res = GCM.runEM(click_mat.to_numpy(), var_dic, var_models, item_pos_mat.to_numpy(), model_def, verbose=True)

    print(res[2])