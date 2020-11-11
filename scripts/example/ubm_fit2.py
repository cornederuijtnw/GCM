import os
import numpy as np
import pandas as pd

from scripts.clickmodel_fitters.clickdefinitionreader import ClickDefinition
from scripts.clickmodel_fitters.GCM import GCM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Identity, Zeros
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import Constraint

# Mask: [np.vstack((np.zeros((t+1, 11)), np.hstack((np.zeros((11-(t+1), t+1)), np.eye(11-(t+1)))))) for t in range(11)]


class SimpleDense(Layer):
    def __init__(self, units=11):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):  # Create the state of the layer (weights)
        T = self.units - 1  # As we exclude the zero state for now
        zero_init = tf.zeros_initializer()

        self._w = tf.Variable(initial_value=zero_init(shape=(int(T * (T + 1) / 2),), dtype='float32'), trainable=True)
        self._zero_col_var = tf.Variable(initial_value=zero_init(shape=(T + 1, 1), dtype='float32'), trainable=False)
        self._zero_row_var = tf.Variable(initial_value=zero_init(shape=(1, T), dtype='float32'), trainable=False)

        # mask:
        t1 = tf.ones((T+1, T+1), dtype="float32")
        t2 = tf.subtract(t1, tf.eye(T+1, dtype="float32"))
        self._wout_mask = tf.multiply(tf.linalg.band_part(t1, 0, -1), t2)

    def call(self, inputs, **kwargs):  # Defines the computation from inputs to outputs
        # print(tf.shape(inputs)[0])
        N = inputs.shape[0]
        T = self.units
        inp_lst = []

        wout = tfp.math.fill_triangular(self._w, upper=True)
        wout = tf.concat([wout, self._zero_row_var], axis=0)
        wout = tf.concat([self._zero_col_var, wout], axis=1)

        for t in range(N):
            cur_input = tf.gather(inputs, t, axis=0)
            cur_input = tf.tile(tf.reshape(cur_input, shape=(1, -1)), [T, 1])
            logis = tf.multiply(cur_input, wout)

            cur_mask = tf.multiply(cur_input, self._wout_mask)

            inp_lst.append(tf.reshape(
                tf.multiply(tf.transpose(tf.transpose(tf.exp(logis)) / tf.reduce_sum(tf.exp(logis), axis=1)),
                              cur_mask), shape=(1, -1)))

        res = tf.concat(inp_lst, axis=0)

        return res

    def get_config(self):
        return {"units": self.units,
                "mask": self._wout_mask}


os.chdir("/Users/cornederuijt/github/GCM/")  # Adjust after construction of the package
np.random.seed(1992)

# Define the model
list_size = 10
no_states = 11
ubm_click_states = np.eye(list_size + 1, list_size + 1)
abs_state = [(i, i) for i in range(no_states)]
init_state = 0
batch_size = 10000
no_items = 100

var_dic = {
    'phi_A': {
        'var_type': 'item',
        'pos_mat': np.triu(np.ones((list_size + 1, list_size + 1)), k=1)
    },
    'gamma': {
        'var_type': 'pos',
        'pos_mat': np.triu(np.ones((list_size + 1, list_size + 1)), k=1)
    }
}

model_def = ClickDefinition(ubm_click_states, init_state, list_size, no_states, batch_size, no_items, abs_state,
                            var_dic)

# Load data:
click_data = pd.read_csv("./data/small_example/simulation_res_train.csv", index_col=False)
prod_position = pd.read_csv("./data/small_example/simulation_item_props.csv", index_col=False)

# Ensure the order is correct:
click_data = click_data.sort_values(['user_id', 'session_count', 'item_order'])

# Add session index:
session_index = (click_data
                 .loc[:, ['user_id', 'session_count']]
                 .drop_duplicates()
                 .reset_index()
                 )

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

# Ensure the order is correct
item_feature_mat_A = (click_data.loc[:, ['item', 'X0', 'X1']]
                                .drop_duplicates()
                                .sort_values('item')
                                .to_numpy())

# pos_feature_mat_tau_lst = []
#
# pos_feature_mat_tau = np.vstack([np.hstack((np.zeros((model_def.list_size + 1, k1)),
#             np.hstack((np.ones(k1), np.zeros(model_def.list_size - k1 + 1))).reshape(-1, 1),
#             np.zeros((model_def.list_size + 1, model_def.list_size - k1)))).flatten()
#  for k1 in range(1, model_def.list_size+1)])

pos_feature_gamma = np.eye(model_def.list_size, model_def.list_size + 1, k=1)

var_dic = {'phi_A': item_feature_mat_A, 'gamma': pos_feature_gamma}

# Model:
model_phi_A = Sequential()
model_phi_A.add(Dense(1, input_dim=var_dic['phi_A'].shape[1], activation='sigmoid', use_bias=False))
model_phi_A.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())

# Note the large output dimension and the softmax. We want multiple transition probabilities that sum up to 1
# Its the shape**2, as we flatten the square matrix.
model_gamma = Sequential()
model_gamma.add(SimpleDense(no_states))
model_gamma.compile(loss=GCM.pos_log_loss, optimizer=RMSprop())

# model_tau = Sequential()
# model_tau.add(Dense(var_dic['tau'].shape[1], input_dim=var_dic['tau'].shape[1], activation=None, use_bias=False,
#                     kernel_initializer=Identity(), trainable=False))
# model_tau.compile('rmsprop', 'binary_crossentropy')  # No trainable weights, so doesn't really matter

var_models = {'phi_A': model_phi_A, 'gamma': model_gamma}

res = GCM.runEM(click_mat, var_dic, var_models, item_pos_mat, model_def, verbose=True)
