from sklearn.model_selection import train_test_split
import numpy as np


class FeatureEngineer:

    @staticmethod
    def add_list_id(sim_data):
        unique_qsessions = sim_data.\
            groupby(['user_id', 'session_count']).\
            size().\
            rename('freq').\
            reset_index(). \
            sort_values(['user_id', 'session_count']).\
            drop(['freq'], axis=1)

        unique_qsessions['list_id'] = np.arange(unique_qsessions.shape[0])

        sim_data_with_list_id = sim_data.\
            set_index(['user_id', 'session_count']).\
            join(unique_qsessions.set_index(['user_id', 'session_count']),
                                            on=['user_id', 'session_count']).\
            reset_index()

        sim_data_with_list_id = sim_data_with_list_id.\
            sort_values(['user_id', 'session_count', 'item_order'])

        return sim_data_with_list_id

    @staticmethod
    def split_dataset(sim_data, test_frac, rand_state, valid_frac=None):
        """
        Splits dataset into train, validation and test based on users
        """
        unique_users = sim_data['user_id'].unique()
        train_users, test_users = train_test_split(unique_users, test_size=test_frac, random_state=rand_state)

        if valid_frac is not None:
            valid_of_train_frac = valid_frac/(1-test_frac)
            train_users, val_users = train_test_split(train_users, test_size=valid_of_train_frac, random_state=rand_state)
            valid_data = sim_data[sim_data['user_id'].isin(val_users)]

        train_data = sim_data[sim_data['user_id'].isin(train_users)]
        test_data = sim_data[sim_data['user_id'].isin(test_users)]

        if valid_frac is not None:
            return train_data, valid_data, test_data

        return train_data, test_data

