class SDBN:
    """
    Class for fitting SDBN click model_definitions using maximum likelihood
    """
    def __init__(self, att_vec, satis_vec):
        self._att_vec = att_vec
        self._satis_vec = satis_vec

    @property
    def attraction(self):
        return self._att_vec

    @property
    def satisfaction(self):
        return self._satis_vec

    @staticmethod
    def fit_model(click_data):
        """
        Estimates the SDBN parameters
        :param click_data: A pandas dataframe containing the columns: list_res_id (id of the list result),
            pos (position of the item), item_id (id of the item), clicked (whether the item was clicked)
        :return: a pandas dataframe with per item the attraction estimate (att)est), satisfaction estimate (satis_est),
            how many times the item was clicked (clicked), and how frequently the item occurred in a list (freq)
        """
        click_data['pos'] = click_data['pos'].astype(int)
        click_data['click_pos'] = click_data['pos'] * click_data['clicked']
        list_max_click_pos = click_data.groupby('list_res_id').agg({'click_pos': 'max'}).\
            rename(columns={'click_pos': 'max_click_pos'})

        click_data = click_data.set_index('list_res_id').join(list_max_click_pos, on='list_res_id')
        click_data['after_last_click'] = (click_data['pos'] > click_data['max_click_pos']).astype(int)
        click_data['before_last_click'] = 1 - click_data['after_last_click']

        click_data['click_corr'] = click_data['clicked'] * click_data['before_last_click']
        click_data['click_last_pos'] = click_data['clicked'] * (click_data['pos'] == click_data['max_click_pos'])

        after_clicks = click_data.groupby('item_id').agg({'after_last_click': 'sum',
                                                          'click_corr': 'sum',
                                                         'clicked': 'sum',
                                                         'click_last_pos': 'sum'})
        item_freq = click_data['item_id'].value_counts().reset_index().rename(columns={'index': 'item_id',
                                                                            'item_id': 'freq'}).set_index('item_id')
        after_clicks = after_clicks.join(item_freq, on='item_id')
        after_clicks['att_est'] = after_clicks['click_corr']/(after_clicks['freq'] - after_clicks['after_last_click'])

        after_clicks['satis_est'] = after_clicks['click_last_pos']/after_clicks['clicked']

        return after_clicks.loc[:, ['att_est', 'satis_est', 'clicked', 'freq']]

