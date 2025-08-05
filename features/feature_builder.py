import pandas as pd
import numpy as np
from datetime import timedelta

class FeatureBuilder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def run_pipeline(self):
        return (
            self._add_amount_features()
                ._add_time_features()
                ._add_device_features()
                ._add_location_features()
                ._add_temporal_flags()
                ._add_transaction_type_features()
                ._finalize()
        )

    def _add_amount_features(self):
        stats = self.df.groupby('user_id')['amount'].agg(
            user_avg_amount='mean',
            user_std_amount='std'
        ).reset_index()
        self.df = self.df.merge(stats, on='user_id', how='left')
        self.df['amount_zscore'] = (self.df['amount'] - self.df['user_avg_amount']) / self.df['user_std_amount']
        self.df['is_high_value_txn'] = (self.df['amount_zscore'] > 2.5).astype(int)
        return self

    def _add_time_features(self):
        self.df['timestamp'] = pd.to_datetime(self.df['date'].astype(str) + ' ' + self.df['time'].astype(str))
        self.df = self.df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
        self.df['prev_timestamp'] = self.df.groupby('user_id')['timestamp'].shift(1)
        self.df['time_since_last_txn_min'] = (self.df['timestamp'] - self.df['prev_timestamp']).dt.total_seconds() / 60

        mean_gap = self.df.groupby('user_id')['time_since_last_txn_min'].mean().reset_index()
        mean_gap.columns = ['user_id', 'mean_time_between_txns']
        self.df = self.df.merge(mean_gap, on='user_id', how='left')

        self.df['txns_in_last_1hr'] = 0
        self.df['txns_in_last_24hr'] = 0
        self.df['is_rapid_txn'] = 0

        for user_id, group in self.df.groupby('user_id'):
            times = group['timestamp'].values
            txns_1hr, txns_24hr, rapid_flags = [], [], []
            for i in range(len(times)):
                now = times[i]
                txns_1hr.append(np.sum((now - times[:i]) <= np.timedelta64(1, 'h')))
                txns_24hr.append(np.sum((now - times[:i]) <= np.timedelta64(24, 'h')))
                rapid_flags.append(int(i > 0 and (now - times[i-1]) < np.timedelta64(5, 'm')))
            self.df.loc[group.index, 'txns_in_last_1hr'] = txns_1hr
            self.df.loc[group.index, 'txns_in_last_24hr'] = txns_24hr
            self.df.loc[group.index, 'is_rapid_txn'] = rapid_flags

        self.df.drop(columns=['prev_timestamp'], inplace=True)
        return self

    def _add_device_features(self):
        self.df['previous_device'] = self.df.groupby('user_id')['device'].shift(1)
        self.df['device_switch'] = (self.df['device'] != self.df['previous_device']).astype(int).fillna(0)
        device_seen = {}
        device_counts = []
        for i, row in self.df.iterrows():
            user, device = row['user_id'], row['device']
            if user not in device_seen:
                device_seen[user] = set()
            device_seen[user].add(device)
            device_counts.append(len(device_seen[user]))
        self.df['num_unique_devices'] = device_counts
        self.df.drop(columns=['previous_device'], inplace=True)
        return self

    def _add_location_features(self):
        location_coords = {
            "Birmingham": (52.4862, -1.8904),
            "Cardiff": (51.4816, -3.1791),
            "Glasgow": (55.8642, -4.2518),
            "Leeds": (53.8008, -1.5491),
            "Liverpool": (53.4084, -2.9916),
            "London": (51.5074, -0.1278),
            "Manchester": (53.4808, -2.2426),
            "None": (np.nan, np.nan)
        }
        self.df[['latitude', 'longitude']] = self.df['location'].map(location_coords).apply(pd.Series)
        self.df = self.df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
        self.df['prev_latitude'] = self.df.groupby('user_id')['latitude'].shift(1)
        self.df['prev_longitude'] = self.df.groupby('user_id')['longitude'].shift(1)

        def fast_haversine_km(lat1, lon1, lat2, lon2):
            R = 6371.0
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return R * c

        valid = self.df[['latitude', 'longitude', 'prev_latitude', 'prev_longitude']].notnull().all(axis=1)
        self.df['location_jump_km'] = np.nan
        self.df.loc[valid, 'location_jump_km'] = fast_haversine_km(
            self.df.loc[valid, 'latitude'],
            self.df.loc[valid, 'longitude'],
            self.df.loc[valid, 'prev_latitude'],
            self.df.loc[valid, 'prev_longitude']
        )
        self.df['is_location_jump'] = (self.df['location_jump_km'] > 100).astype(int)
        if 'location_cluster' in self.df.columns:
            self.df['prev_cluster'] = self.df.groupby('user_id')['location_cluster'].shift(1)
            self.df['location_cluster_change'] = (self.df['location_cluster'] != self.df['prev_cluster']).astype(int)
            self.df.drop(columns=['prev_cluster'], inplace=True)
        self.df.drop(columns=['prev_latitude', 'prev_longitude'], inplace=True)
        return self

    def _add_temporal_flags(self):
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        self.df['hour_of_day'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['is_night_txn'] = self.df['hour_of_day'].between(0, 5).astype(int)
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['is_off_hours_txn'] = (~self.df['hour_of_day'].between(9, 18)).astype(int)
        return self

    def _add_transaction_type_features(self):
        self.df = self.df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
        self.df['last_txn_type'] = self.df.groupby('user_id')['txn_type'].shift(1)
        self.df['txn_type_switch'] = (self.df['txn_type'] != self.df['last_txn_type']).astype(int)
        self.df['txn_type_count'] = self.df.groupby(['user_id', 'txn_type']).cumcount() + 1

        most_freq_txn_type = (
            self.df.groupby(['user_id', 'txn_type'])
            .size()
            .reset_index(name='count')
            .sort_values(['user_id', 'count'], ascending=[True, False])
            .drop_duplicates('user_id')
            .set_index('user_id')['txn_type']
            .to_dict()
        )
        self.df['most_freq_txn_type'] = self.df['user_id'].map(most_freq_txn_type)
        self.df['is_new_txn_type'] = (self.df['txn_type'] != self.df['most_freq_txn_type']).astype(int)
        self.df.drop(columns=['most_freq_txn_type'], inplace=True)
        return self

    def _finalize(self):
        return self.df