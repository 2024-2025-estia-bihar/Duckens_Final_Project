
import pandas as pd
from datetime import datetime

def transform_to_3h_interval(df):
    """
    Regroupe les données horaires en tranches de 3 heures
    """
    df_copy = df.copy()
    df_copy['hour'] = df_copy['time'].dt.hour
    df_copy['group_3h'] = (df_copy['hour'] // 3) * 3
    df_copy['date'] = df_copy['time'].dt.date

    data_columns = [col for col in df_copy.columns if col not in ['time', 'hour', 'group_3h', 'date']]
    result = []

    for date, date_group in df_copy.groupby('date'):
        for group_3h, hour_group in date_group.groupby('group_3h'):
            entry = {
                'time': datetime.combine(date, datetime.min.time()) + pd.Timedelta(hours=group_3h)
            }
            for col in data_columns:
                entry[col] = round(hour_group[col].mean(), 1)
            result.append(entry)

    df_3h = pd.DataFrame(result)
    df_3h = df_3h.sort_values('time')
    return df_3h
