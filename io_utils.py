import pandas as pd

def read_data():
    df = pd.read_csv('./data/WISDM_ar_v1.1_raw.txt', header=0, names=['User', 'Activity', 'Time', 'x', 'y', 'z'])
    df['Activity'].replace(to_replace=['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing'],
                           value=[0, 1, 2, 3, 4, 5], inplace=True)
    data = df.to_numpy()
    return data
