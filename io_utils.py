"""
This document contains the function needed to obtain the data from the CSV file and convert it to a numpy array.

Author: Jacobo Fernandez Vargas
"""
import pandas as pd
from numpy.typing import NDArray

def read_data() -> NDArray:
    """
    Reads the data from the WIDSM file, and replaces the activity categorical value ('Walking', 'Jogging', 'Upstairs',
     'Downstairs', 'Sitting', 'Standing') for a numeric one
    :return data: Nx6 Matrix containing N samnples, and the columns User, Activity, Timestamp, x, y, and z.
    """
    df = pd.read_csv('./data/WISDM_ar_v1.1_raw.txt', header=0, names=['User', 'Activity', 'Time', 'x', 'y', 'z'])
    df['Activity'].replace(to_replace=['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing'],
                           value=[0, 1, 2, 3, 4, 5], inplace=True)
    data = df.to_numpy()
    return data
