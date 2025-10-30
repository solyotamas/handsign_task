import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.interpolate import CubicSpline


def make_sequence_df(path, df_data, sign_list = None, every_sign = False):
    if every_sign == False:
        df_data_subset = df_data[df_data['sign'].isin(sign_list)]
    else:
        df_data_subset = df_data
    # ~~~~~~~~~~~~~
    #p = Path('asl-signs')

    df = pd.DataFrame()
    df_temps = []
    for row in df_data_subset.itertuples():
        file_path = path / row.path
        
        df_temp = pd.read_parquet(file_path)
        df_temp['participant_id'] = row.participant_id
        df_temp['sequence_id'] = row.sequence_id
        df_temp['sign'] = row.sign
        
        df_temps.append(df_temp)
    df = pd.concat(df_temps, ignore_index=True)

    return df

# ~~~~~~~~~~

def reindex_frames(df):
    df['frame'] = df['frame'] - df['frame'].min()
    return df

def handle_missing_face(df):
    face_mask = df['type'] == 'face'

    df.loc[face_mask, ['x', 'y', 'z']] = df[face_mask].groupby('landmark_index')[['x', 'y', 'z']].transform(
        lambda x: x.interpolate(method = 'linear')
    )
    df.loc[face_mask, ['x', 'y', 'z']] = df[face_mask].groupby('landmark_index')[['x', 'y', 'z']].transform(
        lambda x: x.ffill()
    )
    df.loc[face_mask, ['x', 'y', 'z']] = df[face_mask].groupby('landmark_index')[['x', 'y', 'z']].transform(
        lambda x: x.bfill()
    )

    return df

def handle_missing_hands(df):

    for hand_type in ['left_hand', 'right_hand']:

        hand_mask = df['type'] == hand_type
        hand_data = df[hand_mask]
        
        has_any_data = hand_data['x'].notna().any()
        
        if has_any_data:
            df.loc[hand_mask, ['x', 'y', 'z']] = df[hand_mask].groupby('landmark_index')[['x', 'y', 'z']].transform(
                lambda x: x.interpolate(method = 'linear')
            )
            df.loc[hand_mask, ['x', 'y', 'z']] = df[hand_mask].groupby('landmark_index')[['x', 'y', 'z']].transform(
                lambda x: x.ffill()
            )
            df.loc[hand_mask, ['x', 'y', 'z']] = df[hand_mask].groupby('landmark_index')[['x', 'y', 'z']].transform(
                lambda x: x.bfill()
            )
            
    return df

def fill_missing_hands(df):
    hand_mask = df['type'].isin(['left_hand', 'right_hand'])

    df.loc[hand_mask, ['x', 'y', 'z']] = df.loc[hand_mask, ['x', 'y', 'z']].fillna(0)

    return df


def drop_sequences_by_missing_face(df_data, path, threshold = 0.5):
    bad_sequence_ids = []
    
    for index, row in tqdm(df_data.iterrows(), total=len(df_data), desc=f'Dropping sequences with more than >={threshold*100}% face frames missing..'):
        df = pd.read_parquet(path / row['path'])
        
        face_data = df[df['type'] == 'face']

        missing_frames = face_data[face_data['x'].isnull()]['frame'].nunique()
        total_frames = face_data['frame'].nunique()
        
        percent_missing = missing_frames / total_frames
        if percent_missing >= threshold:
            bad_sequence_ids.append(row['sequence_id'])

    print(f'\nSequences before: {len(df_data)}')
    print(f'Dropping: {len(bad_sequence_ids)} sequences (>{threshold*100}% missing face)')
    
    clean_metadata = df_data[~df_data['sequence_id'].isin(bad_sequence_ids)].reset_index(drop=True)
    print(f'Sequences after: {len(clean_metadata)}')

    return clean_metadata


#aaaa
def center_around_nose(df):

    def center_frame(df_frame):
        df_frame = df_frame.copy()

        nose_row = df_frame[(df_frame['type'] == 'face') & (df_frame['landmark_index'] == 4)]
        
        nose_x = nose_row['x'].values[0]
        nose_y = nose_row['y'].values[0]
        nose_z = nose_row['z'].values[0]

        df_frame['x'] = df_frame['x'] - nose_x
        df_frame['y'] = df_frame['y'] - nose_y
        df_frame['z'] = df_frame['z'] - nose_z
        
        return df_frame
    
    df = df.groupby('frame', group_keys=False).apply(center_frame)
    return df

def center_around_nose_2(df):
    nose_coords = df[(df['type'] == 'face') & (df['landmark_index'] == 4)][['frame', 'x', 'y', 'z']].copy()
    nose_coords.columns = ['frame', 'nose_x', 'nose_y', 'nose_z']
    
    df = df.merge(nose_coords, on='frame', how='left')
    
    df['x'] = df['x'] - df['nose_x']
    df['y'] = df['y'] - df['nose_y']
    df['z'] = df['z'] - df['nose_z']
    
    df = df.drop(columns=['nose_x', 'nose_y', 'nose_z'])
    
    return df

#aaaaa
def scale_by_shoulder_width(df : pd.DataFrame) -> pd.DataFrame:
    
    def scale_frame(df_frame):
        df_frame = df_frame.copy()

        left_shoulder_vals = df_frame[(df_frame['type'] == 'pose') & (df_frame['landmark_index'] == 11)][['x', 'y', 'z']].values[0]
        right_shoulder_vals = df_frame[(df_frame['type'] == 'pose') & (df_frame['landmark_index'] == 12)][['x', 'y', 'z']].values[0]

        distance = np.sqrt(
            (left_shoulder_vals[0] - right_shoulder_vals[0])**2 +
            (left_shoulder_vals[1] - right_shoulder_vals[1])**2 +
            (left_shoulder_vals[2] - right_shoulder_vals[2])**2
        )

        df_frame['x'] = df_frame['x'] / distance
        df_frame['y'] = df_frame['y'] / distance
        df_frame['z'] = df_frame['z'] / distance

        return df_frame

    df = df.groupby('frame', group_keys=False).apply(scale_frame)
    return df

def scale_by_shoulder_width_2(df):
    left_shoulder = df[(df['type'] == 'pose') & (df['landmark_index'] == 11)][['frame', 'x', 'y', 'z']].copy()
    right_shoulder = df[(df['type'] == 'pose') & (df['landmark_index'] == 12)][['frame', 'x', 'y', 'z']].copy()
    
    left_shoulder.columns = ['frame', 'ls_x', 'ls_y', 'ls_z']
    right_shoulder.columns = ['frame', 'rs_x', 'rs_y', 'rs_z']
    
    df = df.merge(left_shoulder, on='frame', how='left')
    df = df.merge(right_shoulder, on='frame', how='left')
    
    df['shoulder_width'] = np.sqrt(
        (df['ls_x'] - df['rs_x'])**2 +
        (df['ls_y'] - df['rs_y'])**2 +
        (df['ls_z'] - df['rs_z'])**2
    )
    
    df['x'] = df['x'] / df['shoulder_width']
    df['y'] = df['y'] / df['shoulder_width']
    df['z'] = df['z'] / df['shoulder_width']
    
    df = df.drop(columns=['ls_x', 'ls_y', 'ls_z', 'rs_x', 'rs_y', 'rs_z', 'shoulder_width'])
    
    return df


def resample_sequence(df, target_frames = 60):
    frame_count = df['frame'].nunique()

    if frame_count == target_frames:
        return df

    old_frame_indices = np.arange(frame_count)
    new_frame_indices = np.arange(target_frames)
    interpolation_points = np.linspace(0, frame_count - 1, target_frames)

    new_rows = []

    for (type, landmark), group in df.groupby(['type', 'landmark_index']):
        y_values_x = group['x'].values
        y_values_y = group['x'].values
        y_values_z = group['x'].values

        cubicsplinex = CubicSpline(old_frame_indices, y_values_x)
        cubicspliney = CubicSpline(old_frame_indices, y_values_y)
        cubicsplinez = CubicSpline(old_frame_indices, y_values_z)

        new_y_values_x = cubicsplinex(interpolation_points)
        new_y_values_y = cubicspliney(interpolation_points)
        new_y_values_z = cubicsplinez(interpolation_points)

        for i in range(target_frames):
            new_rows.append({
                'frame': new_frame_indices[i],
                'type': type,
                'landmark_index': landmark,
                'x': new_y_values_x[i],
                'y': new_y_values_y[i],
                'z': new_y_values_z[i]
            })

    return pd.DataFrame(new_rows)


    

           







