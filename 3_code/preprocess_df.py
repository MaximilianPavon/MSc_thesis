from my_functions import preprocess_df
import os

if __name__ == '__main__':

    params = {
        'path_to_csv': '../2_data/01_MAVI_unzipped_preprocessed/MAVI2/2015/rap_2015.csv',
        'path_to_data': '../2_data/03_data/',
        'colour_band': 'BANDS-S2-L1C',
        'file_extension': '.tiff',
    }

    df = preprocess_df(params['path_to_csv'], params['path_to_data'], params['colour_band'], params['file_extension'])
    new_path = os.path.join(*params['path_to_csv'].split('/')[0:-1], 'preprocessed.csv')
    df.to_csv(new_path, index=False)
    print(f'preprocessed .csv file saved under: {new_path}')
