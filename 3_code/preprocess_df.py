from my_functions import preprocess_df_Rehuohra, preprocess_df_top4
import os
import argparse
import sys


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project_path", help="Specify project path, where the project is located.")
    parser.add_argument("-o", "--out_file", help="Specify the output file name")
    req_grp = parser.add_argument_group(title='required arguments')
    req_grp.add_argument("-c", "--computer", help="Specify computer: use \'triton\', \'mac\' or \'workstation\'.",
                         required=True)
    args = parser.parse_args()

    if not args.project_path:
        if args.computer == 'triton':
            args.project_path = '/scratch/cs/ai_croppro'
        elif args.computer == 'mac':
            args.project_path = '/Users/maximilianproll/Dropbox (Aalto)/'
        elif args.computer == 'workstation':
            args.project_path = '/m/cs/scratch/ai_croppro'
        else:
            sys.exit('Please specify the computer this programme runs on using \'triton\', \'mac\' or \'workstation\'')

    params = {
        'path_to_csv': os.path.join(args.project_path, '2_data/01_MAVI_unzipped_preprocessed/MAVI2/2015/rap_2015.csv'),
        'path_to_data': os.path.join(args.project_path, '2_data/03_data/'),
        'colour_band': 'BANDS-S2-L1C',
        'file_extension': '.tiff',
    }

    # df = preprocess_df_Rehuohra(params['path_to_csv'], params['path_to_data'], params['colour_band'], params['file_extension'])
    df = preprocess_df_top4(params['path_to_csv'], params['path_to_data'], params['colour_band'],
                                params['file_extension'])

    if not args.out_file:
        args.out_file = 'preprocessed.csv'
    elif not '.csv' in args.out_file:
        args.out_file = args.out_file + '.csv'

    new_path = os.path.join('/', *params['path_to_csv'].split('/')[0:-1], args.out_file)
    df.to_csv(new_path, index=False)
    print(f'preprocessed .csv file saved under: {new_path}')
