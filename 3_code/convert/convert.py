import matplotlib
matplotlib.use("TkAgg")  # fix for macOS
from matplotlib import pyplot as plt
import glob, os
import argparse
import sys
from skimage import io, exposure
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''Convert all .tiff images in a specified directory to true color.''')
    req_grp = parser.add_argument_group(title='Required')
    req_grp.add_argument("--path", "-p", type=str, default=None,
                         help="Path where all .tiff files shall be converted to true color")
    args = parser.parse_args()

    if not args.path:
        print('Please specify a path using the flag --path path/*.tiff')
        sys.exit(2)

    print(f'Converting all .tiff images in {args.path}.')
    os.chdir(args.path)

    n_files = len(glob.glob("*.tiff"))

    for file in tqdm(glob.glob("*.tiff"), total=n_files):
        f_name = file.split('.')[0]

        im = io.imread(file)
        im = im[:, :, [3, 2, 1]]

        # a tiff file contains the raw channels: return [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12]
        # a rgb images needs the following channels:  return [B04, B03, B02]
        # ==> index [3, 2, 1]

        im = exposure.rescale_intensity(im)

        plt.imshow(im)

        plt.title(f'True color of image: {f_name}')
        plt.savefig(f'{f_name}.png')
        plt.clf()

