import matplotlib
matplotlib.use("TkAgg")  # fix for macOS
from matplotlib import pyplot as plt
from skimage import io, exposure
import argparse
import sys
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''Convert a .tiff image to true color.''')
    req_grp = parser.add_argument_group(title='Required')
    req_grp.add_argument("--file", "-f", type=str, default=None,
                         help="Path to .tiff file to convert to true color")
    args = parser.parse_args()

    if not args.file:
        print('Please specify a file using the flag --file path/foo.tiff')
        sys.exit(2)

    print(f'reading file: {args.file}')
    im_name = args.file.split('/')[-1].split('.')[0]

    im = io.imread(args.file)
    im = im[:, :, [3, 2, 1]]

    # a tiff file contains the raw channels: return [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12]
    # a rgb images needs the following channels:  return [B04, B03, B02]
    # ==> index [3, 2, 1]

    im = exposure.rescale_intensity(im)

    path = './2_data/06_png/'

    plt.imshow(im)
    plt.title(f'True color of image: {im_name}')
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, im_name, '.png'))


