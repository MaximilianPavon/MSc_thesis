import matplotlib
matplotlib.use("TkAgg")  # fix for macOS
from matplotlib import pyplot as plt
from skimage import io
import argparse
import sys

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
    im = im[:, :, [4, 3, 2]]

    plt.imshow(im)
    plt.title(f'True color of image: {im_name}')
    plt.savefig(f'../2_data/04_png/{im_name}.png')


