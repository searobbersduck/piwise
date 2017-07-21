import argparse
import os
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser(description='segmentation post processing')
    parser.add_argument('--root', required=True)
    return parser.parse_args()

args = parse_args()

root = args.root

cat_range = [
    [0.0, 0.25],
    [0.25, 0.5],
    [0.5, 0.75],
    [0.75, 1.0]
]

for range in cat_range:
    cat_path = os.path.join(root, '{0}-{1}'.format(range[0], range[1]))
    assert os.path.isdir(cat_path)
    imgs = glob(os.path.join(cat_path, '*.png'))
    print('files in {0} number is: {1}'.format(cat_path, len(imgs)))
