import argparse
import os
from PIL import Image
from estimator import AnomalyDetector
import numpy as np

# function for segmentations
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

parser = argparse.ArgumentParser()
parser.add_argument('--demo_folder', type=str, default='./sample_images', help='Path to folder with images to be run.')
parser.add_argument('--save_folder', type=str, default='./results', help='Folder to where to save the results')
opts = parser.parse_args()

demo_folder = opts.demo_folder
save_folder = opts.save_folder

images = [os.path.join(demo_folder, image) for image in os.listdir(demo_folder) if image.split('.')[0].lower().endswith('_rgb')]
detector = AnomalyDetector(True)

# Save folders
os.makedirs(save_folder, exist_ok=True)

for idx, image in enumerate(images):
    basename = os.path.basename(image).replace('.jpg', '.png').replace('.png', '')
    print('Evaluating image %i out of %i'%(idx+1, len(images)))
    image = Image.open(image)
    img = np.asarray(image)
    results = detector.estimator_worker(img)
    np.save(os.path.join(save_folder, basename), results.detach().cpu().numpy())
