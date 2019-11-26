import json
import os
import matplotlib.pyplot as plt
from os.path import expanduser
from torch.utils.data import Dataset
from PIL import Image
import pathlib

def isImgFile(file):
    # Check if file has certain base name and is of the proper file type
    return file.endswith(".png") and not(file.endswith(".cs.png")) and not(file.endswith(".depth.png"))


def isBboxFile(file):
    # Check if file has certain base name and is of the proper file type
    return file.endswith(".json")


class Dataloader(Dataset):

    # Initialize the Dataloader (this is the constructor)
    def __init__(self, base_directory):
        # Get the file names
        ROOT = pathlib.Path(expanduser("~")) / base_directory
        self.filenames_img = [os.path.join(path, file) for path, directories, filenames in os.walk(ROOT) for
                             file in filenames if isImgFile(file)]
        self.filenames_bbox = [os.path.join(path, file) for path, directories, filenames in os.walk(ROOT) for
        file in filenames if isBboxFile(file)]

        self.filenames = [os.path.splitext(file)[0] for path, directories, filenames in os.walk(ROOT) for file in filenames if
                              isImgFile(file)]

        # Check if we have same number of images and bounding boxes
        assert len(self.filenames_img) == len(self.filenames_bbox), 'Not the same number of images and bounding boxes'

        # Sort the lists to make sure that we entries of _img and _bbox correspond to one another.
        self.filenames_img.sort()
        self.filenames_bbox.sort()
        self.filenames.sort()

        print(self.filenames)

    def __getitem__(self, index):
        # Get image file name.
        filename_img = self.filenames_img[index]

        # Open file.
        with open(filename_img, 'rb') as f:
            # Read file as RGB image.
            img = Image.open(f).convert('RGB')

        # Get json file name.
        filename_bbox = self.filenames_bbox[index]

        # Open file.
        with open(filename_bbox, 'rb') as g:
            # Read bbox from .json file.
            bbox = json.load(g)["objects"][0]["bounding_box"]

        with open(filename_bbox, 'rb') as h:
            # Read bbox from .json file.
            label = json.load(h)["objects"][0]["class"]

        return img, bbox

    def __len__(self):
        return len(self.filenames_img)

if __name__ == '__main__':
    a = Dataloader("data/Eigenes_Set/Kite")
    b, c = a[0]
    plt.imshow(b)
    print(c["top_left"][0])