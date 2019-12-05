import numpy as np
import cv2
import logging
import pathlib
import os
import json

def isImgFile(file):
    return (file.endswith(".png") or file.endswith(".jpg")) and not(file.endswith(".cs.png")) \
           and not(file.endswith(".depth.png")) and not(file.endswith(".is.png")) and not(file.startswith("."))

def isBboxFile(file):
    return file.endswith(".json") and not(file.startswith("_"))

class VOCDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform

        self.filenames_img = [os.path.join(path, file) for path, directories, filenames in os.walk(root) for
                              file in filenames if isImgFile(file)]
        self.filenames_bbox = [os.path.join(path, file) for path, directories, filenames in os.walk(root) for
                               file in filenames if isBboxFile(file)]
        self.ids = [str(i) for i in range(len(self.filenames_img))]

        assert len(self.filenames_img) == len(self.filenames_bbox), 'Not the same number of images and bounding boxes'

        self.filenames_img.sort()
        self.filenames_bbox.sort()
        self.ids.sort()

        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list
            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default dataset classes.")
            self.class_names = ('BACKGROUND', 'strawberry')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        boxes, labels, is_difficult = self._get_annotation(index)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(index)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image = self._read_image(index)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(index)

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, index):
        boxes = []
        labels = []
        is_difficult = []

        with open(self.filenames_bbox[index], 'rb') as h:
            json_file = json.load(h)
            if "objects" in json_file:
                for counter in range(len(json_file["objects"])):
                    class_name = json_file["objects"][counter]["class"]
                    if class_name in self.class_dict:
                        x1 = json_file["objects"][counter]["bounding_box"]["top_left"][1]  # xmin
                        y1 = json_file["objects"][counter]["bounding_box"]["top_left"][0]  # ymin
                        x2 = json_file["objects"][counter]["bounding_box"]["bottom_right"][1]  # xmax
                        y2 = json_file["objects"][counter]["bounding_box"]["bottom_right"][0]  # ymax
                        boxes.append([x1, y1, x2, y2])

                        labels.append(self.class_dict[class_name])
                        is_difficult.append(0)
            elif "shapes" in json_file:
                for counter in range(len(json_file["shapes"])):
                    class_name = json_file["shapes"][counter]["label"]
                    if class_name in self.class_dict:
                        x1 = json_file["shapes"][counter]["points"][0][0]  # xmin
                        y1 = json_file["shapes"][counter]["points"][0][1]  # ymin
                        x2 = json_file["shapes"][counter]["points"][1][0]  # xmax
                        y2 = json_file["shapes"][counter]["points"][1][1]  # ymax
                        boxes.append([x1, y1, x2, y2])

                        labels.append(self.class_dict[class_name])
                        is_difficult.append(0)
            else:
                print("Choose correct .json file format")

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, index):
        image_file = self.filenames_img[index]
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



