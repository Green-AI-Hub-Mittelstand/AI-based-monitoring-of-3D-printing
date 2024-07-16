import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CustomDataset(Dataset):
    """
    Dataset class to load the labeled 3d-printing error data..
    """
    def __init__(self, json_file, image_dir, height, width, sum_classes=False):
        """
        Initializes the class.
        :param json_file: path to json file containing the annotations of the bounding boxes with labels
        :param image_dir: path to the image folder
        :param height: height that ll images will be resized to
        :param width: width that all images will be resized to
        :param sum_classes: True if similar classes should be summarized to one class
        """
        self.data = self.load_json(json_file)
        self.label_mapping = self.create_mapping(sum_classes)
        self.height = height
        self.width = width
        self.image_dir = image_dir
        self.transform = transforms.Compose([transforms.Resize([self.height, self.width]), transforms.ToTensor()])

    def load_json(self, json_file):
        """
        Loads the json file containing the annotations of the bounding boxes with labels.
        :param json_file: path to json file
        :return: the read data
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        """
        :return: the length of the data
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns an image with its bounding boexes and label as target given an id.
        :param idx: id of the image
        :return: the image and the target
        """
        entry = self.data[idx]
        img_dir = os.path.join(self.image_dir, entry['id'] + '.png')
        image = Image.open(img_dir)

        image = self.transform(image)
        target = self.create_target(entry)
        return image, target

    def create_mapping(self, sum_classes):
        """
        Creates the mapping between classes and ids. If classes are summed, similar classes get the same id.
        :param sum_classes: True if similar classes should be summarized to one class
        :return: dictionary of classes to ids
        """
        labels = list(set([tag['tag_name'] for entry in self.data for tag in entry['tags']]))
        if sum_classes:
            labels_map = dict(enumerate(list(set([label.split('-')[0] for label in labels]))))
            labels_map = {v: k for k, v in labels_map.items()}
            labels = {label: labels_map[label.split('-')[0]] for label in labels}
        else:
            labels = dict(enumerate(labels))
            labels = {v: k for k, v in labels.items()}
        return labels

    def create_target(self, entry):
        """
        Creates the target in the correct format. It takes an entry of the json file as input and rearranges the
        bounding boxes in the format the model requires (coordinates (x1,y1) and (x2,y2)). The coordinates are
        multiplied by the input height and width to fit the image size. If the label is "NoFailure", the class is set to
        -1 with a zero-box.
        :param entry: one entry of annotations of the json file
        :return: the atrget containing the adjusted box coordinates and the respective labels
        """
        target = {'boxes': [], 'labels': []}
        if entry['tags'][0]['tag_name'] == 'NoFailure':
            target = {
                "boxes": torch.tensor(torch.zeros((0, 4)), dtype=torch.float32),  # No boxes
                "labels": torch.tensor([self.label_mapping[entry['tags'][0]['tag_name']]], dtype=torch.int64)  # No labels
            }
        else:
            for region, tag in zip(entry['regions'], entry['tags']):
                x1 = float(region['left'] * self.width)
                y2 = float(region['top'] * self.height)
                x2 = x1 + float(region['width'] * self.width)
                y1 = y2 - float(region['height'] * self.height)
                bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
                target['boxes'].append(bbox)
                target['labels'].append(self.label_mapping[tag['tag_name']])
            target['boxes'] = torch.stack(target['boxes'])
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        return target


def collate_fn(batch):
    """
    Collate function for the dataloader.
    :param batch: a training batch
    :return: the reformatted batch
    """
    return tuple(zip(*batch))