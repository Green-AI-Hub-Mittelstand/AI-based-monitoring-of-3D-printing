import json
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,   FasterRCNN_MobileNet_V3_Large_FPN_Weights

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_model(model_path, num_classes):
    """
    Load the trained model from the given path.
    :param model_path: path to the model state
    :param num_classes: number of classes the model was trained with
    :return: the model
    """
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def classify(image, transform, model):
    """
    Classifies the bounding boxes and labels for an input image.
    :param image: the image to classify
    :param transform: the preprocessing of the image
    :param model: the trained model
    :return: the bounding boxes and labels for the image
    """
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    return output


if __name__ == '__main__':
    # Classifies and plots an image given the path to the image, the label map path and the model path.
    model_path = '/model.pth'
    label_path = '/label_map.json'
    image_path = 'images/Images/'

    label_mapping = json.load(open(label_path, 'r'))
    num_classes = len(label_mapping)
    model = load_model(model_path, num_classes)
    transform = transforms.Compose([transforms.Resize([480, 640]), transforms.ToTensor()])
    images = ['0ae3d5ca-461c-4016-9fbf-9fd8c5a84d28.png', '0d6d6253-ceaa-489b-8dd0-e3080dff2220.png', '0f59d924-4637-4b38-930e-89983dbccab7.png', '1b1c4696-2d9f-451f-8d8f-ddd86c9f6260.png', '1c9da05b-c9ba-480d-a8b0-3628e6ca2d98.png']
    for img in images:
        image = Image.open(image_path + img)
        outputs = classify(image, transform, model)
        plt.imshow(image)
        current_axis = plt.gca()
        outputs = outputs[0]

        for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
            if score > 0.3:  # threshold can be adjusted
                box = box.numpy()
                rect = plt.Rectangle((box[0], box[3]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red',
                                     linewidth=2)
                current_axis.add_patch(rect)
                current_axis.annotate(label_mapping[label], (box[0], box[3]), color='black', weight='bold', fontsize=10)
        plt.show()
