import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


import Dataset
from Dataset import CustomDataset, collate_fn

torch.set_float32_matmul_precision('medium')


class ErrorDetectionModel(L.LightningModule):
    """
    Lightning module for the model class that can train, validate and test the model.
    """
    def __init__(self, json_path, image_path, batch_size, sum_classes):
        """
        Initializes the class.
        :param json_path: path to json file containing the annotations of the bounding boxes with labels
        :param image_path: path to images
        :param batch_size: the number of images in one batch
        :param sum_classes: True if similar classes should be summarized to one class
        """
        super(ErrorDetectionModel, self).__init__()
        self.batch_size = batch_size
        self.json_path = json_path
        self.image_path = image_path
        self.sum_classes = sum_classes
        self.setup_data_model()

    def setup_data_model(self):
        """
        Sets up the dataloaders and the model.
        """
        height = int(480 * self.reduce_factor)
        width = int(640 * self.reduce_factor)

        custom_dataset = CustomDataset(self.json_path, self.image_path, height=height, width=width,
                                       sum_classes=self.sum_classes)

        indices = range(0, len(custom_dataset))
        train_size = int(0.8 * len(custom_dataset))
        val_size = int(0.9 * len(custom_dataset))

        self.train_dataset = Subset(custom_dataset, indices[:train_size])
        self.val_dataset = Subset(custom_dataset, indices[train_size:val_size])
        self.test_dataset = Subset(custom_dataset, indices[val_size:])

        self.num_classes = len(custom_dataset.label_mapping)
        self.model = self.get_error_detection_model(self.num_classes)
        del custom_dataset

    def train_dataloader(self):
        """
        Creates the dataloader for training.
        :return: train dataloader
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True, shuffle=True,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        """
        Creates the dataloader for validation.
        :return: validation dataloader
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True, shuffle=False,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        """
        Creates the dataloader for testing.
        :return: test dataloader
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True, shuffle=False,
                          collate_fn=collate_fn)

    def get_error_detection_model(self, num_classes):
        """
        Initializes the error detection model.
        :param num_classes: the number of classes
        :return: the model
        """
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        :return: dictionary of the optimizer and learning rate scheduler
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, eps=0.006, weight_decay=0.0006)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 17)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler}
        }

    def forward(self, images, targets=None):
        """
        The forward function of the model.
        :param images: the batch of images to learn from
        :param targets: the respective batch of targets
        :return: the output of the model
        """
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)

    def training_step(self, batch, batch_idx):
        """
        The training step of the model.
        :param batch: the input batch of images and targets
        :param batch_idx: the id of the batch
        :return: the loss
        """
        images, targets = batch
        loss_dict = self(images, targets)
        losses = loss_dict['loss_rpn_box_reg'] + loss_dict['loss_classifier'] + loss_dict['loss_box_reg']
        self.log('loss', losses, prog_bar=False)
        return {"loss": losses}

    def validation_step(self, batch, batch_idx):
        """
        The validation step of the model. Logs the results to the progres bar.
        :param batch: the input batch of images and targets
        :param batch_idx: the id of the batch
        """
        images, targets = batch
        outputs = self(images)
        results = pd.DataFrame([])
        for output, target in zip(outputs, targets):
            results = pd.concat([results, self.compute_metrics(output, target)], axis=1)
        self.log_dict(results.T.mean().to_dict(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        The test step of the model. Logs the results to the progres bar.
        :param batch: the input batch of images and targets
        :param batch_idx: the id of the batch
        """
        images, targets = batch
        outputs = self(images)
        results = pd.DataFrame([])
        for output, target in zip(outputs, targets):
            results = pd.concat([results, self.compute_metrics(output, target)], axis=1)
        self.log_dict(results.T.mean().to_dict(), prog_bar=True)

    def compute_metrics(self, output, target, thresh_score=0.12, thresh_iou=0.1):
        """
        Compute the preformance metrics. A box needs to overlap more than a certain threshold to be considered and the
        certainty score of a predicted label needs to surpass another threshold to be considered.
        :param output: the models output
        :param target: the ground truth
        :param thresh_score: the threshold to consider a predicted label
        :param thresh_iou: the threshold to consider a predicted box
        :return: precision, recall, average precision, f-score, man iou
        """
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        true_positives = 0
        false_positives = 0
        false_negatives = len(gt_boxes)
        ious = []
        for pred_box, pred_label, pred_score in zip(output['boxes'], output['labels'], output['scores']):
            if pred_score > thresh_score:
                ious_box = [(i, self.compute_iou(pred_box, gt_box)) for i, (gt_box, gt_label) in
                            enumerate(zip(gt_boxes, gt_labels)) if pred_label == gt_label]
                if len(ious_box) > 0:
                    best_iou = max(ious_box, key=lambda x: x[1])

                    # Check if the prediction is a true positive
                    if best_iou[1] >= thresh_iou:
                        true_positives += 1
                        false_negatives -= 1
                        gt_boxes[best_iou[0]] = torch.tensor([-1, -1, -1, -1], dtype=torch.float32)
                        ious.append(best_iou[1].cpu().item())
                    else:
                        false_positives += 1

        precision = true_positives / (true_positives + false_positives + 1)
        recall = true_positives / (true_positives + false_negatives)
        return pd.Series({
            'precision': precision,
            'recall': recall,
            'average_precision': precision * recall,
            'fscore': (2 * precision * recall) / (precision + recall + 1),
            'mean_iou': sum(ious) / (len(ious) + 1)
        })

    def compute_iou(self, box1, box2):
        """
        Compute the iou (intersection over union) of the two boxes.
        :param box1: coordinates of the predicted box
        :param box2: coordinates of the ground truth box
        :return: the iou
        """
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area_box1 + area_box2 - intersection

        iou = intersection / union
        return iou


