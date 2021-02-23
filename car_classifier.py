import os
import random
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import models, transforms

class CarClassifier:

    def __init__(self, path_to_model, device='cpu'):
        self.path = path_to_model
        self.device = device

        with open(os.path.join(self.path, 'idx_to_class.json'), 'r') as f:
            self.idx_to_class = json.load(f)

        self.model = models.resnet34(pretrained=False)
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=len(self.idx_to_class.keys()), bias=True
        )
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.path, 'model_classifier.pth'),
                map_location=torch.device(self.device))
        )
        self.model = self.model.to(self.device)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.cache = None

        
    def classify(self, path_to_image, illustrate=False):

        orig = cv2.imread(path_to_image)
        orig = cv2.resize(orig, (224,224))
        image_hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        image = torch.tensor(np.transpose(orig, [2,0,1]))
        image = self.transforms(image.float().div(255)).unsqueeze(0)
        
        tensors = []
        def hook(module, inputs, outputs):
            tensors.append(inputs[0])

        with torch.no_grad():
            self.model.avgpool.register_forward_hook(hook)
            self.model.eval()
            output = self.model(image.to(self.device))
            _, predict = output.cpu().topk(10, dim=-1)
            predict = predict.numpy().reshape(-1)
            
        with torch.no_grad():
            tensor = tensors[0].permute([0, 2, 3, 1])
            activations = self.model.fc(tensor.to(self.device))[0].cpu().numpy()

        cam = activations[..., predict].mean(axis=-1)
        cam /= (cam.max() + 1e-5)
        cam = cv2.resize(cam, (224, 224))

        color_filters = {
            'black': cv2.inRange(image_hsv, (0, 0, 0), (255, 255, 40)),
            'grey': cv2.inRange(image_hsv, (0, 0, 0), (255, 40, 200)),
            'white': cv2.inRange(image_hsv, (0, 0, 200), (255, 40, 255)),
            'red': cv2.inRange(image_hsv,(0, 40, 0), (10, 255, 255)) + \
                cv2.inRange(image_hsv, (170, 40, 0), (180, 255, 255)),
            'orange': cv2.inRange(image_hsv, (10, 40, 40), (15, 255, 255)),
            'yellow': cv2.inRange(image_hsv, (15, 40,40), (40, 255, 255)),
            'green': cv2.inRange(image_hsv, (40, 40, 40), (100, 255, 255)),
            'blue': cv2.inRange(image_hsv, (100, 40, 40), (125, 255, 255)),
            'violet': cv2.inRange(image_hsv,(125, 40,40), (140, 255, 255)),
            'pink': cv2.inRange(image_hsv,(140, 40,40), (170, 255, 255))
        }
        cam_filter = (cam >= np.quantile(cam, 0.85)).astype(bool)
        total_px = cam_filter.sum()
        max_rate, max_color = 0, None
        for color, filter_px in color_filters.items():
            rate = (filter_px[cam_filter] > 0).sum() / total_px
            if rate > max_rate:
                max_rate, max_color = rate, color

        self.cache = {
            'RGB': orig, 
            'CAM': cam_filter.astype(int)*cam,
        }

        if illustrate:
            self.illustrate_decision()

        return self.idx_to_class[str(predict[0])], max_color

    def illustrate_decision(self):
        if self.cache is not None:
            _, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
            axs[0].imshow(self.cache['RGB'])
            axs[1].imshow(self.cache['CAM'])
            rgb_cam = self.cache['RGB'].astype(float) / 255
            rgb_cam[..., 2] = self.cache['CAM']
            axs[2].imshow(rgb_cam)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='./')
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()

    classifier = CarClassifier(path_to_model=args.model, device=args.device)
    while True:
        image_file = input('Enter path to image (type "exit" to quit): ')
        if image_file == 'exit':
            break
        model, color = classifier.classify(image_file)
        print(f' -> {model} ({color})')
