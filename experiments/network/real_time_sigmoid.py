import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from .modeling import *

class Real_Time_Sigmoid(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        nclasses = 7
        self.nclasses = nclasses

        self.model = deeplabv3plus_resnet50(num_classes=nclasses, output_stride=16).to(self.device)

        ckpt_filename = './experiments/network/resnet50_best_model.pth'
        pretrained_net = torch.load(ckpt_filename, map_location=torch.device(self.device))
        self.model.load_state_dict(pretrained_net)

        self.input_size = 1280

    def eval(self, numpy_image):
        self.model.eval()
        model = self.model

        image = Image.fromarray(numpy_image)
        image_shape = (self.input_size, self.input_size // 2)
        ow, oh = image.size
        image = image.resize(image_shape, Image.BILINEAR)

        image = self._img_transform(image)
        image = image.to(self.device)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            outputs = model(image)

        pred_prob = torch.sigmoid(outputs)

        P = pred_prob[0, :, :, :]
        anomaly_score = torch.prod(1 - P[:-1, :, :], dim=0) * P[-1, :, :]
        anomaly_score = F.interpolate(anomaly_score.view(1,1,anomaly_score.shape[0], anomaly_score.shape[1]), size=(oh, ow), mode="bilinear", align_corners=True)
        anomaly_score = anomaly_score[0, 0, :, :]

        return anomaly_score



    def _img_transform(self, image):
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        image = image_transform(image)
        return image
