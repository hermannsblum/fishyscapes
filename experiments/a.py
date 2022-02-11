from deeplabv3 import DeepWV3PlusTH
import torch
model = DeepWV3PlusTH(num_classes=19).cuda()
model.eval()
model(torch.zeros(1024, 2048, 3).cuda().float().unsqueeze(0).permute(0, 3, 1, 2) / 255.)