from deeplabv3 import DeepWV3PlusTH
import torch
model = DeepWV3PlusTH(num_classes=19).cuda().half()
model.eval()
from time import time_ns

def wrapper(image):
    with torch.no_grad():
        image = image.numpy().astype('uint8')
        with torch.no_grad():
            img = torch.from_numpy(image).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.
            # img should be in [0, 1] and of shape 1x3xHxW
            logit, logit_ood = model(img.cuda().half())
        out = torch.nn.functional.softmax(logit_ood, dim=1)
        p1 = torch.logsumexp(logit, dim=1)  # ln hat_p(x|din)
        p2 = out[:, 1]  # p(~din|x)
        probs = (- p1) + p2.log()  # - ln hat_p(x|din) + ln p(~din|x)
        probs = probs[0].cpu()
        # output is HxW
        return probs



x = (torch.rand(1024, 2048, 3)*255).long()
t0 = time_ns()
wrapper(x)
t1 = time_ns()
print('total time (ms):', (t1 - t0)/10**6)