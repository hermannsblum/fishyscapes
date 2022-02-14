from deeplabv3 import DeepWV3PlusTH
import torch
model = DeepWV3PlusTH(num_classes=19).cuda()
model.eval()
from time import time_ns

def wrapper(image):
    with torch.no_grad():
        t0 = time_ns()
        image = image.numpy().astype('uint8')
        with torch.no_grad():
            t1 = time_ns()
            img = torch.from_numpy(image).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.
            t2 = time_ns()
            # img should be in [0, 1] and of shape 1x3xHxW
            logit, logit_ood = model(img.cuda())
            torch.cuda.synchronize()
            t3 = time_ns()
        out = torch.nn.functional.softmax(logit_ood, dim=1)
        p1 = torch.logsumexp(logit, dim=1)  # ln hat_p(x|din)
        p2 = out[:, 1]  # p(~din|x)
        probs = (- p1) + p2.log()  # - ln hat_p(x|din) + ln p(~din|x)
        print(probs.dtype)
        probs = probs[0].cpu()
        t4 = time_ns()
        # output is HxW
        print('img->numpy', (t1 - t0) / 10 ** 6)
        print('numpy->torch', (t2 - t1) / 10 ** 6)
        print('model', (t3 - t2) / 10 ** 6)
        print('postprocess', (t4 - t3)/10**6)
        return probs

x = (torch.rand(1024, 2048, 3)*255).long()
t0 = time_ns()
wrapper(x)
t1 = time_ns()
print('total time (ms):', (t1 - t0)/10**6)