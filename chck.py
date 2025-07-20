import torch

ckpt = torch.load("data\cifar100\paco_ldam\ldam_ckpt.pth.tar")
print(ckpt.keys())
if "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]
print("\n".join(ckpt.keys()))