import torch

ckpt = torch.load("data/cifar100/gpaco_cifar100_r32/moco_ckpt.pth.tar")
print(ckpt.keys())
if "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]
print("\n".join(ckpt.keys()))