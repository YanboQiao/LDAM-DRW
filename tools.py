import torch

ckpt = torch.load('data/cifar100/gpaco_cifar100_r32/moco_ckpt.pth.tar', map_location='cpu')
sd = ckpt['state_dict']
new_sd = {}
for k, v in sd.items():
    if k.startswith('module.'):
        new_sd[k[7:]] = v
    else:
        new_sd[k] = v

# 保留原有其他key，只替换state_dict
ckpt['state_dict'] = new_sd
torch.save(ckpt, 'data/cifar100/gpaco_cifar100_r32/moco_ckpt_single.pth.tar')
