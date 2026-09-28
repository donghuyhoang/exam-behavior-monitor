import torch
ckpt = torch.load('r50_deformable_detr-checkpoint.pth', map_location='cpu')
model = ckpt.get('model', ckpt)
print('Số classes:', model['class_embed.weight'].shape[0]) 
# Sẽ in ra 91
