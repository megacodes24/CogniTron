import torch
print(torch.__version__)
print("cuda?", torch.cuda.is_available())
x = torch.rand(1000, 1000).cuda()   # if you have GPU
print(x.mean())
