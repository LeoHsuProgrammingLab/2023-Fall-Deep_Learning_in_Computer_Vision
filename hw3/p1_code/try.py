import torch
x = torch.rand(5, 3)
print(x)
print(torch.version.cuda)
print(torch.cuda.get_arch_list())