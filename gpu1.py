import torch
device = torch.device("cuda:1")
try:
    a = torch.rand(10000, 10000, device=device)
    b = torch.mm(a, a)
    print("GPU1 works!")
except Exception as e:
    print("GPU1 test failed:", e)
