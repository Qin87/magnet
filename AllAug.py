import torch

import args
import main

args = args.parse_args()
seed = args.seed
cuda_device = args.device

if torch.cuda.is_available():
    print("CUDA Device Index:", cuda_device)
    device = torch.device("cuda:%d" % cuda_device)
else:
    print("CUDA is not available, using CPU.")
    device = torch.device("cpu")

list_com = [(True, 1), (True, 2), (True, 4), (True, 21), (True, 22), (True, 23), (False, 0), (True, 20)]
for i in range(len(list_com)):
    (args.WithAug, args.AugDirect) = list_com[i]
    main