# ScaleGraphs
Graph Transformation via Scale Invariance

# Graph Transformation via Scale Invariance of Node classification

Implementation of paper [Graph Transformation via Scale Invariance of Node classification](??).

![]()

## Requirements

This repository has been tested with the following packages:

- Python >= 3.9
- PyTorch == 2.3.0
- PyTorch Geometric == 2.5.3

Please follow official instructions to install [Pytorch](https://pytorch.org/get-started/previous-versions/) and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
For pytorch-scatter,pytorch-sparse, download packages from https://pytorch-geometric.com/whl/torch-2.3.0%2Bcu121.html according to your PyTorch, Python and OS version. 
Then pip install them. By doing this, you can solve the compatibility issues, Segmentation fault.

## Important Hyper-parameters

- `--undirect_dataset`: name of the undirected dataset. Could be one of `['Cora', 'CiteSeer', 'PubMed', 'Coauthor-CS', 'Coauthor-physics']`. 
- `--Direct_dataset`: name of the directed dataset. Could be one of `['citeseer_npz/' , 'cora_ml/',  'WikiCS/', 'telegram/telegram']`. 
- `--net`: GNN backbone. Could be one of `['GCN, GAT, SAGE']`.
- 

Please refer to [args.py](args.py) for the full hyper-parameters.

## How to Run

Pass the above hyper-parameters to `main.py`. For example:

```
python main.py --dataset Cora  --net GCN  --layer=3 --IsDirectedData --Direct_dataset='cora_ml/'  --undirect_dataset="Cora"
```

## License
MIT License

## Contact 
Feel free to email (qj2004 [AT] hw.ac.uk) for any questions about this work.

## Acknowledgements

The code is implemented based on [GraphSHA](https://github.com/wenzhilics/GraphSHA), [DiGCN](https://github.com/flyingtango/DiGCN) and [MagNet](https://github.com/matthew-hirn/magnet).

## Citation

If you find this work is helpful to your research, please consider citing our paper:???

```

```


