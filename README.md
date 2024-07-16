# ScaleGraphs
Graph Transformation via Scale Invariance

# Graph Transformation via Scale Invariance of Node classification

Implementation of paper [Graph Transformation via Scale Invariance of Node classification](??).

![]()

## Requirements

This repository has been tested with the following packages:

- Python == 3.9 or 3.10
- PyTorch == 2.1.2
- PyTorch Geometric == 2.4.0
- torch-scatter==2.1.2
- torch-sparse==0.6.18

## Installation Instructions

1. Follow the official instructions to install [PyTorch](https://pytorch.org/get-started/previous-versions/).
2. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
3. For `pytorch-scatter` and `pytorch-sparse`, download the packages from [this link](https://pytorch-geometric.com/whl/torch-2.3.0%2Bcu121.html) according to your PyTorch, Python, and OS version. Then, use `pip` to install them.

By following these steps, you can resolve compatibility issues and avoid segmentation faults.


## Important Hyper-parameters
### Dataset
Specify the name of the dataset you want to use. The available datasets are categorized as follows:

- **Directed Datasets**:
  - **Assortative Graph**:
    - `citeseer_npz/`
    - `cora_ml/`
    - `WikiCS/`
    - `telegram/telegram`
    - `dgl/cora`
    - `dgl/pubmed`
  
  - **Disassortative Graph**:
    - `WebKB/texas`
    - `WebKB/Cornell`
    - `WebKB/wisconsin`
    - `film/`
    - `WikipediaNetwork/squirrel`
    - `WikipediaNetwork/chameleon`

- **Undirected Datasets**:
  - **Citation**:
    - `Cora/`
    - `CiteSeer/`
    - `PubMed/`
  - **CoPurchase**:
    - `dgl/computer`
    - `dgl/photo`
  
  - **Coauthor**:
    - `dgl/coauthor-cs`
    - `dgl/coauthor-ph`
  
  - **Fraud Review**:
    - `dgl/Fyelp`
    - `dgl/Famazon`
  
  - **Others**:
    - `dgl/yelp`
    - `dgl/reddit`
    - `WikiCS_U`
  

### GNN Backbone
Choose the GNN backbone to use. The available options are:
- `GCN`
- `GAT`
- `SAGE`
- `Cheb`
- `Mag`
- `Sig`

Please refer to [args.py](args.py) for the full hyper-parameters.

## How to Run

Pass the above hyper-parameters to `main.py`. For example:

```
python main.py --dataset cora_ml/  --net GCN  --layer=3 --Dataset='cora_ml/'
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


