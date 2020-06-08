# VectorNet

Pytorch implementation of paper "VectorNet: Encoding HD maps and Agent Dynamics from Vectorized Representation".

目录结构：

```
.
├── README.md
├── data_process.py
├── docs
│   └── dataset.md
├── layers.py
├── model.py
├── run.py
└── save
    ├── features
    └── models
```

使用方法：

```
python3 run.py --root_dir PATH_TO_RAW_DATA --epochs 50
```

Note:

- 目前只用了数据集的 examples 中的 4 个样本进行训练，1 个样本进行验证；
- 目前没有把整个模型的三部分压到一个文件中，有时间可以进行重构。