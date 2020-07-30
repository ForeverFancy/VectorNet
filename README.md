# VectorNet

[![HitCount](http://hits.dwyl.com/ForeverFancy/VectorNet.svg)](http://hits.dwyl.com/ForeverFancy/VectorNet)

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

使用示例：

```
python3 run.py --root_dir ../forecasting_sample/data/ --epochs 50 --feature_path ./save/features/ --logging_steps 50 --train_batch_size=16 --enable_logging
```
