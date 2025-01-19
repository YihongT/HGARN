# HGARN

Source codes for [Activity-aware human mobility prediction with hierarchical graph attention recurrent network.](https://arxiv.org/pdf/2210.07765.pdf), published in IEEE Transactions on Intelligent Transportation Systems ([TITS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6979)) in 2024.

## Requirements

- python == 3.6
- torch == 1.7.0+cu110
- mpu == 0.23.1

See [requirements.txt](https://github.com/YihongT/DASTNet/blob/master/requirements.txt) for more details.

## Datasets

NYC and Tokyo Check-in Dataset.

Please refer to this [repo](https://sites.google.com/site/yangdingqi/home/foursquare-dataset).

## Run

```
python train.py
```

## Reference

Please cite our paper if you use the model in your own work:

```
@article{tang2024activity,
  title={Activity-aware human mobility prediction with hierarchical graph attention recurrent network},
  author={Tang, Yihong and He, Junlin and Zhao, Zhan},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgments

We refer to some of the data processing codes in this [repo](https://github.com/ywhuazhong/CSLSL).
