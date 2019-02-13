# ICLR Reproducibility Challenge 2019

Reproducibility Report: [ Issue on Github ](https://github.com/reproducibility-challenge/iclr_2019/issues/91?fbclid=IwAR3fX_jrgU6VryLYCkpfVwXAcMuUsiKTo_mLq_5zaZBr0BBifDy9SFUln3Y)
Paper: [Paper | OpenReview ](https://openreview.net/forum?id=H1lGHsA9KX)

The CNN for MNIST is described in `neuralNetwork.py`, and for CIFAR 10 in `neuralNetworkCIFAR.py`.


### Running Experiments (MNIST)

The experiments can be run by: 
```python run.py [--optimizer <name>] [--rerun count] [--epoch count]```

Results are stored in `data/experiment_result.pickle`.

### Visualizing results
`heatmap.py` is a script that opens the data (created after running `run.py`) `data/experiment_result.pickle` and creates the heatmaps found in the report

`generate_tables.py` is a script that opens the data (created after running `run.py`) `data/experiment_result.pickle` and generates the accuracy table found in the report
