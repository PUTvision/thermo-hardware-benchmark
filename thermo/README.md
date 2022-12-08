# Neural Network Method for Presence Monitoring in Thermal Images

This directory contains source code enabling the reproduction of a neural network model for presence monitoring tasks in low-resolution thermal images. It is required to download [Thermo Presence data](https://github.com/PUTvision/thermo-presence/tree/master/dataset/hdfs) (in h5 format) directly to [data](./data) folder before starting the training process. In order to do so, run [`get_data.sh`](./get_data.sh) script from the console.

```shell
bash get_data.sh
```

## Model Training

To train the density estimation model first check the parameters configuration in [config.yaml](./configs/config.yaml) file. Then call [train.py](./train.py) script as shown below:

```shell
python train.py -p configs/config.yaml
```

The conducted experiments were tracked using [Neptune](https://neptune.ai/) platform. If one has an account and wants to use this experiment tracker should provide the below data within the `configs/credentials.yaml` file and then run an experiment with the `log_neptune` flag.

```yaml
neptune:
  project: <Project Name>
  api_token: <Project API Token>
```

```console
$ python train.py --help

Usage: train.py [OPTIONS]

Options:
  -p, --config_path TEXT  Path to file with config
  --log_neptune           Use Neptune experiment tracker
  --help                  Show this message and exit.
```


### Neural Network Architecture

The neural network is based on U-Net architecture with shallow structure, single-channel input, and output. The proposed model has only 46 577 parameters.

<p align="center">
  <img width='800px' src="../README/nn_architecture.png" />
</p>


## Optimization and quantization

Neural network optimization, quantization, and conversion processes are described in the [convert_optimize_quantize.ipynb](./convert_optimize_quantize.ipynb) Jupyter Notebook.


## Metrics

<div align="center">

| **Metric Name**   | MAE    | MSE    | Counting MAE | Counting MSE | Counting MRAPE [%] | Accuracy | F1 Score | Model size [kB] |
|:-----------------:|:------:|:------:|:------------:|:------------:|:------------------:|:--------:|:--------:|:---------------:|
| **TF FP32**       | 0.1057 | 0.0323 | 0.0226       | 0.0234       | 0.81               | 0.9778   | 0.9782   | 667             |

</div>

<p align="center">
  <img width='500px' src="../README/confusion_matrix.png" />
</p>
