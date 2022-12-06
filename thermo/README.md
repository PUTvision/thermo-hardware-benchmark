# Neural Network Method for Presence Monitoring in Thermal Images

This directory contains source code enabling the reproduction of a neural network model for presence monitoring tasks in low-resolution thermal images. It is required to download [Thermo Presence data](https://github.com/PUTvision/thermo-presence/tree/master/dataset/hdfs) (in h5 format) directly to [data](./data) folder before starting the training process. In order to do so, run [`get_data.sh`](./get_data.sh) script from the console.

```console
bash get_data.sh
```

## Model Training

To train the density estimation model first check the parameters configuration in [config.yaml](./configs/config.yaml) file. Then call [train.py](./train.py) script as shown below:

```console
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

<p align="center">
  <img width='800px' src="../README/confusion_matrix.png" />
</p>

## Optimization and quantization

