# Raspberry Pi 4B

This document contains configuration steps, source code, and evaluation of Raspberry Pi 4B platform, with and without co-processors in the form of Coral USB Accelerator and Intel Neural Compute Stick 2, to investigate the performance of the proposed neural network model on the Thermo Presence dataset.

## Configuration

### Raspberry Pi 4B


### Coral USB Accelerator


### Intel Neural Compute Stick 2


## Source Code


## Evaluation

```console
python evaluate_rpi4b.py --inference_type tflite --model-path ./models/model_int8.tflite --test-input-path ../data/test_input_2347.npy --output-path ./raspberry_pi_4b_tflite_results.npy
```

```console
$ python evaluate_rpi4b.py --help

Usage: evaluate_rpi4b.py [OPTIONS]

Options:
  --inference-type [tflite|edgetpu|myriad]
                                  Inference framework (device)
  --model-path PATH               Path to model
  --test-input-path PATH          Path to npy file with test input
  --output-path PATH              Path for output file with results
  --help                          Show this message and exit.
```
