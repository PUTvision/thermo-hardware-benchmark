import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from pathlib import Path

import yaml
import click
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

from models.unet import UNet
from metrics import CountAccuracy, CountMAE, CountMSE, CountMeanRelativeAbsoluteError
from data_generator.thermal_data_generator import ThermalDataset
from utils.model_utils import check_model_prediction, evaluate


@click.command()
@click.option('-p', '--config_path', help='Path to file with config', type=str)
@click.option('--log_neptune', is_flag=True, help='Use Neptune experiment tracker')
def train(config_path, log_neptune):
    print(f'TensorFlow version: {tf.__version__}')
    print(tf.config.list_physical_devices('GPU'))

    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    model_save_name = config['model']['model_save_name']

    if log_neptune:
        with open('./configs/credentials.yaml', 'r') as stream:
            credentials = yaml.safe_load(stream)

        run = neptune.init(
            project=credentials['neptune']['project'],
            api_token=credentials['neptune']['api_token']
        )

        run["model/config"] = {
            "model_name": config['model']['model_type'],
            "input_shape": config['model']['input_shape'],
            "batch_norm": config['model']['batch_norm'],
            "conv_transpose": config['model']['conv_transpose'],
            "squeeze": config['model']['squeeze'],
            "double_double_conv": config['model']['double_double_conv'],
            "filters": config['model']['in_out_filters'],
            "optimizer": config['model']['optimizer'],
            "loss": config['model']['loss'],
            "metrics": config['model']['metrics'],
            "batch_size": config['model']['batch_size'],
            "epochs": config['model']['epochs'],
            "early_stopping_patience": config['model']['early_stopping_patience']
        }

        run["model/config_file"].upload(config_path)

    ppw = config['dataset']['sum_of_values_for_one_person']
    
    train_data_gen = ThermalDataset(data_path=Path('./data'), sequences_names=config['dataset']["training_dirs"],
                                    person_point_weight=ppw, batch_size=config['model']['batch_size'], augment=True)
    val_data_gen = ThermalDataset(data_path=Path('./data'), sequences_names=config['dataset']["validation_dirs"],
                                  person_point_weight=ppw, batch_size=config['model']['batch_size'])
    test_data_gen = ThermalDataset(data_path=Path('./data'), sequences_names=config['dataset']["test_dirs"],
                                   person_point_weight=ppw, batch_size=1)

    model = UNet(
        input_shape=config['model']['input_shape'],
        in_out_filters=config['model']['in_out_filters'],
        batch_norm=config['model']['batch_norm'],
        conv_transpose=config['model']['conv_transpose'],
        squeeze=config['model']['squeeze'],
        double_double_conv=config['model']['double_double_conv']
    )

    metrics = config['model']['metrics']
    metrics=[
        *metrics,
        CountAccuracy(person_point_weight=ppw, name='count_acc'),
        CountMAE(person_point_weight=ppw, name='count_mae'),
        CountMSE(person_point_weight=ppw, name='count_mse'),
        CountMeanRelativeAbsoluteError(person_point_weight=ppw, name='count_mrae')
    ]

    model.compile(
        optimizer=config['model']['optimizer'],
        loss=config['model']['loss'],
        metrics=metrics
    )

    if log_neptune:
        with open('model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        run["model/model_summary"].upload("./model_summary.txt")

    patience = config['model']['early_stopping_patience']
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience//2)
    ]

    if log_neptune:
        callbacks.append(NeptuneCallback(run=run, base_namespace='model'))

    history = model.fit(
        x=train_data_gen,
        batch_size=config['model']['batch_size'],
        epochs=config['model']['epochs'],
        callbacks=callbacks,
        validation_data=val_data_gen
    )

    eval_metrics = model.evaluate(test_data_gen)

    model.save(f'./{model_save_name}.h5')

    print('Test data:')
    test_acc, test_f1, test_cm = evaluate(f'./{model_save_name}.h5', 'keras', test_data_gen, config)
    print('Train data:')
    train_acc, train_f1, train_cm = evaluate(f'./{model_save_name}.h5', 'keras', train_data_gen, config)
    print('Validation data:')
    val_acc, val_f1, val_cm = evaluate(f'./{model_save_name}.h5', 'keras', val_data_gen, config)
    for cm, cm_name in [(test_cm, 'test'), (train_cm, 'train'), (val_cm, 'val')]:
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, cmap="Blues", fmt='0.0f',
            xticklabels=np.arange(config['dataset']["max_people_count"]+1),
            yticklabels=np.arange(config['dataset']["max_people_count"]+1)
        )
        plt.xlabel('Labels')
        plt.ylabel('Predictions')
        plt.savefig(f'./{cm_name}_confusion_matrix.png')
    count, output_frame = check_model_prediction(model, config)
    print(f'People count for raw frame: {round(count, 4)}')
    plt.figure(figsize=(16, 12))
    plt.imshow(output_frame[0][..., 0])
    plt.savefig('./output_frame.png')

    if log_neptune:
        for i, metric in enumerate(eval_metrics):
            run["model/eval/{}".format(model.metrics_names[i])] = metric

        run["model/model"].upload(f'./{model_save_name}.h5')

        run["model/eval/accuracy"] = test_acc
        run["model/eval/F1"] = test_f1
        run["model/eval/raw_confusion_matrix"] = test_cm
        run["model/eval/confusion_matrix"].upload(f'./test_confusion_matrix.png')
        run["model/eval/output_frame_count"] = count
        run["model/eval/raw_output_frame"] = output_frame
        run["model/eval/output_frame"].upload(f'./output_frame.png')

        run["model/train/accuracy"] = train_acc
        run["model/train/F1"] = train_f1
        run["model/train/raw_confusion_matrix"] = train_cm
        run["model/train/confusion_matrix"].upload(f'./train_confusion_matrix.png')

        run["model/val/accuracy"] = val_acc
        run["model/val/F1"] = val_f1
        run["model/val/raw_confusion_matrix"] = val_cm
        run["model/val/confusion_matrix"].upload(f'./val_confusion_matrix.png')

    del model


if __name__ == '__main__':
    train()
