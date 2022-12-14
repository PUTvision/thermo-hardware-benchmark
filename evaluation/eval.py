from pathlib import Path

import click
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


OUT_ZERO_POINT = -127
OUT_SCALE = 0.00815824
SUM_OF_VALUES_FOR_ONE_PERSON = 51.35


@click.command()
@click.option('--gt-test-output-path', help='Path to npy file with test input', type=click.Path(exists=True, file_okay=True))
@click.option('--estimated-test-output-path', help='Path to npy / npz file with estimated results', type=click.Path(exists=True, file_okay=True))
@click.option('--stm', help='Preprocess STM32 output data', is_flag=True)
@click.option('--int8', help='Preprocess STM32 output data for int8 model', is_flag=True)
def eval(gt_test_output_path, estimated_test_output_path, stm, int8):
    gt_test_output = np.load(Path(gt_test_output_path)).reshape((-1, 24, 32))

    if stm:
        estimated_test_output = np.load(Path(estimated_test_output_path))
        estimated_test_output = estimated_test_output['c_outputs_1'].reshape((-1, 24, 32))

        if int8:
            estimated_test_output = (estimated_test_output - OUT_ZERO_POINT) * OUT_SCALE
    else:
        estimated_test_output = np.load(Path(estimated_test_output_path)).reshape((-1, 24, 32))

    assert estimated_test_output.shape == gt_test_output.shape, f"Shapes are different {estimated_test_output.shape} != {gt_test_output.shape}"
    print(f'Loaded test gt file and neural network estimation with shape: {gt_test_output.shape}')

    correct_count = 0
    tested_frames = 0
    number_of_frames_with_n_persons = {}
    number_of_frames_with_n_persons_predicted_correctly = {}

    mae_sum = 0
    mse_sum = 0
    count_mae_sum = 0
    count_mse_sum = 0
    count_mrae_sum = 0

    vec_real_number_of_persons = []
    vec_predicted_number_of_persons = []

    for idx, gt_mask in enumerate(tqdm(gt_test_output)):
        predicted_img = np.array(estimated_test_output[idx])
        
        pred_people = np.sum(predicted_img) / SUM_OF_VALUES_FOR_ONE_PERSON
        pred_label = round(pred_people)
        
        true_label = round(np.sum(gt_mask) / SUM_OF_VALUES_FOR_ONE_PERSON)
        
        error = abs(pred_people - true_label)
        mae_sum += error
        mse_sum += error*error

        rounded_error = abs(pred_label - true_label)
        count_mae_sum += rounded_error
        count_mse_sum += rounded_error*rounded_error
        if rounded_error != 0:
            count_mrae_sum += (rounded_error + 1) / (true_label + 1) if true_label == 0 else rounded_error / true_label

        number_of_frames_with_n_persons[pred_label] = number_of_frames_with_n_persons.get(pred_label, 0) + 1

        if true_label == pred_label:
            correct_count += 1
            number_of_frames_with_n_persons_predicted_correctly[pred_label] = number_of_frames_with_n_persons_predicted_correctly.get(
                            pred_label, 0) + 1

        vec_real_number_of_persons.append(true_label)
        vec_predicted_number_of_persons.append(pred_people)
        tested_frames += 1
        
    mae = mae_sum / tested_frames
    mse = mse_sum / tested_frames
    count_mae = count_mae_sum / tested_frames
    count_mse = count_mse_sum / tested_frames
    count_mrae = count_mrae_sum / tested_frames

    model_accuracy = correct_count / tested_frames
    model_f1_score = f1_score(vec_real_number_of_persons, 
                            np.round(vec_predicted_number_of_persons).astype(int), 
                            average='weighted')

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'Count MAE: {count_mae}')
    print(f'Count MSE: {count_mse}')
    print(f'Count count_mrae: {count_mrae}')
    print(f'Accuracy: {model_accuracy}')
    print(f'F1 Score: {model_f1_score}')


if __name__ == '__main__':
    eval()
