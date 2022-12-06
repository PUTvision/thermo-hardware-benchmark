import struct
from pathlib import Path

import click
import numpy as np
import serial
from tqdm import tqdm


@click.command()
@click.option('--test-input-path', help='Path to npy file with test input', type=click.Path(exists=True, file_okay=True))
@click.option('--output-path', help='Path for output file with results', type=click.Path())
@click.option('--serial-port', help='Serial port', type=str, default='/dev/ttyUSB1')
@click.option('--baudrate', help='Baud rate value', type=int, default=115200)
def main(test_input_path, output_path, serial_port, baudrate):
    test_input = np.load(Path(test_input_path))
    print(f'Test array shape: {test_input.shape}')

    input_scale = 0.0038248365744948387
    input_zero_point = -128
    output_scale = 0.00815824419260025
    output_zero_point = -127

    time_bytes = 4
    results_bytes = 768

    time_outputs = []
    result_outputs = []
    
    with serial.Serial(serial_port, baudrate=baudrate, timeout=5) as s:
        print(f'Serial name: {s.name}\nSerial baudrate: {s.baudrate}')

        arr = test_input[0].flatten() / input_scale
        s.write(arr.astype(np.uint8).tobytes())

        # check connection and read selected number of bytes
        _ = s.read(time_bytes+results_bytes)

        for arr_num in tqdm(range(test_input.shape[0])):
            arr = test_input[arr_num].flatten() / input_scale
            s.write(arr.astype(np.uint8).tobytes())

            time_serial_output = s.read(time_bytes)
            results_serial_output = s.read(results_bytes)

            time_output = struct.unpack("f", time_serial_output)
            result_output = np.array(list(results_serial_output)) * output_scale

            time_outputs.append(time_output)
            assert len(result_output) == 768
            result_outputs.append(result_output)

    result_outputs = np.vstack(result_outputs)
    time_outputs = np.vstack(time_outputs)
    np.save(Path(output_path), result_outputs)
    np.save(Path(output_path.replace('.npy', '_inference_time.npy')), time_outputs)
    print(f'Results saved in {output_path}\nEvaluation process finished successfully!')


if __name__ == '__main__':
    main()
