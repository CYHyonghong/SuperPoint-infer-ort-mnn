import onnxruntime as ort
import numpy as np
import cv2


def main():
    ort_session = ort.InferenceSession('./weights/superpoint_v1.onnx')
	
    input_name = ort_session.get_inputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]
    print(f'output_names: {output_names}')

    for i in range(len(ort_session.get_inputs())):
        print(ort_session.get_inputs()[i].name, ort_session.get_inputs()[i].shape)

    for i in range(len(ort_session.get_outputs())):
        print(ort_session.get_outputs()[i].name, ort_session.get_outputs()[i].shape)

    inp = np.zeros([1, 1, 416, 416], dtype=np.float32)
    outs = ort_session.run(output_names, input_feed={input_name: inp})

    print(f'len(outs) = {len(outs)}')
    print(f'semi shape: {outs[0].shape}')
    print(f'desc shape: {outs[1].shape}')


if __name__ == '__main__':
    pass