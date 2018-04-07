"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import os
sys.path.insert(1, './src')
from crfrnn_model import get_crfrnn_model_def
import util
import numpy as np
np.set_printoptions(threshold='nan')

def save_mask(model, input_file, output_file):
    img_data, img_h, img_w = util.get_preprocessed_image(input_file)
    probs = model.predict(img_data, verbose=False)[0, :, :, :]

    segmentation = util.get_label_image(probs, img_h, img_w)
    segmentation.save(output_file)


def main():
    base_path = sys.argv[1]
    blurred_imgs_dir = os.path.join(base_path, 'blurred')
    normall_imgs_dir = os.path.join(base_path, 'normal')

    # Download the model from https://goo.gl/ciEYZi
    saved_model_path = 'crfrnn_keras_model.h5'

    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)

    images = os.listdir(normall_imgs_dir)
    input_images = [os.path.join(base_path, 'normal', img) for img in images]
    target_images = [os.path.join(base_path, 'blurred', img).replace('jpg', 'png') for img in images]
    bar_len = 30
    for idx, (in_file, out_file) in enumerate(zip(input_images, target_images)):

        # progress bar
        sys.stdout.write('\r')
        percent = (idx+1.)/len(input_images)
        done_len = int(percent * bar_len)
        args = ['='*done_len, ' '*(bar_len-done_len-1), percent*100]
        sys.stdout.write('[{}>{}] {:.0f}%'.format(*args))
        sys.stdout.flush()

        save_mask(model, in_file, out_file)

    sys.stdout.write('\n')

if __name__ == '__main__':
    main()
