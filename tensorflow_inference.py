# inference with tensorflow
from __future__ import absolute_import
import argparse
import numpy as np
from six import text_type as _text_type
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument('-n', type=_text_type, default='kit_doc',
                    help='Network structure name')

parser.add_argument('-w', type=_text_type, required=True,
                    help='Network weights file name')

parser.add_argument('-l', nargs='*', help='Layers to select')

parser.add_argument('--image', '-i',
                    type=_text_type, help='Test image path.',
                    default="img/41004.jpg"
                    )

args = parser.parse_args()

# import converted model
if args.n == 'doc':
    import model.doc as m
elif args.n == 'hed':
    import model.hed as m

model_converted = m.KitModel(args.w)

# load img with BGRTranspose=True
img = image.load_img(args.image)
img = image.img_to_array(img)
img = img[..., ::-1]  # needed?

input_data = np.expand_dims(img, 0)

# inference with tensorflow
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    if args.l is None:
        input_tf, model_tf = model_converted
    else:
        input_tf, _ = model_converted
        device_append = ':0'
        model_tf = ()
        # select which layers to output here
        for layer in args.l:
            model_tf += (sess.graph.get_tensor_by_name(layer + device_append),)
    predict = sess.run(model_tf, feed_dict={input_tf: input_data})

# import matplotlib.pyplot as plt

# # Plot outputs @ Different scales
# for layer_idx in range(len(predict)):
#     plt.figure()
#     plt.imshow(predict[layer_idx].squeeze())
#     plt.title("Output of layer {0}, [idx={1}]".format(
#         model_tf[layer_idx].name, layer_idx))

# plt.show()
