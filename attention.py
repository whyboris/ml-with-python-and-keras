print('visualize what convnet focuses on')

from hack import hack
hack()

from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

import numpy as np 

img_path = '../catsdogssmall/test/cats/cat.1717.jpg'

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

predictions = model.predict(x)
print('predicted: ', decode_predictions(predictions, top=3)[0])

print(np.argmax(predictions[0])) # output of this needs to be on the next line 

cat_output = model.output[:, 285] # this number comes from the output of previous line

last_conv_layer = model.get_layer('block3_conv3')

from keras import backend as K

grads = K.gradients(cat_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range (255):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

import matplotlib.pyplot as plt

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

import cv2

img = cv2.imread(img_path)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img

cv2.imwrite('../heatmap.jpg', superimposed_img)

