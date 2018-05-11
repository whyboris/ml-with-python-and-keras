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

print(np.argmax(predictions[0]))

