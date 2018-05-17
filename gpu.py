print('checking what CPU / GPU you have available for Keras & Tensorflow')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

