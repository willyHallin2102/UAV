import tensorflow as tf
import sionna as sn
import numpy as np


print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices("GPU"))


print("Sionna imported successfully")
print(sn.__version__)

print("Numpy successfully installed")
print(np.__version__)

# Tiny tensor test (checks TF + GPU execution path)
x = tf.random.normal([2, 3])
y = tf.matmul(x, tf.transpose(x))

print("Tensor test output:\n", y)
print("Script finished successfully")