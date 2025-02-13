import tensorflow as tf
import tensorflow as tf
import numpy as np
import matplotlib
import pandas as pd
import h5py
import PIL

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))


# Print versions to confirm installation
print("TensorFlow version:", tf.__version__)
print("Numpy version:", np.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Pandas version:", pd.__version__)
print("PIL version:", PIL.__version__)
print("h5py version:", h5py.__version__)