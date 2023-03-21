import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils.py import load_dataset

plt.show()


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()