import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('./model/comp6--10--0.7989.hdf5')

model.pred
