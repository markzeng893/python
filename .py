import numpy as np
import scipy.io
import matplotlib.pyplot as plt


image = train_x[0]
image = np.reshape(image, (32, 32))
plt.imshow(image)
plt.show()