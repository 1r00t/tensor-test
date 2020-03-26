from matplotlib import image
import numpy as np


img = image.imread("images/sneakernike-28.png")
img = np.dot(img[..., :3], [0.299, 0.587, 0.144])
img = img / 255.0
print(img)
