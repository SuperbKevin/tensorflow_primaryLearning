# Import the `transform` module from `skimage`
from skimage import transform
# Import `rgb2gray` from `skimage.color`
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_implement_learning.UDF import load_data

random_num = [300, 2250, 3650, 4000]
images, labels = load_data('./Training')
four_images = []
counter = 0

for i in range(counter, len(random_num)):
    four_images.append(images[random_num[i]])

    plt.subplot(4, 4, i+1)
    # plt.axis('off')
    plt.imshow(images[random_num[i]])
    plt.subplots_adjust(wspace=0.5)
    print("shape: {0}, min: {1}, max: {2}".format(images[random_num[i]].shape,
                                                  images[random_num[i]].min(),
                                                  images[random_num[i]].max()))

counter += len(random_num)
# print("image:", "\n", images[random_num[0]])

# Rescale the images in the `images` array
# images28_all = [transform.resize(image, (28, 28)) for image in images]
images28 = [transform.resize(image, (28, 28)) for image in four_images]
print('after resize:')
for i in range(len(images28)):
    plt.subplot(4, 4, counter+i+1)
    plt.imshow(images28[i])
    plt.subplots_adjust(wspace=0.5)
    print("shape: {0}, min: {1}, max: {2}".format(images28[i].shape,
                                                  images28[i].min(),
                                                  images28[i].max()))

counter += len(images28)
# print("image:", "\n", images[random_num[0]])

# Convert `images28` to an array
images28 = np.array(images28)
# Convert `images28` to gray scale
gray_images28 = rgb2gray(images28)

for i in range(len(gray_images28)):
    plt.subplot(4, 4, counter+i+1)
    plt.imshow(gray_images28[i], cmap="gray")
    plt.subplots_adjust(wspace=0.5)

    plt.subplot(4, 4, counter + i + 1+4)
    plt.imshow(gray_images28[i])
    plt.subplots_adjust(wspace=0.5)

    print("shape: {0}, min: {1}, max: {2}".format(gray_images28[i].shape,
                                                  gray_images28[i].min(),
                                                  gray_images28[i].max()))

# Show the plot plt.show()
plt.show()
