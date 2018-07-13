# Import `matplotlib`import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import random
from tensorflow_implement_learning.UDF import load_data

images, labels = load_data('./Training')

# Determine the (random) indexes of the images traffic_signs = [300, 2250, 3650, 4000]
traffic_signs = []
for i in range(4):
    traffic_signs.append(int(random.uniform(0, len(images))))

# Fill out the subplots with the random images and add shape,
# min and max values for i in range(len(traffic_signs)):
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i + 1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
                                                  images[traffic_signs[i]].min(),
                                                  images[traffic_signs[i]].max()))
