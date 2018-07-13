# Import the `pyplot` module as `plt`import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tensorflow_implement_learning.UDF import load_data

# Get the unique labels unique_labels = set(labels)
images, labels = load_data('./Training')
unique_labels = set(labels)

# Initialize the figure plt.figure(figsize=(15, 15))
plt.figure(figsize=(15, 15))

# Set a counter = 1
counter = 1

# For each unique label,for label in unique_labels:    
# You pick the first image for each label
for label in unique_labels:
    image = images[labels.index(label)]
    # Define 64 subplots
    plt.subplot(8, 8, counter)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    counter += 1
    # And you plot this first image
    plt.imshow(image)
# Show the plot plt.show()
plt.show()
