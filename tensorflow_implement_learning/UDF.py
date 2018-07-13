import os
from skimage import data

def load_data( root_path):
    directories = [d for d in os.listdir(root_path)
                   if os.path.isdir(os.path.join(root_path, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(root_path, d)
        file_names = [os.path.join(label_directory, f)
                       for f in os.listdir(label_directory)
                       if f.endswith('.ppm')]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))

    return images, labels