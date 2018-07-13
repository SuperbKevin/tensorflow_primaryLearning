import matplotlib.image as mpimg
import tensorflow as tf

filename = "example_image.jpg"
image = mpimg.imread(filename)

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    result = sess.run(x)

    print(x.get_shape())