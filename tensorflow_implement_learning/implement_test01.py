import matplotlib.pyplot as plt
import numpy as np
import tensorflow_implement_learning.UDF as udf

images_test, labels_test = np.array(udf.load_data('./Training'))
# images_train = np.array(udf.load_data('./Training'))

print('image:')
print('ndim:', images_test.ndim)
print('size:', images_test.size)
print('label:')
print('ndim:', labels_test.ndim)
print('size:', labels_test.size)
print(len(set(labels_test)))
print(labels_test)
# 内存布局（memory layout）
print('flags:', images_test.flags)
# 一个数组元素的字节长度
print('itemsize:', images_test.itemsize)
# 总消耗字节数
print('nbytes:', images_test.nbytes)

# plt.hist(x, bins = 100, color = "red", normed = False,)
plt.hist(labels_test, 62, color='green')
plt.show()



