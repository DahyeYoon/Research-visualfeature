from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


base_model = VGG16(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)

img_path = 'dog.jpg'
im = Image.open(img_path)
# im.show()
# im.thumbnail(size, Image.ANTIALIAS)
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
#
# np.save('features.npy', features)
#
# # Show one of feature maps
# img = Image.fromarray(features[0,2], 'RGB')
# img.save('feature_map_figure.png')
# img.show()

# plt.imshow(features[0,1])
# plt.show()
# plt.savefig('feature_fig')


print "finish!"