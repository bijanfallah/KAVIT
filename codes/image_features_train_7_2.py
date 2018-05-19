'''
Program to extract the image features using the Keras 
pretrained models:
https://www.kaggle.com/lemonkoala/pretrained-keras-models-symlinked-not-copied

'''
file_name='features_quality'

import matplotlib.pyplot as plt
import numpy             as np
import os
from keras.preprocessing               import image
from keras.applications.resnet50       import ResNet50, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
# activate the next 9 lines only once:
#inputs_dir = "/home/bkd/Softwares/keras-pretrained-models"
#models_dir = os.path.expanduser(os.path.join("~", ".keras", "models"))
#os.makedirs(models_dir)
#for file in os.listdir(inputs_dir):
#    if file.endswith(".json") or file.endswith(".h5"):
#        os.symlink(
#            os.path.join(inputs_dir, file),
#            os.path.join(models_dir, file)
#        )
import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
# read the train data:
images_path = '/media/bkd/ext01/train_figs/train_7/'
imgs = os.listdir(images_path)
features = pd.DataFrame()
features['image'] = imgs
# ---------------------------------------------------




sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

%matplotlib inline
#the model is inception V3
inception_model = inception_v3.InceptionV3(weights='imagenet')
def image_classify(model, pak, img, top_n=3):
    """Classify image and return top matches."""
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = pak.preprocess_input(x)
    preds = model.predict(x)
    return pak.decode_predictions(preds, top=top_n)[0]
    
images_dir = '/media/bkd/ext01/train_figs/train_7/'
image_files = [x.path for x in os.scandir(images_dir)]
def incept_class(incept):
    if incept[0][2] > .8 :
        return 1
    if (incept[0][2] < .8) &  (incept[0][2] > .5):
        return 2
    if (incept[0][2] < .5):
        return 3
        
# loop over all images:
class = []
for i in range(0,len(image_files)):
    img = Image.open(image_files[i])
    inception_preds = image_classify(inception_model, inception_v3, img)
    class.extend(incept_class(inception_preds))
    
features['class_qual'] =  class
features.to_pickle(file_name)      

