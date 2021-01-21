#!/usr/bin/env python
# coding: utf-8

# # CS542 - Class Challenge - fine-grained classification of plants:
# 
# Our class challenge will consists of two tasks addressing an image recognition task where our dataset contains about 1K categories of plants with only about 250,000 images.  There will be two parts to this task:
# 
# 1. Image classification. Imagine we have cateloged all the plants we care to identify, now we just need to create a classifier for them! Use your skills from the supervised learning sections of this course to try to address this problem.
# 
# 2. Semi-Supervised/Few-Shot Learning.  Unfortunately, we missed some important plants we want to classify!  We do have some images we think contain the plant, but we have only have a few labels.  Our new goal is to develop an AI model that can learn from just these labeled examples.
# 
# Each student must submit a model on both tasks.  Students in the top 3 on each task will get 5% extra credit on this assignment.
# 
# This notebook is associated with the first task (image classification).
# 
# 
# # Dataset
# The dataset is downloaded on scc in the address: "/projectnb2/cs542-bap/classChallenge/data". You can find the python version of this notebook there as well or you could just type "jupyter nbconvert --to script baselineModel_task1.ipynb" and it will output "baselineModel_task1.py". You should be able to run "baselineModel_task1.py" on scc by simply typing "python baselineModel_task1.py"
# 
# Please don't try to change or delete the dataset.
# 
# # Evaluation:
# You will compete with each other over your performance on the dedicated test set. The performance measure is top the 5 error, i.e: if the true class is in one of your top 5 likely predictions, then its error is 0, otherwise its error is 1.  So, your goal is to get an error of 0. This notebook outputs top5 accuracy, so it is 1 - top5 error.
# 
# # Baseline:
# The following code is a baseline which you can use and improve to come up with your model for this task
# 
# # Suggestion
# One simple suggestion would be to use a pretrained model on imagenet and finetune it on this data similar to this [link](https://keras.io/api/applications/)
# Also you should likely train more than 2 epochs.

# ## Import TensorFlow and other libraries

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# ## Explore the dataset

# In[ ]:


import pathlib
data_dir = '/projectnb2/cs542-bap/class_challenge/'
image_dir = os.path.join(data_dir, 'images')
image_dir = pathlib.Path(image_dir)
image_count = len(list(image_dir.glob('*.jpg')))
print("Total number of images = ",image_count)


# ## Here are some images

# In[ ]:


# PIL.Image.open(os.path.join(image_dir, '100.jpg'))


# # Create a dataset

# In[ ]:


train_ds = tf.data.TextLineDataset(os.path.join(data_dir, 'train.txt'))
val_ds = tf.data.TextLineDataset(os.path.join(data_dir, 'val.txt'))
test_ds = tf.data.TextLineDataset(os.path.join(data_dir, 'test.txt'))

with open(os.path.join(data_dir, 'classes.txt'), 'r') as f:
  class_names = [c.strip() for c in f.readlines()]
  
num_classes = len(class_names)


# ## Write a short function that converts a file path to an (img, label) pair:

# In[ ]:


def decode_img(img, crop_size=224):
  img = tf.io.read_file(img)
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # resize the image to the desired size
  return tf.image.resize(img, [crop_size, crop_size])

def get_label(label):
  # find teh matching label
  one_hot = tf.where(tf.equal(label, class_names))
  # Integer encode the label
  return tf.reduce_min(one_hot)

def process_path(file_path):
  # should have two parts
  file_path = tf.strings.split(file_path)
  # second part has the class index
  label = get_label(file_path[1])
  # load the raw data from the file
  img = decode_img(tf.strings.join([data_dir, 'images/', file_path[0], '.jpg']))
  return img, label

def process_path_test(file_path):
  # load the raw data from the file
  img = decode_img(tf.strings.join([data_dir, 'images/', file_path, '.jpg']))
  return img, file_path


# ## Finish setting up data

# In[ ]:


batch_size = 32


# In[ ]:


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path_test, num_parallel_calls=AUTOTUNE)


# In[ ]:


for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())


# ## Data loader hyper-parameters for performance!

# In[ ]:


def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)


# ## Here are some resized images ready to use!

# In[ ]:


image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(class_names[label])
  plt.axis("off")


# ## EfficientB0 transfer learning

# In[ ]:

class EfficientB0(tf.keras.Model):

    def __init__(self):
        super(EfficientB0, self).__init__()
        self.EfficientB0 = keras.applications.EfficientNetB0(
             include_top=False,
             weights='imagenet',
             input_shape=(224, 224, 3), 
             drop_connect_rate=0.4
        )
        for layer in self.EfficientB0.layers[:-3]:
            layer.trainable = False
        self.pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc_1 = layers.Dense(1024)
        self.dropout = layers.Dropout(0.3)
        self.fc_2 = layers.Dense(units=num_classes)

    def call(self, inputs):
        x = self.EfficientB0(inputs)
        # x = self.flatten(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.fc_1(x)
        x = self.dropout(x)
        output = self.fc_2(x)

        return output

# compile the model
model = Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal'),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
    EfficientB0()
])


# ## Training, first 20 epochs, using data augmentation and dropouts, avoid overfitting
# In[ ]:

learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                            patience=2, 
                                                            verbose=1, 
                                                            factor=0.5)

checkpoint_path = "checkpoints/my_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)




model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy',
                       tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=20,
          shuffle=True,
          callbacks=[
              cp_callback,
              learning_rate_reduction
          ])



# ## Define the model again, remove certain regularizations

# In[ ]:


class EfficientB0(tf.keras.Model):

    def __init__(self):
        super(EfficientB0, self).__init__()
        self.EfficientB0 = keras.applications.EfficientNetB0(
             include_top=False,
             weights='imagenet',
             input_shape=(224, 224, 3), 
             drop_connect_rate=0.4
        )
        for layer in self.EfficientB0.layers[:-3]:
            layer.trainable = False
        self.pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc_1 = layers.Dense(1024)
        self.dropout = layers.Dropout(0.3)
        self.fc_2 = layers.Dense(units=num_classes)

    def call(self, inputs):
        x = self.EfficientB0(inputs)
        # x = self.flatten(x)
        x = self.pool(x)
        # x = self.dropout(x)
        x = self.fc_1(x)
        # x = self.dropout(x)
        output = self.fc_2(x)

        return output

# compile the model
model = Sequential([
    #layers.experimental.preprocessing.RandomFlip('horizontal'),
    #layers.experimental.preprocessing.RandomZoom(0.1),
    #layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
    EfficientB0()
])

model.load_weights("checkpoints/my_model.ckpt")


# ## Training, second 20 epochs
# In[ ]:

learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                            patience=2, 
                                                            verbose=1, 
                                                            factor=0.5)

checkpoint_path = "checkpoints/my_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)




model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy',
                       tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=20,
          shuffle=True,
          callbacks=[
              cp_callback,
              learning_rate_reduction
          ])


# # Output submission csv for Kaggle
# 

# In[ ]:


with open('/projectnb2/cs542-bap/frankchz/submission_task1_supervised.csv', 'w') as f:
  f.write('id,predicted\n')
  for image_batch, image_names in test_ds:
    predictions = model.predict(image_batch)
    for image_name, predictions in zip(image_names.numpy(), model.predict(image_batch)):
      inds = np.argpartition(predictions, -5)[-5:]
      line = str(int(image_name)) + ',' + ' '.join([class_names[i] for i in inds])
      f.write(line + '\n')






