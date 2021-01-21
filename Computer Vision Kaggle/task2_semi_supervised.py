#!/usr/bin/env python
# coding: utf-8

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

# ## Create a dataset

# In[ ]:

data_dir = '/projectnb2/cs542-bap/class_challenge/'

train_samps = np.loadtxt(os.path.join(data_dir, 'train_held_out_labeled.txt'), dtype='str', delimiter=" ")
val_samps = np.loadtxt(os.path.join(data_dir, 'val_held_out.txt'), dtype='str', delimiter=" ")

train_len = len(train_samps)
val_len = len(val_samps)

samples = np.concatenate((train_samps, val_samps))

unlabeled_samps = np.loadtxt(os.path.join(data_dir, 'train_held_out.txt'), dtype='str')
unlabeled_len = len(unlabeled_samps)

test_ds = tf.data.TextLineDataset(os.path.join(data_dir, 'test_held_out.txt'))

with open(os.path.join(data_dir, 'classes_held_out.txt'), 'r') as f:
    class_names = [c.strip() for c in f.readlines()]

num_classes = len(class_names)


# ## Write a short function that converts a file path to an (img, label) pair

# In[ ]:

def decode_img(img, test=False, crop_size=224):
    img = tf.io.read_file(img)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)

    return tf.image.resize(img, [crop_size, crop_size])
  
def get_label(label):
    # find teh matching label
    one_hot = tf.where(tf.equal(label, class_names))
    # Integer encode the label
    return tf.reduce_min(one_hot)

def process_path(path, label):
    # should have two parts
    # file_path = tf.strings.split(file_path)
    # second part has the class index
    label = get_label(label)
   # load the raw data from the file
    img = decode_img(tf.strings.join([data_dir, 'images/', path, '.jpg']))
    return img, label

def process_path_test(file_path):
    # load the raw data from the file
    img = decode_img(tf.strings.join([data_dir, 'images/', file_path, '.jpg']))
    return img, file_path


# ## Finish setting up data

# In[ ]:

batch_size = 25

AUTOTUNE = tf.data.experimental.AUTOTUNE
test_ds = test_ds.map(process_path_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(1)

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def shuffle_train_val(train_perc = 0.2):
    """
    This function returns shuffled train data and val data
    The default is we take 20% samples as training data
    """
    # define the train length
    train_len = int(train_perc*len(samples))
    
    # idexing train set and val set by random choice
    train_idx = np.random.choice(range(len(samples)), train_len, replace=True)
    val_idx = [idx for idx in range(len(samples)) if idx not in train_idx]
    
    # get train_ds and val_ds based on indexes
    train_ds = tf.data.Dataset.from_tensor_slices((samples[train_idx, 0], samples[train_idx, 1]))
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds)
    val_ds = tf.data.Dataset.from_tensor_slices((samples[val_idx, 0], samples[val_idx, 1]))
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds)

    return train_ds, val_ds
    

# ## Models

# ## ResNet50

# In[ ]:

class ResNet50(tf.keras.Model):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.ResNet50 = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        
        # unfreeze the last two layers
        for layer in self.ResNet50.layers[:-2]:
            layer.trainable = False
        
        # define layers
        self.pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc_1 = layers.Dense(1024)
        self.fc_2 = layers.Dense(units=num_classes)

    def call(self, inputs):
        x = keras.applications.resnet.preprocess_input(inputs)
        x = self.ResNet50(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        output = self.fc_2(x)

        return output

# data augmentation
model = Sequential([
    layers.experimental.preprocessing.RandomFlip(
        mode='horizontal'),
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
    ResNet50()
])

# compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
    
    
# ## EfficientB0

# In[ ]:

class EfficientB0(tf.keras.Model):

    def __init__(self):
        super(EfficientB0, self).__init__()
        self.EfficientB0 = keras.applications.EfficientNetB0(
             include_top=False,
             weights='imagenet',
             input_shape=(224, 224, 3), 
             # add stronger reguarliztions
             drop_connect_rate=0.4
        )
        
        # unfreeze top 20 layers
        for layer in self.EfficientB0.layers[:-20]:
            layer.trainable = False
            
        # define layers
        self.pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc_1 = layers.Dense(1024)
        self.dropout = layers.Dropout(0.3)
        self.fc_2 = layers.Dense(units=num_classes)

    def call(self, inputs):
        x = self.EfficientB0(inputs)
      # x = self.flatten(x)
        x = self.pool(x)
        x = self.fc_1(x)
        x = self.dropout(x)
        output = self.fc_2(x)

        return output

# image augmentation
model = Sequential([
    layers.experimental.preprocessing.RandomFlip(
       mode='horizontal'),
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
    EfficientB0()
])

# compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
    



# ## Add lables to unlabeled data

# In[ ]:
 

def Add_labels(unlabeled_samps, model, unlabeled_batch):
    """
    unlabeled_samps: the indexes of unlabeled samples
    unlabeled_batch: the number of unlabelled data to be predicted
    return: a bunch of predcitions of unlabeled samples
    """
    # decode the unlabelled images
    unlabeled_ds = tf.data.Dataset.from_tensor_slices(unlabeled_samps)
    unlabeled_ds = unlabeled_ds.map(process_path_test, num_parallel_calls=AUTOTUNE)
    unlabeled_ds = unlabeled_ds.batch(1)
    
    # initialize prediction tracker
    predictions = None
    # initialize indexes tracker
    inds = []
    for image, image_name in unlabeled_ds:
        preds = model.predict(image)
        ind = np.argmax(preds)
        cls = class_names[ind]
        pred = (str(int(image_name)), cls)
        
        # keep tracking predictions
        if predictions is None:
            predictions = np.array(pred)
        else:
            predictions = np.vstack((predictions, pred))
            
        # keep tracking the indexes
        inds.append(preds[0, ind])
        
    # output top n predictions, n = max_unlabeled
    inds = np.argpartition(inds, -unlabeled_batch)[-unlabeled_batch:]
    predictions = predictions[inds]
    return predictions


# ## Semi-supervised learning

# In[ ]:

def train_(num_iters = 10, threshold = 0.01, train_perc = 0.1):
    """
    num_iters: the number of iterations, for each iteration, 
               the model trains both labeled and unlabeled samples
    threshold: the threshold of model improvements, if improvements less than the threshold, 
               start training unlabeled examples.
    train_perc: the percentage of unlabeled data being trained for reaching the threshold
    return: a list of trained models
    """
    
    global model
    
    model_list = [None] * num_iters
    
    # the main training loop
    for i in range(num_iters):
  
        model = model
        train_ds, val_ds = shuffle_train_val()
        samps = samples
        unlabeled = unlabeled_samps

        print(f"Iteration {i+1}")
        unlabeled_batch = int(train_perc * unlabeled_len)
    
        # finish training this iteration until all unlabeled data are used
        while len(unlabeled) > 0:
            hist = model.fit(train_ds, validation_data=val_ds, epochs=2, shuffle=True)
            improvement = hist.history['val_accuracy'][-1] - hist.history['val_accuracy'][-2]
            
            # as long as the model stop moving forward, start training unlabeled samples
            if improvement <= threshold:
                preds = Add_labels(unlabeled, model, min(len(unlabeled), unlabeled_batch))
                pred_ds = tf.data.Dataset.from_tensor_slices((preds[:,0], preds[:,1]))
                pred_ds = pred_ds.map(process_path, num_parallel_calls=AUTOTUNE)
                pred_ds = configure_for_performance(pred_ds)
      
                # keep updating the training set and the unlabeled set
                train_ds.concatenate(pred_ds)
                unlabeled = [j for j in unlabeled if j not in preds[:,0]]
                print(f"number of unlabeled samples remained: {len(unlabeled)}")
           
        # train all labeled and unlabeled data
        print(f"fine tuning the model (iteration {i+1})")
        model.fit(train_ds,validation_data=val_ds,epochs=20,shuffle=True)
        
        # keep track of the trained models
        model_list[i] = model

    
    
    
# ## Train the model

# In[ ]:

train_()



# ## Save the output for Kaggle

# In[ ]:

with open('submission_semisupervised.csv', 'w') as f:
  f.write('id,predicted\n')
  for image_batch, image_names in test_ds:
    predictions = model.predict(image_batch)
    for image_name, predictions in zip(image_names.numpy(), model.predict(image_batch)):
      inds = np.argmax(predictions)
      line = str(int(image_name)) + ',' + class_names[inds]
      f.write(line + '\n')




