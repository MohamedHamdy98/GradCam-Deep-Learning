#!/usr/bin/env python
# coding: utf-8

# In[18]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing import image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from IPython.display import Image, display
from keras.applications.mobilenet import preprocess_input


# In[19]:


def get_img_array(path, size):
    img = plt.imread(path)
    img = img[:, :, :3]
    img = cv2.resize(img, (size, size))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


# In[20]:


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# In[21]:


def save_and_display_gradcam(img_path, model, last_conv_layer_name, alpha, img_size, cam_path="cam.jpg"):
    
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)
    #display(Image(cam_path))
    #Display heatmap
    #plt.matshow(heatmap)
    #plt.show()
    #---------------------------------------------------------
    pred = model.predict(img_array)
    #---------------------------------------------------------
    img_heat = cv2.imread(img_path, 1)
    gray_img = cv2.cvtColor(img_heat, cv2.COLOR_BGR2GRAY)
    heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, 0.7, img_heat, 0.3, 0)
    #---------------------------------------------------------
    upsample = cv2.resize(heatmap, (img_size, img_size))
    #---------------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.tight_layout()
    
    
    plt.subplot(1, 3, 1)
    plt.imshow((img).astype(np.uint0))
    plt.title('Original image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow((img).astype(np.uint0))
    plt.imshow(superimposed_img)
    plt.title('GradCam')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(fin)
    plt.title('HeatMap')
    plt.axis('off')
    
    plt.show()


# In[17]:





# In[ ]:




