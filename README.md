# Natural Scene Classification

This project uses Intel Scene Classification data from Kaggle. The dataset includes training, testing and prediction data. A deep learning algorithm : CNN(convolutional neural network) is used in this project. 



## About Dataset
About Dataset
Context
This is image data of Natural Scenes around the world.

Content
This Data contains around 25k images of size 150x150 distributed under 6 categories.
{'buildings' -> 0,
'forest' -> 1,
'glacier' -> 2,
'mountain' -> 3,
'sea' -> 4,
'street' -> 5 }

The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction.
This data was initially published on https://datahack.analyticsvidhya.com by Intel to host a Image classification Challenge.


## What is a convolutional neural network (CNN)?
A convolutional neural network (CNN or convnet) is a subset of machine learning. It is one of the various types of artificial neural networks which are used for different applications and data types. A CNN is a kind of network architecture for deep learning algorithms and is specifically used for image recognition and tasks that involve the processing of pixel data.

## Installation

To run this project install these python libraries

```bash
pip install numpy 
pip install pandas
pip intsall matplotlib.pyplot
pip intsall matplotlib.image
pip install seaborn
pip install torch
pip install os
pip install tensorflow
```
    
## Running Tests

There are different tests attached that were done with different parameters. For example: by changing epoch number between 3 and 50 and by chaning the size and number of kernel.

```bash
tf.keras.layers.Conv2D(filters=24, kernel_size=(5,5), activation='relu', padding='same')

hist = mod.fit(train_ds, epochs=50, batch_size=batch, verbose=1, shuffle=shuff, validation_data=val_ds, callbacks=[earlystop, lr])
   
 
```


## Results
Using different size and number of kernels and different number of epoch gave different accuracy.
```bash
using epoch= 50, filters=24, kernel_size=(5,5) gave accuracy score of 17%
```
```bash
using epoch =50, filters=16, kernel_size=(3,3) gave accuracy score of 75%
```
```bash
using epoch =3, filters=16, kernel_size=(3,3) gave accuracy score of 57%
```
## To run the trained model
```bash
import os
import tensorflow as tf
tf.random.set_seed(0)
loaded_model = tf.keras.models.load_model("D:/Downloads/CNN/your_model1.keras")
```
## Graph
![image](https://github.com/swerajaved7/scene_classification/assets/153985092/555607b0-a53c-4488-b43b-e3b02e6ffb83)
