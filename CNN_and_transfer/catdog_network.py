# -*- coding: utf-8 -*-

'''We have two neural network models. The first has an input layer with 32 filters and ReLu activation. We then have 4 hidden layers with (64,128,256, 512) filters and then an output layer with 2 possibilities and softmax activation to get a probability of either class. Each layer uses the ReLu activation, a 3x3 filter and single unit stride. We also apply normalization and droput to each layer to prevent overfitting. Our dropout for each hidden layer is 30,40,50 and 60 percent respectively. We use the RMSprop optimizer which has shown an ability to be more effective than Adam for certain classification problems. Our loss function is the categorical crossentropy function.''' 

'''For our second model we switch to the LeakyReLu activation function will allows for some negative values in the activation function, preventing the vanishing gradient problem. We also switch to the Adam optimizer and allow for padding for our first two hidden layers and input layer, meaning the outputs will have the same shape as the input. Lastly our filter setup is (16,32,32,64,32).'''

import tensorflow as tf
import tensorflow_hub as hub

# first model

def catdog_net1(height, width, channels, n_class, filter_size=(3,3), lmb=None, regularization=None):
    '''
    

    Parameters
    ----------
    height : (integer)
             The height dimension of our input images.
    width : (integer)
            The width dimension of our input images.
    channels : (integer)
               The number of channels for our image (i.e. RGB channels = 3).
    n_class : (integer)
              The number of class we wish to predict.
    lmb : (float), optional
          Regularization parameter. Considers values > 0. The default is None.
    regularization : (string)
                     Select whether to use L1, L2 or L1L2 regularization if lmb is not None.
                     The default is None.

    Returns
    -------
    model : 
        The neural network model
        

    '''
    
    if lmb is not None and regularization is not None:
        
        if regularization == 'L1':
            
            reg = tf.keras.regularizers.L1(lmb)
        
        elif regularization == 'L2':
            
            reg = tf.keras.regularizers.L2(lmb)
            
        elif regularization == 'L1L2':
        
            reg = tf.keras.regularizers.L1L2(lmb)
            
        else:
            
            raise ValueError("Regularizer must be 'L1' or 'L2'.")
    else:
        
        reg = None
            
    pool_size = (2,2)
    
    filter_size = filter_size
    
    model = tf.keras.models.Sequential([  
        
    # Input layer
    tf.keras.layers.Conv2D(32, filter_size, activation='relu', input_shape=(height, width, channels), kernel_regularizer=reg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    
    #layer 1
    tf.keras.layers.Conv2D(64, filter_size, activation='relu', kernel_regularizer=reg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.Dropout(0.2),
    
    # layer 2    
    tf.keras.layers.Conv2D(128, filter_size, activation='relu', kernel_regularizer=reg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.Dropout(0.3),
    
    # layer 3    
    tf.keras.layers.Conv2D(256, filter_size, activation='relu', kernel_regularizer=reg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=pool_size),
    tf.keras.layers.Dropout(0.4),

    # layer 4 / dense   
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=reg),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    # output    
    tf.keras.layers.Dense(n_class, activation='softmax')
        
    ])
    
    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=1e-3)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=RMSprop, 
                  metrics=['accuracy']
                 )

    return model



# second model


def catdog_net2(height, width, channels, n_class, filter_size=(3,3), lmb=None, regularization=None):
    '''
    

    Parameters
    ----------
    height : (integer)
             The height dimension of our input images.
    width : (integer)
            The width dimension of our input images.
    channels : (integer)
               The number of channels for our image (i.e. RGB channels = 3).
    n_class : (integer)
              The number of class we wish to predict.
    lmb : (float), optional
          Regularization parameter. Considers values > 0. The default is None.
    regularization : (string)
                     Select whether to use L1, L2 or L1L2 regularization if lmb is not None.
                     The default is None.

    Returns
    -------
    model : 
        The neural network model
        

    '''
    
    
    if lmb is not None and regularization is not None:
        
        if regularization == 'L1':
            
            reg = tf.keras.regularizers.L1(lmb)
        
        elif regularization == 'L2':
            
            reg = tf.keras.regularizers.L2(lmb)
            
        elif regularization == 'L1L2':
        
            reg = tf.keras.regularizers.L1L2(lmb)
            
        else:
            
            raise ValueError("Regularizer must be 'L1' or 'L2'.")
    else:
        
        reg = None

    
    # initialize the model
    
    model = tf.keras.models.Sequential()
    
    
    model.add(tf.keras.layers.Conv2D(filters=16, 
                                     kernel_size=filter_size, 
                                     padding="same", 
                                     input_shape=(height,width,channels),
                                     kernel_regularizer=reg))
    
    model.add(tf.keras.layers.LeakyReLU(0.1))
    
    
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=filter_size, padding='same', kernel_regularizer=reg))
    model.add(tf.keras.layers.LeakyReLU(0.1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    
    
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=filter_size, padding='same', kernel_regularizer=reg))
    model.add(tf.keras.layers.LeakyReLU(0.1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    
    
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=filter_size, padding='same', kernel_regularizer=reg))
    model.add(tf.keras.layers.LeakyReLU(0.1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, kernel_regularizer=reg))
    model.add(tf.keras.layers.LeakyReLU(0.1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.6))
    
    model.add(tf.keras.layers.Dense(n_class, activation='softmax'))
        
    adam = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model 


# using transfer learning. We will use MobileNet.

def transfer_model(n_class):
    
    # MobileNet expect images of shape 224x224x3
    IMAGE_RES = 224
    
    # The URL to scrap our model with the last layer stripped out
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
    
    # the transfer network
    feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES,3))
    
    # freeze the trainable parameters to save computation time
    feature_extractor.trainable = False
    
    # add equivalent final layer for our specific problem
    model = tf.keras.Sequential([
        
        # MobileNet layers - 1
        feature_extractor,
        
        # output layer
        tf.keras.layers.Dense(n_class, activation='softmax')
   ])
    
    # compile
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
    
    
    
    
    
    