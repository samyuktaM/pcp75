Loss Measure of the Model
==============================

In this activity, you will learn to plot the graph of epoch and loss.


<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10583080/image__20_.png" width = "480" height = "220">


Follow the given steps to complete this activity:


1. Plot the graph


* Open the file main.py.


* Change the categories to `numpy` array of `int64`.


    `categories = np.array(categories,dtype=np.int64)`


* Change images to numpy array.


    `images = np.array(images)`


* Split the images and categories using `train_test_split()`.


    `training_images, testing_images, training_categories, testing_categories = train_test_split(images, categories)`


* Create a `sequential` model.


    `model = Sequential()`


* Add First layer of the model.


    `model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200,200,3)))`


    `model.add(MaxPool2D(pool_size=3, strides=2))`


* Add the second layer of the model.


    `model.add(Conv2D(128, kernel_size=3, activation='relu'))`


    `model.add(MaxPool2D(pool_size=3, strides=2))`


* Add the third layer of the model.


    `model.add(Conv2D(256, kernel_size=3, activation='relu'))`


    `model.add(MaxPool2D(pool_size=3, strides=2))`


* Add the fourth layer of the model.


    `model.add(Conv2D(512, kernel_size=3, activation='relu'))`


    `model.add(MaxPool2D(pool_size=3, strides=2))`


* Flatten the model using`flatten()` .


    `model.add(Flatten())`


* Remove unwanted layers calling `Dropout()`  and passing `0.2` as parameter.


    `model.add(Dropout(0.2))`


* Add dense layer to model with 512 size and relu activation.


    `model.add(Dense(512, activation='relu'))`


* Add dense layer to model with size `1`, `activation='linear` and `name = 'age'`.


    `model.add(Dense(1, activation='linear', name='categories'))`


* Compile the model.

    `model.compile(optimizer='adam', loss='mse', metrics=['mae'])`

* Print model summary.

    `print(model.summary())`


* Train  the model using `fit()`.

    `history = model.fit(training_images, training_categories,
                    validation_data=(testing_images, testing_categories), epochs=10)`


* Save the model.

    `model.save('model_10epochs.h5')`


* Plot the training and validation loss at each epoch.

    `loss = history.history['loss']`

    `val_loss = history.history['val_loss']`

    `epochs = range(1, len(loss) + 1)`

    `plt.plot(epochs, loss, 'y', label='Training loss')`

    `plt.plot(epochs, val_loss, 'r', label='Validation loss')`

    `plt.title('Training and validation loss')`

    `plt.xlabel('Epochs')`

    `plt.ylabel('Loss')`

    `plt.legend()`

    `plt.show()`


* Save and run the code to check the output.
