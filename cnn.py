from keras.datasets import mnist
datasets = mnist.load_data(mnist.db)
train , test = datasets
xtrain , xtrain = train
xtest , ytest = test
img_rows = xtrain[0].shape[0]
img_cols = xtrain[1].shape[0]
xtrain = xtrain.reshape(xtrain.shape[0], img_rows, img_cols, 1)
xtest = xtest.reshape(xtest.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain /= 255
xtest /= 255
ytrain = np_utils.to_categorical(ytrain)
ytest = np_utils.to_categorical(ytest)
num_classes = ytest.shape[1]
num_pixels = xtrain.shape[1] * xtrain.shape[2]


# create model
model = Sequential()
model.add(Conv2D(20, (5, 5),
                 padding = "same",
                 input_shape = input_shape))

model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# Fully connected layers (w/ RELU)
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# Softmax (for classification)
model.add(Dense(num_classes))
model.add(Activation("softmax"))

model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])


batch_size = 128
epochs = 10

model.fit(xtrain, ytrain,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(xtest, ytest),
          shuffle=True)
