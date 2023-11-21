import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import losses_utils
import math
import os

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0:2], device_type='GPU')


def AlexNet8(
        inputs,
        classes,
):

    x = tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, name='Conv2D_1')(inputs)
    x = tf.keras.layers.Activation('relu', name='Activation_1')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_1')(x)  # 使用 BatchNormalization 代替LRN
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, name='MaxPool2D_1')(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', name='Conv2D_2')(x)
    x = tf.keras.layers.Activation('relu', name='Activation_2')(x)
    x = tf.keras.layers.BatchNormalization(name='BN_2')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name='MaxPool2D_2')(x)


    x = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same', name='Conv2D_3')(x)
    x = tf.keras.layers.Activation('relu', name='Activation_3')(x)

    x = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same', name='Conv2D_4')(x)
    x = tf.keras.layers.Activation('relu', name='Activation_4')(x)


    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name='Conv2D_5')(x)
    x = tf.keras.layers.Activation('relu', name='Activation_5')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=2, name='MaxPool2D_5')(x)


    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=4096, activation='relu', name='Dense_6')(x)
    x = tf.keras.layers.Dropout(0.3, name='Dropout_6')(x)

    x = tf.keras.layers.Dense(units=4096, activation='relu', name='Dense_7')(x)
    x = tf.keras.layers.Dropout(0.3, name='Dropout_7')(x)

    outputs = tf.keras.layers.Dense(classes, activation='softmax', name='Output_8')(x)

    return outputs





(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape)
print(x_test.shape)

inputs = tf.keras.Input(shape=(32,32,3))
model = tf.keras.Model(inputs=inputs, outputs=AlexNet8(inputs,10))

x_train = x_train.astype('float32')
x_train = x_train / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
x_test = x_test.astype('float32')
x_test = x_test / 255.0
y_test = tf.keras.utils.to_categorical(y_test, 10)

batch_size = 64
epochs = 400
learning_rate = tf.Variable(0.3, dtype=tf.float32)
decay = 1e-6

optimizer = tf.optimizers.Adam(
    learning_rate = learning_rate,
    decay = decay
)
loss = tf.keras.losses.CategoricalCrossentropy(
                                               from_logits=False,
                                               label_smoothing=0.3,
                                               reduction=losses_utils.ReductionV2.AUTO)

# 训练
model.compile(
    loss='categorical_crossentropy',
    optimizer= 'adam',
    metrics=['categorical_accuracy']
)
checkpoint_save_path = './临时文件/mobnet.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('-----------load the model--------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True,
                                              save_best_only=True)

data_generate = ImageDataGenerator( rescale=1,
                                     rotation_range=0,
                                     width_shift_range=0,
                                     height_shift_range=0,
                                     horizontal_flip=True,
                                     zoom_range=1,
                                     validation_split=0.3)



history = model.fit(
    data_generate.flow(x_train, y_train, batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    # batch_size=batch_size,
    shuffle=True,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[cp_callback]

)

print(model.summary())

acc = history.history["categorical_accuracy"]
val_acc = history.history["val_categorical_accuracy"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]
epochs = range(len(acc))
plt.plot(epochs,acc,'b',label="Training acc")
plt.plot(epochs,val_acc,'b',label="Validation acc")
plt.title("Training and Validation acc ")
plt.figure()

plt.plot(epochs,loss,marker="o",label="Training loss",linewidth=1)
plt.plot(epochs,val_loss,marker="o",label="Validation loss",linewidth=1)
plt.title("Training and Validation loss ")
plt.legend()

plt.show()

model.save('VGG_cifar.h5')
model.save_weights('VGG_cifar#.h5')

model = tf.keras.models.load_model('VGG_cifar.h5')


loss,accuracy = model.evaluate(x_test,  y_test, batch_size=batch_size,verbose=1)
print('\ntest loss',loss)
print('accuracy',accuracy)
