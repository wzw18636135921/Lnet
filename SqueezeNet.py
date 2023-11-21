
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import keras
from tensorflow.python.keras import layers
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from keras.preprocessing.image import ImageDataGenerator
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.utils import losses_utils
import os


tf.test.gpu_device_name()
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0:2], device_type='GPU')


def fire_module(x, squeeze_filters, expand_filters):
    squeeze = tf.keras.layers.Conv2D(filters=squeeze_filters, kernel_size=(1, 1), activation='relu')(x)
    expand_1x1 = tf.keras.layers.Conv2D(filters=expand_filters, kernel_size=(1, 1), activation='relu')(squeeze)
    expand_3x3 = tf.keras.layers.Conv2D(filters=expand_filters, kernel_size=(3, 3), padding='same', activation='relu')(
        squeeze)
    output = tf.keras.layers.concatenate([expand_1x1, expand_3x3], axis=-1)
    return output


def SqueezeNet(input_shape=(32, 32, 3), num_classes=10):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(
        input_layer)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = fire_module(x, squeeze_filters=64, expand_filters=256)
    x = fire_module(x, squeeze_filters=64, expand_filters=256)

    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Softmax()(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)

    return model



(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(x_train.shape)
print(x_test.shape)


inputs = tf.keras.Input(shape=(32,32,3))

model = SqueezeNet(input_shape=(32, 32, 3), num_classes=10)


x_train = x_train.astype('float32')
x_train = x_train / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
x_test = x_test.astype('float32')
x_test = x_test / 255.0
y_test = tf.keras.utils.to_categorical(y_test, 10)

Y_test = label_binarize(y_test, classes=[0, 1, 2, 3,4 ,5,6,7,8,9])
n_classes = Y_test.shape[1]  # 有几列，就是几分类！
Y_train = label_binarize(y_train, classes=[0, 1, 2, 3,4 ,5,6,7,8,9])

dtc = OneVsRestClassifier(DecisionTreeClassifier(criterion="gini", min_samples_leaf=3, max_depth=15))



batch_size = 64
epochs = 300
learning_rate = tf.Variable(0.0003, dtype=tf.float32)
decay = 1e-6

optimizer = tf.optimizers.RMSprop(
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
    optimizer= optimizer,
    metrics=['categorical_accuracy']
)
checkpoint_save_path = './临时文件/mobnet.ckpt'
# 生成ckpt的同时会生成索引文件
if os.path.exists(checkpoint_save_path + '.index'):
    print('-----------load the model--------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True,
                                              save_best_only=True)



reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=10,
                                              verbose=1,
                                              factor=0.5,
                                              min_lr=1e-6)




data_generate = ImageDataGenerator( rescale=1,#归至0～1
                                     rotation_range=0,#随机0度旋转
                                     width_shift_range=0,#宽度偏移
                                     height_shift_range=0,#高度偏移
                                     horizontal_flip=True,#水平翻转
                                     zoom_range=1,
                                     validation_split=0.3)



history = model.fit(
    data_generate.flow(x_train, y_train, batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    shuffle=True,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[cp_callback,reduce],
)

y_score = model.predict(x_test,batch_size=batch_size)
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])

    average_precision[i] = average_precision_score(Y_test[:, i],
                                                   y_score[:, i])


precision["macro"], recall["macro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                y_score.ravel())


average_precision["macro"] = average_precision_score(Y_test, y_score,
                                                     average="macro")
print(model.summary())
print('Average precision score, macro-averaged over all classes: {0:0.2f}'.format(average_precision["macro"]))
plt.step(recall['macro'], precision['macro'], where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.plot(recall["macro"],precision["macro"],label="micro_average P_R(area={0:0.2f})".format(average_precision["macro"]))
for i in range(n_classes):
    plt.plot(recall[i],precision[i],label="P_R curve of class{0}(area={1:0.2f})".format(i,average_precision[i]))
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, macro-averaged over all classes: AP={0:0.3f}'.format(average_precision["macro"]))
plt.legend(loc = "lower left")
plt.figure()

acc = history.history["categorical_accuracy"]
val_acc = history.history["val_categorical_accuracy"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]
epochs = range(len(acc))
plt.plot(epochs,acc,marker="o",label="Training acc",linewidth=1)
plt.plot(epochs,val_acc,marker="o",label="Validation acc",linewidth=1)
plt.title("Training and Validation acc ")
plt.legend(loc = "lower left")
plt.figure()

plt.plot(epochs,loss,marker="o",label="Training loss",linewidth=1)
plt.plot(epochs,val_loss,marker="o",label="Validation loss",linewidth=1)
plt.title("Training and Validation loss ")
plt.legend(loc = "lower left")
plt.figure()
plt.show()

model.save('SqueezeNet.h5')
model.save_weights('SqueezeNet#.h5')
model = tf.keras.models.load_model('SqueezeNet.h5')
loss,accuracy = model.evaluate(x_test,  y_test, batch_size=batch_size,verbose=1)

print('\ntest loss',loss)
print('accuracy',accuracy)







