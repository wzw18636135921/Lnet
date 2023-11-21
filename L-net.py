import tensorflow as tf
from tensorflow import keras

from keras.layers import ELU
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from sklearn.tree import DecisionTreeClassifier
from keras.utils import losses_utils
import math
import os

def depthwise_conv_block(input_tensor, point_filters, alpha, depth_multiplier, strides=(1, 1)):
    point_filters = int(point_filters * alpha)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                               strides=strides,
                               padding='same',
                               depth_multiplier=depth_multiplier,
                               use_bias=False)(input_tensor)

    x = tf.keras.layers.BatchNormalization()(x)

    x = ELU()(x)

    x = tf.keras.layers.Conv2D(point_filters, kernel_size=(1, 1),
                      padding='same',
                      strides=(1, 1),
                      use_bias=False)(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = ELU()(x)

    return x

def conv_block(input_tensor, filters, alpha, kernel_size=(3, 3), strides=(1, 1)):
    filters = int(filters * alpha)
    x = tf.keras.layers.Conv2D(filters, kernel_size,
                      strides=strides,
                      padding='same',
                      use_bias=False)(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = ELU()(x)
    return x
def bypass_block(
        inputs,
        filters,
        kernel_size=(1, 1),
        strides=(1, 1)
):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(
        inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def eca_block(
        inputs,
        b=1,
        gama=2):

    in_channel = inputs.shape[-1]

    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))

    if kernel_size % 2:
        kernel_size = kernel_size

    else:
        kernel_size = kernel_size + 1

    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    x = tf.keras.layers.Reshape(target_shape=(in_channel, 1))(x)

    x = tf.keras.layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)

    x = tf.nn.softmax(x)

    x = tf.keras.layers.Reshape((1, 1, in_channel))(x)

    outputs = tf.keras.layers.multiply([inputs, x])

    return outputs

def lnet(
        inputs,
        classes,
        alpha,
        depth_multiplier

):

    y = bypass_block(inputs, 64, strides=(2, 2))
    x = conv_block(inputs, 32, alpha, strides=(2, 2))

    # [112,112,32]==>[112,112,64]
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier)
    x = eca_block(x)
    x = tf.math.add(x, y)

    # [112,112,64]==>[56,56,128]
    y = bypass_block(x, 128, strides=(2, 2))
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2))
    x = tf.keras.layers.ReLU(6.0)(x)
    # [56,56,128]==>[56,56,128]
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier)
    # [56,56,128]==>[28,28,256]
    x = eca_block(x)
    x = tf.math.add(x, y)


    y = bypass_block(x, 256, strides=(2, 2))
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2))
    # [28,28,256]==>[28,28,256]
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier)
    x = eca_block(x)
    x = tf.math.add(x, y)

    # [28,28,256]==>[14,14,512]
    y = bypass_block(x, 512, strides=(2, 2))
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2))
    # [14,14,512]==>[14,14,512]
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier)
    x = eca_block(x)
    x = tf.math.add(x, y)
    # [14,14,512]==>[7,7,1024]
    y = bypass_block(x, 1024, strides=(2, 2))
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2))
    # [7,7,1024]==>[7,7,1024]
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier)
    x = eca_block(x)
    x = tf.math.add(x, y)
    shape = (1, 1, int(1024 * alpha))
    x = tf.keras.layers.Reshape(target_shape=shape)(x)

    x = tf.keras.layers.Dropout(rate=1e-6)(x)

    x = tf.keras.layers.Conv2D(classes, kernel_size=(1, 1), padding='same')(x)

    x = tf.keras.layers.Activation('softmax')(x)

    pred = tf.keras.layers.Reshape(target_shape=(classes,))(x)

    return pred

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print(x_train.shape)
print(x_test.shape)

inputs = tf.keras.Input(shape=(32,32,3))
model = tf.keras.Model(inputs=inputs, outputs=lnet(inputs,10,1,1))

x_train = x_train.astype('float32')
x_train = x_train / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
x_test = x_test.astype('float32')
x_test = x_test / 255.0
y_test = tf.keras.utils.to_categorical(y_test, 10)

Y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = Y_test.shape[1]  # 有几列，就是几分类！
Y_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

dtc = OneVsRestClassifier(DecisionTreeClassifier(criterion="gini", min_samples_leaf=3, max_depth=15))



batch_size = 128
epochs = 400

learning_rate = tf.Variable(0.003, dtype=tf.float32)
decay = 1e-6

optimizer =  keras.optimizers.RMSprop(
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
checkpoint_save_path = './临时文件/lent.ckpt'

if os.path.exists(checkpoint_save_path + '.index'):
    print('-----------load the model--------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True,
                                              save_best_only=True)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='categorical_accuracy',
                                                          factor=0.5,
                                                          patience=5,
                                                          min_lr=1e-6,
                                                          verbose=1)

data_generate = ImageDataGenerator( rescale=1,#归至0～1
                                     rotation_range=0,#随机0度旋转
                                     width_shift_range=0,#宽度偏移
                                     height_shift_range=0,#高度偏移
                                     horizontal_flip=True,#水平翻转
                                     zoom_range=1,
                                     validation_split=0.3)


import time
start_time = time.time()
history = model.fit(
    data_generate.flow(x_train, y_train, batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    shuffle=True,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[cp_callback,reduce_lr_callback]


)
end_time = time.time()
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

model.save('mobilenet_v1_cifar.h5')
model.save_weights('mobilenet_v1_cifar.h5#')
model = tf.keras.models.load_model('mobilenet_v1_cifar.h5')


loss,accuracy = model.evaluate(x_test,  y_test, batch_size=batch_size,verbose=1)




print('\ntest loss',loss)
print('accuracy',accuracy)
