import os

import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.utils import losses_utils
import matplotlib.pyplot as plt


def channel_shuffle(x, groups):
    batch_size, height, width, channels = x.shape.as_list()

    channels_per_group = channels // groups

    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, height, width, channels])

    return x

def shuffle_unit(x, in_channels, out_channels, groups):
    bottleneck_channels = out_channels // 4

    x = Conv2D(bottleneck_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = channel_shuffle(x, groups)
    x = Conv2D(bottleneck_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', groups=bottleneck_channels)(x)
    x = BatchNormalization()(x)

    x = Conv2D(out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if in_channels == out_channels:
        x = tf.keras.layers.add([x, x])
    else:
        shortcut = Conv2D(out_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        shortcut = BatchNormalization()(shortcut)
        x = tf.keras.layers.add([x, shortcut])

    x = Activation('relu')(x)

    return x

def shuffle_net_v1(input_shape, num_classes, groups=1, in_channels=None):
    inputs = tf.keras.Input(shape=input_shape)

    # 第一个卷积层
    x = Conv2D(24, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # ShuffleNet单元部分
    x = shuffle_unit(x, in_channels=24, out_channels=240, groups=groups)
    x = shuffle_unit(x, in_channels=240, out_channels=480, groups=groups)
    x = shuffle_unit(x, in_channels=480, out_channels=960, groups=groups)
    x = shuffle_unit(x, in_channels=960, out_channels=1920, groups=groups)
    x = shuffle_unit(x, in_channels=240, out_channels=3840, groups=groups)
    x = shuffle_unit(x, in_channels, out_channels=960, groups=groups)

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)


    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

input_shape = (32, 32, 3)
num_classes = 10

model = shuffle_net_v1(input_shape, num_classes)


model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
Y_test = label_binarize(y_test, classes=[0, 1, 2, 3,4 ,5,6,7,8,9,
                                        ])
n_classes = Y_test.shape[1]  # 有几列，就是几分类！

Y_train = label_binarize(y_train, classes=[0, 1, 2, 3,4 ,5,6,7,8,9,])

dtc = OneVsRestClassifier(DecisionTreeClassifier(criterion="gini", min_samples_leaf=3, max_depth=15))



batch_size = 128

epochs = 200
learning_rate = tf.Variable(0.003, dtype=tf.float32)

decay = 1e-6

optimizer = tf.keras.optimizers.Adam(
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
    # batch_size=batch_size,
    shuffle=True,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[cp_callback]


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
plt.savefig('./Average precision score, macro-averaged over all classes1RES1.jpg')
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
plt.savefig('./Training and Validation accRES1.jpg')
plt.figure()


plt.plot(epochs,loss,marker="o",label="Training loss",linewidth=1)
plt.plot(epochs,val_loss,marker="o",label="Validation loss",linewidth=1)
plt.title("Training and Validation loss ")
plt.legend(loc = "lower left")
plt.savefig('./Training and Validation lossRES1.jpg')
plt.figure()
plt.show()

model.save('shuffilnetv1.h5')
model.save_weights('shuffilnetv1#.h5')
model = tf.keras.models.load_model('shuffilnetv1.h5')

loss,accuracy = model.evaluate(x_test,  y_test, batch_size=batch_size,verbose=1)

print('\ntest loss',loss)
print('accuracy',accuracy)
