import os

import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.utils import losses_utils


def residual_block(x, filters, strides=(1, 1)):
    shortcut = x

    x = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    if strides != (1, 1) or shortcut.shape[-1] != filters:

        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)


    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')

        self.conv2 = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = BatchNormalization()

        self.shortcut = None
        if strides != 1:
            self.shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')
            self.bn_shortcut = BatchNormalization()

        self.relu2 = Activation('relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = inputs
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut)

        x = Add()([x, shortcut])
        x = self.relu2(x)

        return x




def resnet_20():
    inputs = tf.keras.Input(shape=(32, 32, 3))

    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = residual_block(x, filters=16)
    x = residual_block(x, filters=16)
    x = residual_block(x, filters=16)

    x = residual_block(x, filters=32, strides=(2, 2))
    x = residual_block(x, filters=32)
    x = residual_block(x, filters=32)

    x = residual_block(x, filters=64, strides=(2, 2))
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)

    x = GlobalAveragePooling2D()(x)

    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(x_train.shape)
print(x_test.shape)

model =resnet_20()

x_train = x_train.astype('float32')
x_train = x_train / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
x_test = x_test.astype('float32')
x_test = x_test / 255.0
y_test = tf.keras.utils.to_categorical(y_test, 10)

Y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = Y_test.shape[1]
Y_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

dtc = OneVsRestClassifier(DecisionTreeClassifier(criterion="gini", min_samples_leaf=3, max_depth=15))



batch_size = 128

epochs = 200
learning_rate = tf.Variable(0.003, dtype=tf.float32)

decay = 1e-6

optimizer =  tf.keras.optimizers.RMSprop(
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


# 保存
model.save('Res.h5')
model.save_weights('Res#.h5')
model = tf.keras.models.load_model('Res.h5')

loss,accuracy = model.evaluate(x_test,  y_test, batch_size=batch_size,verbose=1)




print('\ntest loss',loss)
print('accuracy',accuracy)
