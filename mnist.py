import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pandas as pd

from tensorflow import keras
from tensorflow._api.v2 import data
from tensorflow.keras import optimizers
from tensorflow.python.ops.gen_batch_ops import batch
import time
import itertools
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config = config)


def load_data(img_path,batch_size):
    """
    输入：路径
    输出：dataset格式的train data/label 和 test data/label
    加载数据集，1：9划分训练集和测试集，返回dataset格式
    """
    mnist = pd.read_csv(img_path)
    
    mnist_label = mnist['label']
    print('标签：',mnist_label.shape)
    mnist = mnist.iloc[:,1:]
    print('数据：',mnist.shape)
    
    x_train,x_test,y_train,y_test = train_test_split(mnist,mnist_label,test_size=0.2,random_state=0)
    
    train_dataset = convert_to_dataset(x_train,y_train,10)
    test_dataset  = convert_to_dataset(x_test,y_test,10) 
    
    train_dataset = train_dataset.shuffle(1000, reshuffle_each_iteration=True)  #这数字没影响把？
    train_dataset = train_dataset.repeat(5)
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    #print('------------',test_dataset.element_spec)
    #(TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None, 10), dtype=tf.float32, name=None))
    
    return train_dataset,test_dataset

def convert_to_dataset(input_data,input_label,class_num=10):
    """
    输入：数据，数据标签，分类个数
    输出：dataset格式
    """
    input_data = tf.convert_to_tensor(input_data,dtype=float)
    input_label = tf.convert_to_tensor(input_label)
    input_label = tf.one_hot(  input_label,class_num)
    
    input_data = input_data[...,tf.newaxis]
    input_data = tf.reshape(input_data,[input_data.shape[0],28,28,1])
    
    image_data = (input_data,input_label)
    dataset = tf.data.Dataset.from_tensor_slices(image_data)
    return dataset

class Net(tf.keras.Model):
    """
    构建网络
    """
    def __init__(self):
        super(Net,self).__init__()
        fliters = 64 #卷积核个数

        #必须写不同的bn层，不然会报错，因为shape变了

        #卷积1（fliter，3x3，1，same） ，bn  -> 
        self.conv1 = tf.keras.layers.Conv2D(fliters,3,1,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(fliters,3,1,padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.maxpool1 = tf.keras.layers.MaxPool2D(2)

        #
        self.conv3 = tf.keras.layers.Conv2D(fliters*2,4,1,padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        self.conv4 = tf.keras.layers.Conv2D(fliters*2,4,1,padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.maxpool2 = tf.keras.layers.MaxPool2D(3)

        #
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256,activation='relu')
        self.dense2 = tf.keras.layers.Dense(10,activation='softmax')

    def call(self,input,training=True):
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.relu(x)
        x = self.maxpool2(x)
        
        # 打平
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.nn.dropout(x,0.3)
        x = self.dense2(x)
    
        return x


def k_test():
    """
    k折检验，未做
    """
    pass

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    来源：https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6#3.-CNN
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



if __name__ == '__main__':

    learning_rate = 0.0001 #adam
    epochs = 10 #
    batch_size = 96

    mnist_data,mnist_val_data= load_data(r"D:\USE\kaggle\mnist\train.csv",batch_size = batch_size)
    
    Mynet = Net()   #创建网络
    Mynet.build(input_shape=[None,28,28,1])
    Mynet.summary()
    optimizers = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    acc = tf.keras.metrics.CategoricalAccuracy()   #准确率 accuracy是具体类别，categoricalaccur是one hot 编码
    val_acc = tf.keras.metrics.CategoricalAccuracy()
    
    begin = time.time()
    train_loss , val_loss = 0 , 0
    
    #训练
    for epoch in range(epochs):
        print('training...')
        for _,(data,label) in enumerate(mnist_data):
            with tf.GradientTape() as tape: #更新
                y_pred = Mynet(data)
                loss1 = tf.keras.losses.categorical_crossentropy(y_true=label , y_pred=y_pred )
                train_loss = tf.reduce_sum(loss1)
                acc.update_state(label,y_pred)
            grads = tape.gradient(train_loss,Mynet.trainable_variables)
            optimizers.apply_gradients(zip(grads,Mynet.trainable_variables))

        #测试集
        if epoch % 2 == 0:
            #求val loss
            print('testing...')
            for _,(test_data,test_label) in enumerate(mnist_val_data):
                val_pred = Mynet(test_data)
                loss2 = tf.keras.losses.categorical_crossentropy(y_true=test_label , y_pred=val_pred )
                val_loss = tf.reduce_sum(loss2)
                val_acc.update_state(test_label,val_pred)

            #
            

        print('epoch:',epoch)
        print('train loss:',float(train_loss),'test loss:',float(val_loss),'acc: ',acc.result().numpy(),'val_acc:',val_acc.result().numpy())
        acc.reset_states()
        val_acc.reset_states()
        end = time.time()
        print('time:%.5f s ' %(end-begin),'\n')
        begin = time.time()

    #画图
    Y_pred_classes=[]
    Y_true=[]
    y_martix = []
    y_martix_true = []
    print('draw confusion martix...')
    for _,(test_data,test_label) in enumerate(mnist_val_data):
        val_pred = Mynet(test_data)
        # Predict the values from the validation dataset
        Y_pred = Mynet.predict(test_data)
        # Convert predictions classes to one hot vectors 
        Y_pred_classes = np.argmax(Y_pred,axis = 1) 
        y_martix.extend(Y_pred_classes)
        # Convert validation observations to one hot vectors
        Y_true = np.argmax(test_label,axis = 1) 
        y_martix_true.extend(Y_true)
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_martix_true[1:], y_martix[1:]) 
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes = range(10)) 
    print('end...')





