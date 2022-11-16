import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
import time
import argparse


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def make_maxpool(input_shape,pool_size=2,stride=1,dtype="float32"):
    model = tf.keras.Sequential([
        tf.keras.layers.MaxPooling2D(input_shape = input_shape, pool_size=(pool_size, pool_size),strides=(stride,stride), padding='same')
    ])

    # optimizer , loss function, metircs 수정할 경우 재작성 필요 
    model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def make_random_dataset(img_size,input_channel,pool_size,batch_size,dtype="float32"):
    input_shape = (batch_size, img_size, img_size, input_channel)
    output_shape = (batch_size, img_size//pool_size, img_size//pool_size, input_channel)
    input_arr = np.random.uniform(0, 255, size=input_shape).astype(dtype)
    output_arr = np.random.uniform(0, 255, size=output_shape).astype(dtype)
    return input_arr, output_arr

def inf_benchmark(device,mpool_layer,img_size,input_channel,pool_size,batch_size=1,train=False,repeat=20):
    print("Using ",device)
    with tf.device("/%s:0" % device):
        if train : 
            input_data, output_data = make_random_dataset(img_size,input_channel,pool_size,batch_size)
            time_callback = TimeHistory()
            mpool_layer.fit(input_data,output_data,epochs=repeat,callbacks=[time_callback])
            print(f'train average time: {np.median(np.array(time_callback.times)) * 1000}ms')

        # 추론
        time_list = []
        for i in range(repeat):
            input_data, output_data = make_random_dataset(img_size,input_channel,pool_size,batch_size)
            start_time = time.time()
            mpool_layer.predict(input_data)
            running_time = time.time() - start_time
            time_list.append(running_time)
        print(f'inference time for {repeat} :',time_list)
        print()
        print(f'inference median time: {np.median(np.array(time_list)) * 1000}ms')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsize',default=224 , type=int)
    parser.add_argument('--input_channel',default=3 , type=int )
    parser.add_argument('--pool_size',default=2 , type=int )
    parser.add_argument('--stride',default=1 , type=int)
    parser.add_argument('--dtype',default='float32' , type=str)
    parser.add_argument('--device',default='CPU' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)


    img_size = parser.parse_args().imgsize
    input_channel = parser.parse_args().input_channel
    pool_size = parser.parse_args().pool_size
    stride = parser.parse_args().stride
    dtype = parser.parse_args().dtype
    device = parser.parse_args().device
    batchsize = parser.parse_args().batchsize

    # make maxpool layer
    input_shape = (img_size,img_size,input_channel)
    mpool_layer = make_maxpool(input_shape,pool_size,stride)
    
    # maxpool layer inference 
    inf_benchmark(device,mpool_layer,img_size,input_channel,pool_size,batchsize)
