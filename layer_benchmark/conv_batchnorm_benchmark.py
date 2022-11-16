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

def make_conv(input_size, conv_size,kernel_size,dtype="float32"):

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_size, dtype=dtype, name='input'),
        tf.keras.layers.Conv2D(conv_size,(kernel_size,kernel_size), activation='relu',padding='same'),
        tf.keras.layers.BatchNormalization()
    ])

    # optimizer , loss function, metircs 수정할 경우 재작성 필요 
    model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def make_random_dataset(img_size,input_channel,output_channel,batch_size,dtype="float32"):
    input_shape = (batch_size, img_size, img_size, input_channel)
    output_shape = (batch_size, img_size, img_size, output_channel)
    input_arr = np.random.uniform(0, 255, size=input_shape).astype(dtype)
    output_arr = np.random.uniform(0, 255, size=output_shape).astype(dtype)
    return input_arr, output_arr

def inf_benchmark(device,conv_layer,img_size,input_channel,output_channel,batch_size=1,train=True,repeat=20):

    print("Using ",device)
    with tf.device("/%s:0" % device):
        if train : 
            time_callback = TimeHistory()
            input_data, output_data = make_random_dataset(img_size,input_channel,output_channel,batch_size)
            conv_layer.fit(input_data,output_data,epochs=repeat,callbacks=[time_callback])
            print(f'train average time: {np.median(np.array(time_callback.times)) * 1000}ms')

        # 추론
        time_list = []
        for i in range(repeat):
            input_data, output_data = make_random_dataset(img_size,input_channel,output_channel,batch_size)
            start_time = time.time()
            conv_layer.predict(input_data)
            running_time = time.time() - start_time
            time_list.append(running_time)
        print(f'inference time for {repeat} :',time_list)
        print()
        print(f'inference median time: {np.median(np.array(time_list)) * 1000}ms')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsize',default=224 , type=int )
    parser.add_argument('--input_channel',default=3 , type=int )
    parser.add_argument('--output_channel',default=64 , type=int )
    parser.add_argument('--kernel',default=3 , type=int)
    parser.add_argument('--dtype',default='float32' , type=str)
    parser.add_argument('--device',default='CPU' , type=str)
    parser.add_argument('--batchsize',default=1 , type=int)


    img_size = parser.parse_args().imgsize
    input_channel = parser.parse_args().input_channel
    output_channel = parser.parse_args().output_channel
    kernel_size = parser.parse_args().kernel
    dtype = parser.parse_args().dtype
    device = parser.parse_args().device
    batchsize = parser.parse_args().batchsize

    # make conv model
    input_shape = (img_size,img_size,input_channel)
    print(f"CONV Layer with input shape {input_shape} & output channel {output_channel}")
    conv_layer = make_conv(input_shape,output_channel,kernel_size)
    
    # conv layer inference 
    inf_benchmark(device,conv_layer,img_size,input_channel,output_channel,batchsize,dtype)
