from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa 
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import pix2pix
if tf.test.gpu_device_name(): 

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:

   print("Please install GPU version of TF")
class model:
    def __init__(self):
        # self.model = pix2pix.unet_generator(3, norm_type='instancenorm')
        # self.model.load_weights("model_gen_a_vgg_15.hdf5")
        self.model = self.vgg19()
        self.img_inputs = []
        self.outputs = []
        self.list_file = []
        
    def vgg19(self):
        """
        It loads the weights of the generator model.
        :return: The model is being returned.
        """
        vgg19 = tf.keras.applications.VGG19(include_top=False, input_shape=(512,512,3))
        list_out = [2,5,10,15,20]
        gen_s2w_vgg19 = self.gen_model(vgg19,list_out)
        gen_s2w_vgg19.load_weights(".\\gen_s2w_vgg19_26.hdf5")
        return gen_s2w_vgg19
    def res_net(self,number,filter=512, kernel_size=3, stride=1,padding="same"):
        """
        This function creates a convolutional layer with a kernel size of 3, stride of 1, and padding of
        "same". It then adds an instance normalization layer and a ReLU activation layer
        
        :param number: the number of the res_block
        :param filter: The number of filters in the convolutional layer, defaults to 512 (optional)
        :param kernel_size: The size of the convolutional kernel, defaults to 3 (optional)
        :param stride: The stride of the convolution. If you're not familiar with this, check out this
        guide, defaults to 1 (optional)
        :param padding: "same" means that the output will have the same size as the input, defaults to
        same (optional)
        :return: A sequential model with two layers.
        """
        initializer = tf.random_uniform_initializer(.0,.02)
        name = "res_block"+str(number)
        res_block = tf.keras.models.Sequential(name=name)
        res_block.add(tf.keras.layers.Conv2D(filters=filter,
                                            kernel_size=(kernel_size,kernel_size),
                                            strides=stride,
                                            padding=padding,
                                            kernel_initializer=initializer))
        res_block.add(pix2pix.InstanceNormalization())
        res_block.add(tf.keras.layers.ReLU())
        return res_block
    def resnet(self):
        """
        It creates a resnet model with 4 residual blocks.
        :return: The res_model is being returned.
        """
        num_res = 4
        res_block=[]
        for i in range(num_res):
            res_block.append(self.res_net(number=(i+1)))
        res_input = tf.keras.layers.Input(shape=(16,16,512))
        x = res_input
        for res in res_block:
            out=res(x)
            x = tf.keras.layers.Concatenate()([out,x])
        output = self.res_net(number=9)(x)
        res_model = tf.keras.models.Model(inputs=res_input, outputs=output)
        return res_model
    def up_sample(self,filter, size):
        """
        This function takes in a filter and a size and returns a sequential model that upsamples the
        input by a factor of 2, convolves it with the filter, normalizes it, and applies a ReLU
        activation
        
        :param filter: number of filters in the convolutional layer
        :param size: The size of the up-sampling filter
        :return: The up_block is being returned.
        """
        init = tf.random_uniform_initializer(0.0,.002)
        up_block = tf.keras.models.Sequential()
        up_block.add(tf.keras.layers.UpSampling2D(size=(2,2)))
        up_block.add(tf.keras.layers.Conv2D(filters=filter,kernel_size=size,padding="same",strides=(1,1),kernel_initializer=init ))
        up_block.add(pix2pix.InstanceNormalization())
        up_block.add(tf.keras.layers.ReLU())
        return up_block
    def head_model(self,VGG19,list_out):
        """
        The function takes in a VGG19 model and a list of layers to output. It then creates a new model
        with the input layer of the VGG19 model and the layers in the list of layers to output
        
        :param VGG19: The VGG19 model
        :param list_out: list of layers to be outputted
        :return: The head model is being returned.
        """
        list_output = []
        input = tf.keras.layers.Input(shape=(512,512,3))
        x= input
        for i in range(1,len(VGG19.layers)):
            x = VGG19.layers[i](x)
            if i in list_out:
                list_output.append(x)
        list_output.append(x)
        head = tf.keras.Model(inputs=input, outputs = list_output)
        return head
    def tail_model(self):
        """
        The function takes in a list of inputs, and then upsamples each input to the next size up
        :return: The model is being returned.
        """
        initializer = tf.random_uniform_initializer(.0,.02)
        input_2 = tf.keras.layers.Input((16,16,512))

        list_input = [tf.keras.layers.Input(shape=(32,32,512)),
                    tf.keras.layers.Input(shape=(64,64,512)),
                    tf.keras.layers.Input(shape=(128,128,256)),
                    tf.keras.layers.Input(shape=(256,256,128)),
                    tf.keras.layers.Input(shape=(512,512,64)),]
        x_2 = input_2
        ups = [
                self.up_sample(512,3),
                self.up_sample(512,3),#(32,32,512)
                self.up_sample(512,3),#(64,64,256)
                self.up_sample(256,3),#(128,128,128)
                self.up_sample(128,3),#(256,256,64)
                ]
        for up,skip in zip(ups,list_input):
            x_2 = up(x_2)
            filter = x_2.shape[-1]
            x_2 = tf.keras.layers.Conv2D(filter,(3,3),strides=(1,1),padding="same",kernel_initializer=initializer)(x_2)
            x_2 = tf.keras.layers.Conv2D(filter,(3,3),strides=(1,1),padding="same",kernel_initializer=initializer)(x_2)
            x_2 = tf.keras.layers.Concatenate()([x_2,skip])
        x_2 = tf.keras.layers.Conv2D(64,3,strides=(1,1), padding="same", kernel_initializer=tf.random_uniform_initializer(.0,.002))(x_2)
        x_2 = pix2pix.InstanceNormalization()(x_2)
        x_2 = tf.keras.layers.ReLU()(x_2)
        x_2 = tf.keras.layers.Conv2D(3,(5,5),(1,1),padding="same",kernel_initializer=tf.random_uniform_initializer(.0,.002))(x_2)
        x_2 = tf.keras.activations.tanh(x_2)

        tail_model = tf.keras.Model(inputs=[input_2,list_input[:]], outputs=x_2)
        return tail_model
    def gen_model(self,pretrain,list_out):
        """
        The function takes in a pretrained model, and a list of layers to be outputted from the
        pretrained model. The function then creates a new model with the pretrained model as the head,
        and a new model as the tail. The output of the pretrained model is then fed into the new model
        
        :param pretrain: pretrained model
        :param list_out: list of output layers from the pretrained model
        :return: The model is being returned.
        """
        # It creates a new model with the input layer of the VGG19 model and the layers in the list of
        # layers to output
        head = self.head_model(pretrain,list_out)
        # It creates a new model with the input layer of the VGG19 model and the layers in the list of
        # layers to output
        tail  = self.tail_model()
        # It creates a resnet model with 4 residual blocks.
        res = self.resnet()
        # Setting the trainable attribute of the head, tail, and res models to False, True, and True
        # respectively.
        head.trainable = False
        tail.trainable = True
        res.trainable = True
        # It creates a placeholder for the input of the model.
        input_gen = tf.keras.layers.Input((512,512,3))
        # Reversing the list and then adding the first element to the end of the list.
        output_head = head(input_gen)
        x_head = output_head.pop(-1)
        output_head.reverse()
        output_res = res(x_head)
        output_head.insert(0,output_res)
        output_tail = tail(output_head)
        gen_model = tf.keras.models.Model(inputs=input_gen,outputs=output_tail)
        return gen_model
    def processing_img(self, input_img, size = (512,512)):
        """
        It takes an image, converts it to an array, normalizes it, and returns the array
        
        :param input_img: The image to be processed
        :param size: The size of the image to be generated
        :return: The image is being returned as an array.
        """
        array = img_to_array(input_img)
        array = (array - 127.5) / 127.5
        return array
    def get_inputs(self,input_dir):
        """
        It takes a directory as input, and then it takes all the images in that directory and puts them
        into a list.
        
        :param input_dir: The directory where the images are stored
        """
        self.img_inputs.clear()
        if(os.path.isdir(input_dir)):
            list_files = os.listdir(input_dir)
            for i in list_files:
                self.get_img(input_dir + "/" + i)
            
    def get_img(self, path):
        """
        If the path is a file, and the file extension is in the list of image formats, then read the
        image, convert it to RGB, resize it to 256x256, and append it to the list of images.
        
        :param path: The path to the image you want to classify
        """
        
        img_format = ['jpg' , "jpeg", "jfif" , "pjpeg" , "pjp", "png"]
        # if(os.path.isfile(path)):
        #     list_file = os.listdir(path)
        #     for i in list_file:
        #         if(i.split(".")[1] in img_format):
        #             path_img = os.path.join(path,i)
        #             img = self.processing_img(path_img)
        #             self.img_inputs.append(img)
        if(os.path.isfile(path)):
            if path.split(".")[1] in img_format:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,(512,512))
                self.img_inputs.append(img)
                img_name = path.split("/")[-1]    
                self.list_file.append(img_name)
                
            
    def predict(self, img):
        """
        The function takes an image, processes it, and then predicts the output of the image
        
        :param img: the image to be processed
        """
        
        img = self.processing_img(img)
        img = np.expand_dims(img, axis=0)
        output = self.model.predict(img,verbose=0)
        img_process = self.after_process(output[0])
        self.outputs.insert(len(self.outputs),img_process)
    def after_process(self, img):
        """
        It takes the image, adds 1 to it, multiplies it by 127.5, converts it to an unsigned integer,
        converts it to BGR, denoises it, and converts it back to RGB
        
        :param img: The image to be processed
        :return: The image is being returned.
        """
        img =(img + 1)*127.5
        img = img.astype(np.uint8)
        img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.fastNlMeansDenoisingColored(img,None,5,3,7,21) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img 
        
        
    def getvideo(self,path):
        video_format = ["mp4", "avi", "mov", "wmv", "flv", "mkv"]
        if path.split(".")[1] in video_format:
            cap = cv2.VideoCapture(path)
            while(True):
                ret,frame = cap.read()
                if not ret:
                    break
                else:
                    frame = cv2.resize(frame,(256,256))
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    frame = (frame - 127.5) / 127.5
                    self.img_inputs.append(frame)
    def save(self, dir):
        """
        It takes the output of the model and saves it to the directory specified in the argument
        
        :param dir: the directory where the images are stored
        """
        for i in range(len(self.outputs)):
            img = self.outputs[i]
            img = img.astype(np.uint8)
            plt.imsave(dir + "/" +self.list_file[i].split(".")[0] + "_chuyen.jpg",img)