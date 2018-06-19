#To run sub_directory python file on cloud IDE
import os, sys


#Ignore CPU supports instrctions AVX2 FMA 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



#Folder List
value = ["00_tf_tutorial","01_linear_regression","02_artificial_neuralnet","03_tensorboard","04_mnist"] #

#Run Python File
os.system("python /projects/tf_project/" + value[0] + ".py")      


"""
http://blog.danggun.net/4064

sudo apt-get install python-pip python-dev

# Ubuntu/Linux 64-bit, CPU 전용, Python 3.5
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl
sudo pip3 install --upgrade $TF_BINARY_URL

pip3 install tensorflow
"""