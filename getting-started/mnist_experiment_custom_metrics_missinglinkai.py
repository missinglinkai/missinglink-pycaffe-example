# Adopted and modified from PyCaffe's MNIST example.
# https://github.com/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb
#
# In this example, we will train a LeNet network on the MNIST dataset
# We will then integrate MissingLink SDK in order to remotely monitor our training, validation
# and testing process.

import argparse
import caffe
import missinglink
import os

from caffe import layers as L, params as P
from subprocess import call

caffe_root = '../'  # This file should be run from {caffe_root}/examples (otherwise change this line)

os.environ['GLOG_minloglevel'] = '1'  # Set the logging level

os.chdir(caffe_root)  # Preare to run scripts from caffe root
call('data/mnist/get_mnist.sh')  # Download MNIST data
call('examples/mnist/create_mnist.sh')  # Prepare MNIST data
os.chdir('examples')  # Back to examples

# MissingLink credentials
OWNER_ID = 'Fill in your owner id'
PROJECT_TOKEN = 'Fill in your project token'

def lenet(lmdb, batch_size):
    # Our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

with open('mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_train_lmdb', 64)))

with open('mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('mnist/mnist_test_lmdb', 100)))

caffe.set_mode_cpu()  # Or use gpu by running the next line instead if your machine has access to a GPU
# caffe.set_mode_gpu()

# Provide an alternative to provide MissingLinkAI credential
parser = argparse.ArgumentParser()
parser.add_argument('--owner-id')
parser.add_argument('--project-token')

# Override credential values if provided as arguments
args = parser.parse_args()
OWNER_ID = args.owner_id or OWNER_ID
PROJECT_TOKEN = args.project_token or PROJECT_TOKEN

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id=OWNER_ID, project_token=PROJECT_TOKEN)
missinglink_callback.set_properties(
    display_name='MNIST', description='LeNet network')
solver = missinglink_callback.create_wrapped_solver(
    caffe.SGDSolver, 'mnist/lenet_auto_solver.prototxt')

from datetime import datetime

time_before_experiment = datetime.utcnow()
kwh_cost = 0.25  # A very expensive electrical rate in USD
gpu_wattage_kW = 16  # 64 typical modern single GPU unit each ~ 0.25 kW

def cost_of_running_experiment():
    time_elapsed_hours = \
        (datetime.utcnow() - time_before_experiment).total_seconds() / 3600
    return gpu_wattage_kW * time_elapsed_hours * kwh_cost

missinglink_callback.set_monitored_blobs(
    ['loss', cost_of_running_experiment])

solver.solve()
