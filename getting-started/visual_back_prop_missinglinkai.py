import caffe
import missinglink

net_model = 'models/vgg16/VGG16-deploy.prototxt'
net_weights = 'models/vgg16/VGG16.caffemodel'

net = caffe.Net(net_model, net_weights, caffe.TEST)

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id='replace with owner id',
    project_token='replace with project token')

path = 'http://l7.alamy.com' + \
    '/zooms/b76d255dd51e493e8c0fd5d5aa85f96f/lumbermill-cp93p7.jpg'

missinglink_callback.generate_grad_cam(path, model=net)
