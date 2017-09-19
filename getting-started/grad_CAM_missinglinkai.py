import caffe
import missinglink

net_model = 'models/vgg16/VGG16-deploy.prototxt'
net_weights = 'models/vgg16/VGG16.caffemodel'

net = caffe.Net(net_model, net_weights, caffe.TEST)

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id='replace with owner id',
    project_token='replace with project token')

missinglink_callback.set_properties(class_mapping={0:'dog', 1:'cat'})

path = 'http://cmeimg-a.akamaihd.net/640/photos.demandstudios.com' + \
    '/getty/article/103/49/516464087.jpg'

missinglink_callback.generate_grad_cam(path, model=net)
