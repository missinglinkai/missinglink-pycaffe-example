# MissingLinkAI SDK example for PyCaffe

## Requirements

You need Python 2.7 or 3.5 on your system to run this example.

To create an environment to run the example:
- You are strongly recommended to use [`Docker`](https://www.docker.com/) to create a sandboxed environment for Caffe
- You can run an official Caffe Docker by following the instructions [`here`](https://github.com/BVLC/caffe/tree/master/docker)

## Run

In order to run an experiment with MissingLinkAI, you would need to first create a
project and obtain the credentials on the MissingLinkAI's web dashboard.

With the `owner_id` and `project_token`, you can run this example from terminal within Docker, from Caffe's root,
after copying the source file from mnist_missinglinkai.py into Docker.
```bash
python mnist_missinglinkai.py --owner-id 'owner_id' --project-token 'project_token'
```

Alternatively, you can copy these credentials and set them in source files.

## Examples

These examples train classification models for MNIST dataset.

- [mnist.py](https://github.com/missinglinkai/missinglink-keras-example/blob/master/getting-started/mnist_missinglinkai.py): training with PyCaffe's solve
