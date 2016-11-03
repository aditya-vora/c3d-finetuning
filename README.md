
# C3D-finetuning
This repository contains the scripts to fine tune the currently available C3D model for video classification using Theano/Lasagne as well testing the new fine tuned model for new test videos.

# Getting Started

The C3D pre-trained model that we use which can be loaded with Theano/Lasagne packages can be downloaded by executing the following line of code from the terminal: 

```
wget -N https://data.vision.ee.ethz.ch/gyglim/C3D/c3d_model.pkl
```

### Prerequisites

The scripts assumes following packages are successfully installed on the system: 

* Theano
* Lasagne
* OpenCV 
* lmdb
* pickle

### References

[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 2015 IEEE International Conference on Computer Vision (ICCV). IEEE, 2015.
