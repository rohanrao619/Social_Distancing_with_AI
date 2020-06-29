# Social Distancing with Artificial Intelligence
This project aims at monitoring people violating Social Distancing over video footage coming from CCTV Cameras. Uses YOLOv3 along with DBSCAN clustering for recognizing potential intruders. A Face Mask Classifier model (Resnet50) is trained and deployed for identifying people not wearing a face mask. Augmented masked faces are generated using facial landmarks for aiding the training process.

A detailed description of this project along with the results can be found [here](#project-description-and-results).

## Getting Started

### Prerequisites
Running this project on your local system requires the following packages to be installed :

* numpy
* matplotlib
* sklearn
* PIL
* cv2
* keras 
* face_detection
* face_recognition
* tqdm

They can be installed from the Python Package Index using pip as follows :
 
     pip install numpy
     pip install matplotlib
     pip install sklearn
     pip install Pillow
     pip install opencv-python
     pip install Keras
     pip install face-detection
     pip install face-recognition
     pip install tqdm
     
You can also use [Google Colab](https://colab.research.google.com/) in a Web Browser with most of the libraries preinstalled.
 
### Usage
This project is implemented using interactive Jupyter Notebooks. You just need to open the notebook on your local system or on [Google Colab](https://colab.research.google.com/) and execute the code cells in sequential order. The function of each code cell is properly explained with the help of comments.

Please download the following files (from the given links) and place them in the Models folder in the root directory :
1. YOLOv3 weights : https://pjreddie.com/media/files/yolov3.weights
2. Face Mask Classifier ResNet50 Keras Model : https://drive.google.com/drive/folders/1Q59338kd463UqUESwgt7aF_W46Fj5OJd?usp=sharing

Also before starting you need to make sure that the path to various files and folders in the notebook are updated according to your working environment. If you are using [Google Colab](https://colab.research.google.com/), then :
1. Mount Google Drive using : 

        from google.colab import drive
        drive.mount('drive/')
        
2. Update file/folder locations as `'drive/path_to_file_or_folder'`.

## Tools Used
* [NumPy](https://numpy.org/) : Used for storing and manipulating high dimensional arrays.
* [Matplotlib](https://matplotlib.org/) : Used for plotting.
* [Scikit-Learn](https://scikit-learn.org/stable/) : Used for DBSCAN clustering.
* [PIL](https://pillow.readthedocs.io/en/stable/) : Used for manipulating images.
* [OpenCV](https://opencv.org/) : Used for manipulating images and video streams.
* [Keras](https://keras.io/) : Used for designing and training the Face_Mask_Classifier model.
* [face-detection](https://github.com/hukkelas/DSFD-Pytorch-Inference) : Used for detecting faces with Dual Shot Face Detector.
* [face-recognition](https://github.com/ageitgey/face_recognition) : Used for detecting facial landmarks.
* [tqdm](https://github.com/tqdm/tqdm) : Used for showing progress bars.
* [Google Colab](https://colab.research.google.com/) : Used as the developement environment for executing high-end computations on its backend GPUs/TPUs and for editing Jupyter Notebooks. 

## Contributing
You are welcome to contribute :

1. Fork it (https://github.com/rohanrao619/Social_Distancing_with_AI/fork)
2. Create new branch : `git checkout -b new_feature`
3. Commit your changes : `git commit -am 'Added new_feature'`
4. Push to the branch : `git push origin new_feature`
5. Submit a pull request !

## License
This Project is licensed under the MIT License, see the [LICENSE](LICENSE) file for details.

## Project Description and Results
### Person Detection
[YOLO](https://pjreddie.com/darknet/yolo/) (You Only Look Once) is a state-of-the-art, real-time object detection system. It's Version 3 (pretrained on COCO dataset), with a resolution of 416x416 in used in this project for obtaining the bounding boxes of individual persons in a video frame. To obtain a faster processing speed, a resolution of 320x320 can be used. YOLOv3-tiny can also be used for speed optimization. However it will result in decreased detection accuracy.

### Face Detection
[Dual Shot Face Detector](https://github.com/Tencent/FaceDetection-DSFD) (DSFD) is used throughout the project for detecting faces. Common Face Detectors such as the Haar-Cascades or the MTCNN are not efficient in this particular use-case as they are not able to detect faces that are covered or have low-resolution. DSFD is also good in detecting faces in wide range of orientations. It is bit heavy on the pipeline, but produces accurate results.

### Face Mask Classifier
A slighly modified ResNet50 model (with base layers pretrained on imagenet) is used for classifying whether a face is masked properly or not. Combination of some AveragePooling2D and Dense (with dropout) layers ending with a Sigmoid or Softmax classifier is appended on top of the base layers. Different architectures can be used for the purpose, however complex ones should be avoided to reduce overfitting. For this classifier to work properly in all conditions, we need a diverse dataset that contains faces in various orientations and lighting conditions. For better results, our dataset should also cover people of different ages and gender. Finding such wide range of pictures having people wearing a face mask becomes a challenging task. Thus we need to apply numerous kinds of Augmentation before we start training.

### Masked Face Augmentation
It may seem a little akward, but with the power of deep learning in our hands, impossible is nothing!</br>
|![](Results/Masked_Face_Augmentation/Original.jpg)|![](Results/Masked_Face_Augmentation/Landmarks.jpg)|
|:---:|:---:|
|**Original**|**Facial Landmarks**|

## Final Notes
**Thanks for going through this Repository! Have a nice day.**</br>
</br>**S Rohan Rao**</br> 
Contact me : rohanrao619@gmail.com
