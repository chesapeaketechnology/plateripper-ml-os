# plateripper-ml-os
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

An opensource ANPR implementation for experimentation.

## Currently Supported Features
* [x] [Crop Detections and Save as New Image](#crop)
* [x] [License Plate Recognition Using Tesseract OCR](#license)
* [x] [Apply Tesseract OCR to Detections to Extract Text](#ocr)

## Getting Started
### Conda
Depending on hardware you have available to you:
```
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

Installing Darknet is optional if you want to do training:
```
git clone https://github.com/AlexeyAB/darknet
cd darknet
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
#decide if you want GPU or not
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
make
```


## Training
Existing weights are included for ANPR in the repository at `data/custom.weights` as well as converted TensorFlow models under `checkpoints/custom-416`.

Pull the OIDv4 Downloader included as a submodule:
```
git submodule update --init
```


Training Dataset
Leverage the OIDv4 Downloader to pull vehicle images from OID.
```
## Download training dataset
python main.py downloader --classes 'Vehicle registration plate' --type_csv train
```

Validation Dataset
```
## Download training dataset
python main.py downloader --classes 'Vehicle registration plate' --type_csv validation
```

Cover all the labels to YOLOv4:
```
python convert_annotations.py
```

Download pre-trained existing weights:
```
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

Organize training data and generate text files:
```
python generate_train.py
python generate_test.py
```

Train!
```
./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -map
```


## After training weights

Copy and paste your custom .weights file into the 'data' folder and copy and paste your custom .names into the 'data/classes/' folder if you have additional labels for vehicles etc. Currently this repo only supports license plate recognition.

## Convert YOLOv4 To TensorFlow
The following commands will allow you to run your custom yolov4 model. (video and webcam commands work as well)
```
# custom yolov4
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 

# Run custom yolov4 tensorflow model
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/car.jpg
```


## License Plate Recognition Using Tesseract OCR
I have created a custom function to feed Tesseract OCR the bounding box regions of license plates found by my custom YOLOv4 model in order to read and extract the license plate numbers. Thorough preprocessing is done on the license plate in order to correctly extract the license plate number from the image. 

<strong>Disclaimer: In order to run tesseract OCR you must first download the binary files and set them up on your local machine. Please do so before proceeding or commands will not run as expected!</strong>

Official Tesseract OCR Github Repo: [tesseract-ocr/tessdoc](https://github.com/tesseract-ocr/tessdoc)

How To Install Tesseract on Mac or Linux Machines: [PyImageSearch Article](https://www.pyimagesearch.com/2017/07/03/installing-tesseract-for-ocr/)

For Windows: [Windows Install](https://github.com/UB-Mannheim/tesseract/wiki)

Once you have Tesseract properly installed you can move onwards. The included trained YOLOv4 model will detect license plates based on limited training.

### Running License Plate Recognition on Images (video example below)
The license plate recognition works wonders on images. All you need to do is add the `--plate` flag on top of the command to run the custom YOLOv4 model.

Try it out on this image in the repository!
```
# Run License Plate Recognition
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/car2.jpg --plate
```

### Image Output
The output from the above command should print any license plate numbers found to your command terminal as well as output and save the following image to the `detections` folder.


### Running License Plate Recognition on Video
Running the license plate recognition straight on video at the same time that YOLOv4 object detections causes a few issues. Tesseract OCR is fairly expensive in terms of time complexity and slows down the processing of the video to a snail's pace. It can still be accomplished by adding the `--plate` command line flag to any detect_video.py commands.

Running License Plate Recognition with detect_video.py is done with the following command.
```
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/license_plate.mp4 --output ./detections/recognition.avi --plate
```

The recommended route I think is more efficient is using this command. Customize the rate at which detections are cropped within the code itself.
```
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/license_plate.mp4 --output ./detections/recognition.avi --crop
```

Now play around with [license_plate_recognizer.py](https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/license_plate_recognizer.py) and have some fun!


## YOLOv4 Using TensorFlow Lite (.tflite model)
Can also implement YOLOv4 using TensorFlow Lite. TensorFlow Lite is a much smaller model and perfect for mobile or edge devices (raspberry pi, etc).
```bash
# Save tf model for tflite converting
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 --framework tflite

# yolov4
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416.tflite

# yolov4 quantize float16
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416-fp16.tflite --quantize_mode float16

# yolov4 quantize int8
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416-int8.tflite --quantize_mode int8 --dataset ./coco_dataset/coco/val207.txt

# Run tflite model
python detect.py --weights ./checkpoints/yolov4-416.tflite --size 416 --model yolov4 --images ./data/images/car.jpg --framework tflite
```
### Result Image (TensorFlow Lite)
You can find the outputted image(s) showing the detections saved within the 'detections' folder.

These models work well on Android and other embedded devices.

## Command Line Args Reference

```bash
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)

detect.py:
  --images: path to input images as a string with images separated by ","
    (default: './data/images/kite.jpg')
  --output: path to output folder
    (default: './detections/')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.25)
  --count: count objects within images
    (default: False)
  --dont_show: dont show image output
    (default: False)
  --info: print info on detections
    (default: False)
  --crop: crop detections and save as new images
    (default: False)
    
detect_video.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/video.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.25)
  --count: count objects within video
    (default: False)
  --dont_show: dont show video output
    (default: False)
  --info: print info on detections
    (default: False)
  --crop: crop detections and save as new images
    (default: False)
```

### References  

  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Automatic-License-Plate-Recognition](https://github.com/udaylunawat/Automatic-License-Plate-Recognition)
  * [AI Guy](https://github.com/theAIGuysCode/yolov4-custom-functions)
  * [Tensorflow Examples](https://github.com/tensorflow/examples)


