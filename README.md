# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, we first develop code for an image classifier built with PyTorch, then convert it into a command line application.
# Flower Image Classifier

This project is part of Udacity's AI Programming with Python Nanodegree program. It implements an image classifier using PyTorch models to recognize different species of flowers. The dataset used contains 102 flower categories.

## Project Overview

The image classifier is built using PyTorch and consists of two command line applications: `train.py` and `predict.py`. The `train.py` script trains the image classifier on the provided dataset, while the `predict.py` script uses a trained model to predict the class of a given input image.

## Usage

### Training the Classifier

To train the image classifier, use the `train.py` script. You need to provide the following arguments:

- `data_dir` (mandatory): Provide the path to the data directory containing the training and validation sets.
- `--save_dir` (optional): Provide the directory where the trained model checkpoint will be saved. If not specified, the default directory will be used.
- `--arch` (optional): Specify the architecture to use for the classifier. Use `--arch vgg16` to use the VGG16 model, otherwise, the AlexNet model will be used.
- `--lrn` (optional): Learning rate for the training. Default value is 0.001.
- `--hidden_units` (optional): Number of hidden units in the classifier.(there are 3 hidden layers:{6272,1045,522}hidden units)
- `--epochs` (optional): Number of epochs for training. Default value is 10.
- `--GPU` (optional): Specify this flag if you want to use GPU for training.

Example usage:
python train.py data_dir --save_dir save_dir --arch vgg13 --lrn 0.001 --hidden_units 2048 --epochs 10 --GPU

### Predicting with the Classifier

To make predictions using the trained image classifier, use the `predict.py` script. You need to provide the following arguments:

- `image_dir` (mandatory): Provide the path to the input image file.
- `load_dir` (mandatory): Provide the path to the trained model checkpoint.
- `--top_k` (optional): Return the top K most likely classes. Default value is 1.
- `--category_names` (optional): Provide the JSON file name that maps the category labels to their real names. If not specified, the class indices will be used.
- `--GPU` (optional): Specify this flag if you want to use GPU for prediction.

Example usage:
python predict.py image_dir load_dir --top_k 3 --category_names cat_to_name.json --GPU

## Acknowledgements

The flower image dataset used in this project is provided by Udacity. The pre-trained models used are from the torchvision.models module in PyTorch.

