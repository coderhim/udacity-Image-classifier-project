import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import check_gpu
from torchvision import models
def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image',type=str,help='Point to impage file for prediction.',required=True)
    parser.add_argument('--checkpoint',type=str,help='Point to checkpoint file as str.',required=True)
    parser.add_argument('--top_k',type=int,help='Choose top K matches as int.',default=5)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    args = parser.parse_args()
    
    return args
def load_the_checkpoint(path='checkpoint.pth', device='cpu'):
    try:
        checkpoint = torch.load(path, map_location=device)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Failed to load checkpoint from file '{path}': {e}")
        return None

    model = models.vgg16(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Load model state dict from checkpoint
    model.class_to_idx = checkpoint['model_class_2_index']
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move the model to the specified device
    model.to(device)
def process_image(image):
    """Preprocesses an image and returns it as a numpy array.

    Args:
        image_path (str): The path of the image to preprocess.

    Returns:
        np.ndarray: The preprocessed image as a numpy array.
    """
    # Open the image and convert to RGB
    a_image = Image.open(image).convert('RGB')
    
    # Resize the image while keeping its aspect ratio
    a_image.thumbnail(size=(256, 256))
    
    # Get the dimensions of the image
    width, height = a_image.size
    
    # Crop the center of the image to a 224x224 square
    crop_size = 224
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    a_image = a_image.crop((left, top, right, bottom))
    
    # Convert the image to a PyTorch tensor and normalize it
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tensor = normalize(to_tensor(a_image))
    
    # Convert the tensor to a numpy array
    np_array = tensor.numpy()
# np_processed_image = tensor.numpy()
    return np_array

def predict(image, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    image: string. Path to image, directly to image and not to folder.
    model: pytorch neural network.
    top_k: integer. The top K classes to be calculated
    
    returns top_probabilities(k), top_labels
    '''
    
    # No need for GPU on this part (just causes problems)
    model.to("cpu")
    
    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] # This is not the correct way to do it but the correct way isnt working thanks to cpu/gpu issues so I don't care.
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers
def print_probability(probs, flowers):
    # Convert two lists into a dictionary to print on screen
    for i, (flower, prob) in enumerate(zip(flowers, probs), 1):
        print(f"Rank {i}: Flower: {flower}, likelihood: {ceil(prob*100)}%")
def main():
    
    # parse command line arguments using arg_parser() function
    args = arg_parser()
    
    # load a mapping of category names to flower names from a JSON file using json.load()
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    # load the pre-trained neural network model checkpoint using load_the_checkpoint() function
    model = load_the_checkpoint(args.checkpoint)
    
    # process the input image using process_image() function
    image_tensor = process_image(args.image)
    
    # determine whether to use the GPU or CPU using check_gpu() function
    device = check_gpu(gpu_arg=args.gpu)
    
    # predict the flower name and probability of the input image using predict() function
    top_probs, top_labels, top_flowers = predict(image_tensor, model, device, cat_to_name, args.top_k)
    
    # print the top K probabilities and their corresponding flower names using print_probability() function
    print_probability(top_flowers, top_probs)

if __name__ == '__main__':
    # call the main() function if the script is executed directly
    main()