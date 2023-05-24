# Author:Himanshu Singh Sikarwar
# Date:feb-2023
import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args
# import torch
# from torchvision import datasets, transforms

def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data



def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader
# import torch

def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cpu':
        # No need to print this message since the function is already returning the CPU device
        pass
    
    return device
# import torch
# from torchvision import models

def primaryloader_model(architecture="vgg16"):
    if architecture == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")
        
    for param in model.parameters():
        param.requires_grad = False
        
    return model
def initial_classifier(model, hidden_units):
    #Used OrderedDict to preserve the order in which the keys are inserted
    # defines the custom classifier with 3 hidden layers and an output layer
    classifier = nn.Sequential(OrderedDict([
        ('input', nn.Linear(25088, 6272)), # input layer
        ('relu1', nn.ReLU()), # first ReLU activation function
        ('dropout01', nn.Dropout(0.05)), # first dropout layer
        ('hiddenlayer1', nn.Linear(6272,1045)), # first hidden layer
        ('relu2',nn.ReLU()), # second ReLU activation function
        ('hiddenlayer2', nn.Linear(1045,522)), # second hidden layer
        ('relu3',nn.ReLU()), # third ReLU activation function
        ('hiddenlayer3', nn.Linear(522,102)), # third hidden layer
        ('output', nn.LogSoftmax(dim=1)) # output layer with LogSoftmax activation function
    ]))
    # replaces the pre-trained classifier with the custom one defined above
    model.classifier = classifier
    return classifier
# import torch

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            output = model(inputs)
            test_loss += criterion(output, labels).item()
            
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy =(accuracy+ equality.type(torch.FloatTensor).mean().item())/len(testloader)
            
    return test_loss, accuracy

def network_trainer(Model, Trainloader, Testloader, Device, Criterion, Optimizer, Epochs=None, Print_every=50, Steps=0):
    
    # If the number of epochs is not specified, set it to 12
    if Epochs is None:
        Epochs = 12
        print("Number of Epochs specified as 12.")
    
    # Move the model to the specified device
    Model.to(Device)
    print("Training process initializing .....\n")

    # Loop over the specified number of epochs
    for epoch in range(Epochs):
        running_loss = 0
        # Set the model to train mode
        Model.train()
        
        # Loop over the training data in batches
        for batch_idx, (inputs, labels) in enumerate(Trainloader):
            Steps += 1
            # Move the inputs and labels to the specified device
            inputs, labels = inputs.to(Device), labels.to(Device)
            
            # Zero the gradients, perform a forward pass, compute the loss, and perform a backward pass
            Optimizer.zero_grad()
            outputs = Model(inputs)
            loss = Criterion(outputs, labels)
            loss.backward()
            Optimizer.step()
            
            # Add the loss for this batch to the running loss
            running_loss += loss.item()
            
            # If the number of steps is a multiple of print_every, print the validation metrics
            if Steps % Print_every == 0:
                # Set the model to evaluation mode
                Model.eval()
                with torch.no_grad():
                    # Compute the validation loss and accuracy on the testing set
                    valid_loss, accuracy = validation(Model, Testloader, Criterion, Device)
                    
                # Print the epoch number, training loss, validation loss, and validation accuracy
                print("Epoch: {}/{} | ".format(epoch+1, Epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/Print_every),
                      "Validation Loss: {:.4f} | ".format(valid_loss/len(Testloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(Testloader)))
                running_loss = 0
                # Set the model back to train mode
                Model.train()

    # Return the trained model
    return Model
def validate_model(Model, Testloader, Device):
    # Set the model to evaluation mode
    Model.eval()
    
    # Initialize variables to keep track of accuracy and total loss
    correct = 0
    total = 0
    total_loss = 0
    
    # Loop over the test data in batches
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(Testloader):
            # Move the inputs and labels to the specified device
            inputs, labels = inputs.to(Device), labels.to(Device)
            
            # Perform a forward pass and compute the loss
            outputs = Model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            
            # Compute the number of correctly classified samples in this batch
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    # Compute and print the accuracy and average loss over all test samples
    accuracy = 100 * correct / total
    average_loss = total_loss / len(Testloader)
    print('Accuracy on test images: {:.2f}%'.format(accuracy))
    print('Average loss on test images: {:.4f}'.format(average_loss))
def initial_checkpoint(Model, Optimizer, Save_Dir, Train_data):
    # Save model at checkpoint
    if Save_Dir:
        Model.class_to_idx = Train_data.class_to_idx
        torch.save({
            'epochs': 12,
            'state_dict': Model.state_dict(),
            'class_to_idx': Model.class_to_idx,
            'optimizer_dict': Optimizer.state_dict()
        }, 'checkpoint.pth')

        # Save checkpoint
        checkpoint = {
            'class_to_idx': Model.class_to_idx,
            'state_dict': Model.state_dict()
        }
        torch.save(checkpoint, 'my_checkpoint.pth')
    else:
        print("Model checkpoint directory not specified, model will not be saved.")
def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create data loaders
    train_data = test_transformer(train_dir)
    valid_data = train_transformer(valid_dir)
    test_data = train_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # Load model and set classifier
    model = primaryloader_model(architecture=args.arch)
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    device = check_gpu(gpu_arg=args.gpu)
    model.to(device)
    
    # Set learning rate and optimizer
    learning_rate = args.learning_rate or 0.005
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Train model
    print_every = 50
    steps = 0
    trained_model = network_trainer(model, trainloader, validloader, device, criterion, optimizer, args.epochs, print_every, steps)
    print("\nTraining process is completed!!")
    
    # Test model
    validate_model(trained_model, testloader, device)
   
    # Save model checkpoint
    if args.save_dir:
        initial_checkpoint(trained_model, args.save_dir, train_data)
    else:
        print("Model checkpoint directory not specified, model will not be saved.")

if __name__ == '__main__':
    main()

