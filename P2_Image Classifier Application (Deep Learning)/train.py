#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 16:34:59 2018

@author: Jen-Feng Hsieh
"""

import argparse
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def parse_inputs():
    '''
	Parse inputs from command line
	Arguments:
		- data_dir : path of data directory
		- save_dir : path of directory to save a checkpoint
		- arch : choose architecture (AlexNet, VGG, Densenet)
		- learning_rate : learning rate for optimizer
        - hidden_units : hidden units for fully connected layer
        - epochs : epochs
		- gpu : use gpu for inference
    '''
    parser = argparse.ArgumentParser(description='Parse input arguments')
    parser.add_argument('data_dir', type=str, 
    					help='path of data directory')
    parser.add_argument('--save_dir', type=str, default='', 
    					help='path of directory to save a checkpoint')
    parser.add_argument('--arch', type=str, default='VGG', 
                        help='choose architecture (AlexNet, VGG, Densenet)')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
    					help='learning rate for optimizer')
    parser.add_argument('--hidden_units', type=str, default='1024', 
    					help='hidden units for fully connected layer')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='epochs')
    parser.add_argument('--gpu', default=False, action='store_true', 
                        help='use gpu for inference')

    return parser.parse_args()


def load_data(data_dir):
    if len(data_dir) > 0:
        data_dir = data_dir + '/'

    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'

    # Define the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    return (train_data, valid_data, test_data, 
            train_loader, valid_loader, test_loader)


def build_model(arch, hidden_units, output_size, drop_p=0.5):
    pretrained = True
    
    arch = arch.lower()
    if arch == 'alexnet':
        model = models.alexnet(pretrained)
        input_size = 9216
    elif arch == 'vgg':
        model = models.vgg19_bn(pretrained)
        input_size = 25088
    elif arch == 'densenet':
        model = models.densenet161(pretrained)
        input_size = 2208

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    hidden_units = list(map(int, hidden_units.split(',')))

    layers = OrderedDict([('fc1', nn.Linear(input_size, hidden_units[0])), 
                          ('relu1', nn.ReLU()), 
                          ('drop1', nn.Dropout(p=drop_p))])
    
    if len(hidden_units) > 1:
        layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
        for i, (h1, h2) in enumerate(layer_sizes):
            layers.update({'fc{:d}'.format(i+2): nn.Linear(h1, h2)})
            layers.update({'relu{:d}'.format(i+2): nn.ReLU()})
            layers.update({'drop{:d}'.format(i+2): nn.Dropout(p=drop_p)})
    
    layers.update({'fc{:d}'.format(len(hidden_units) + 1): nn.Linear(hidden_units[-1], 
                                                                     output_size)})
    layers.update({'output': nn.LogSoftmax(dim=1)})
    
    classifier = nn.Sequential(layers)
    model.classifier = classifier
    
    return model


def device_mode(gpu):
    if gpu and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device    


def validation(model, valid_loader, criterion, gpu=False):
    loss = 0
    accuracy = 0
    
    device = device_mode(gpu)
    model.to(device)
    
    # Make sure network is in eval mode for inference
    model.eval()

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)
            loss += criterion(outputs, labels).item()

            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss, accuracy


def training(model, train_loader, valid_loader, epochs, print_every, criterion, 
             optimizer, gpu=False):
    steps = 0
    running_loss = 0
    accuracy = 0
    
    device = device_mode(gpu)
    model.to(device)
    
    for e in range(epochs):
        # Make sure training is back on
        model.train()
        
        for inputs, labels in train_loader:
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean() 
            
            if steps % print_every == 0:
                valid_loss, valid_accuracy = validation(model, valid_loader, criterion, gpu)
                
                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Training Accuracy: {:.3f}.. ".format(accuracy / print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss / len(valid_loader)),
                      "Validation Accuracy: {:.3f}".format(valid_accuracy / len(valid_loader)))
                
                running_loss = 0
                accuracy = 0
                
                # Make sure training is back on
                model.train()

    return model


def main():
    # Parse Inputs
    inputs = parse_inputs()


    # Load Data
    (train_data, valid_data, test_data, 
     train_loader, valid_loader, test_loader) = load_data(inputs.data_dir)


    # Build Model
    output_size = 102
    model = build_model(inputs.arch, inputs.hidden_units, output_size)


    # Train Model
    print_every = 64
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=inputs.learning_rate)
    model = training(model, train_loader, valid_loader, inputs.epochs, print_every, 
                     criterion, optimizer, inputs.gpu)


    # Test Model
    test_loss, accuracy = validation(model, test_loader, criterion, inputs.gpu)
    print("Testing Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
          "Testing Accuracy: {:.3f}".format(accuracy / len(test_loader)))


    # Save Checkpoint
    model.class_to_idx = train_data.class_to_idx
    # Move to CPU 
    model.cpu()

    checkpoint = {'arch': inputs.arch, 
                  'classifier': model.classifier, 
                  'class_to_idx': model.class_to_idx, 
                  'state_dict': model.state_dict()}

    if len(inputs.save_dir) > 0:
        save_path = inputs.save_dir + '/'

    save_path = save_path + 'checkpoint.pth'
    torch.save(checkpoint, save_path)

    pass


if __name__ == '__main__':
	main()