#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 16:34:59 2018

@author: Jen-Feng Hsieh
"""

import argparse
import numpy as np
import json

import torch
from torchvision import models


def parse_inputs():
    '''
	Parse inputs from command line
	Arguments:
		- input : path of image file
		- checkpoint : path of checkpoint file
		- top_k : return top KK most likely classes
		- category_names : path of mapping file for categories' real names
		- gpu : use gpu for inference
    '''
    parser = argparse.ArgumentParser(description='Parse input arguments')
    parser.add_argument('input', type=str, 
    					help='path of image file for prediction')
    parser.add_argument('checkpoint', type=str, 
    					help='path of checkpoint file')
    parser.add_argument('--top_k', type=int, default=5,  
    					help='return top KK most likely classes')
    parser.add_argument('--category_names', type=str, default='', 
    					help='path of mapping file')
    parser.add_argument('--gpu', default=False, action='store_true', 
    					help='use gpu for inference')

    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    pretrained = True

    arch = checkpoint['arch']
    if arch == 'AlexNet':
        model = models.alexnet(pretrained)
    elif arch == 'VGG':
        model = models.vgg19_bn(pretrained)
    elif arch == 'Densenet':
        model = models.densenet161(pretrained)

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
    img = Image.open(image_path)
    
    # Resize the images
    # width > height
    if img.size[0] > img.size[1]:
        # Set the height to 256
        img.thumbnail((10000, 256))
    else:
        # Set the width to 256
        img.thumbnail((256, 10000)) 
    
    # Crop image to 224x224
    left = (img.width - 224) / 2
    bottom = (img.height - 224) / 2
    img = img.crop((left, bottom, left + 224, bottom + 224))
    
    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    img = (img - mean) / std
    
    # Reorder Dimensions to Color Channels X Width X Height
    img = img.transpose((2, 0, 1))
    
    return img


def device_mode(gpu):
    if gpu and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device    


def predict(image_path, model, gpu=False, top_n=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''  
    image = process_image(image_path)
    # Convert to Tensor 
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    # Add one dimension to image as batch size 1
    image_input = image_tensor.unsqueeze(0)
    
    device = device_mode(gpu)
    image_input = image_input.to(device)
    model.to(device)
    
    # Make sure network is in eval mode for inference
    model.eval()
    # Predict
    output = model.forward(image_input)
    
    # Convert LogSoftmax values to probabilities
    ps = torch.exp(output)
    
    # Returns top n largest elements
    top_ps, top_idx = ps.topk(top_n)
    
    # Convert to Lists
    if gpu == True:
        top_ps = top_ps.cpu()
        top_idx = top_idx.cpu()
    top_ps = top_ps.detach().numpy()[0].tolist() 
    top_idx = top_idx.detach().numpy()[0].tolist() 
    
    # Invert class_to_idx
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    # Convert indices to classes
    top_classes = [idx_to_class[idx] for idx in top_idx]
    
    return top_ps, top_classes


def main():
	# Parse Inputs
	inputs = parse_inputs()


	# Load Model
	model = load_checkpoint(inputs.checkpoint)


	# Make Prediction
	probs, classes = predict(inputs.input, model, inputs.gpu, inputs.top_k)


	# Convert Category to Name
	if len(inputs.category_names) > 0:
		# Load mapping file for categories' real names
		with open(inputs.category_names, 'r') as f:
		    cat_to_name = json.load(f)

		# Convert classes to names
		classes = [cat_to_name[c] for c in classes]


	# Print the results
	results = list(zip(classes, probs))
	for i in range(len(results)):
		print('{}: {:.3%}'.format(results[i][0], results[i][1]))

	pass


if __name__ == '__main__':
	main()