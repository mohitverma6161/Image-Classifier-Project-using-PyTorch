import argparse
import json
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch
from PIL import Image
import json
import math
def argument_parser():
    
    parser=argparse.ArgumentParser(description="Predicting")
    
    parser.add_argument('--topk',type=int,help='describe how many top k classes you want')
    parser.add_argument('--image_jpg',type=str,help='image for predicting the most likely image class')
    parser.add_argument('--device',type=str,help='use for calculations')
    parser.add_argument('--names',type=str,help='To Load a json file')
    args=parser.parse_args()
    return args

def load_checkpoint():
    
    
    checkpoint = torch.load("checkpoint.pth")
    structure=checkpoint['architecture']
    
    
    if structure == 'vgg13':
        model = models.vgg13(pretrained=True)
        model.name = "vgg13"
    else:
        a={}
        exec("model = models.{}(pretrained=True)".format(structure),globals(),a)
        model=a["model"]
        model.name=structure
    
    
    
    model.state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier'] 
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
#model = load_checkpoint('checkpoint.pth')
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    Image_for_test=Image.open(image)
    rearrangement=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    Image_final=rearrangement(Image_for_test)
    
    return Image_final

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk,device,cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.to("cpu")
    model.eval()
    #model=load_checkpoint().cpu()
    image_processed = process_image(image_path)
    
    
    image_processed_tensor = torch.from_numpy(np.expand_dims(image_processed, 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")
    #image_processed_tensor=image_processed_tensor
    #with torch.no_grad():
    output = model.forward(image_processed_tensor)
    #model.eval()
    linear=torch.exp(output)
    
    top_probabilities,top_index_list=linear.topk(topk)
    
    
    top_probabilities_list = np.array(top_probabilities.detach())[0]
    top_index_list = np.array(top_index_list.detach())[0]
    
    index = {x: y for y, x in model.class_to_idx.items()}
    top_index_list = [index[z] for z in top_index_list]
    top_flowers_list = [cat_to_name[z] for z in top_index_list]
    
    return top_probabilities, top_index_list, top_flowers_list

def main():
    args=argument_parser()
    
    if type(args.names)==type(None):
        names='cat_to_name.json'
    else:
        names=args.names
    import json
    with open(names, 'r') as f:
        cat_to_name = json.load(f)
    
    
    
    final_model=load_checkpoint()
    
    image=args.image_jpg
    if type(args.device)==type(None):
        device='cuda'
    else:
        device=args.device
    device=args.device
    topk=args.topk
    a,b,c=predict(image, final_model, topk,device,cat_to_name)
    print("Input label  = {}".format(b))
    print("Probabilities = {}".format(a))
    
if __name__ == '__main__': main()    