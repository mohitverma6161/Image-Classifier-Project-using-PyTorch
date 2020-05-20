import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import seaborn as sns
from PIL import Image
from collections import OrderedDict


#%config InlineBackend.figure_format = 'retina'

def argument_parser():
    parser = argparse.ArgumentParser(description='Train the model.')
    
    parser.add_argument('--arch',type=str,help='Choose the architecture')
    
    parser.add_argument('--epochs',type=int,help='Number of epochs for training purpose')
    parser.add_argument('--learning_rate',type=float,help='Learning rate')
    parser.add_argument('--hidden_units',type=int,help='Number of hidden units')
    parser.add_argument('--device',type=str)
    args=parser.parse_args()
    return args

def define_transforms(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'training_t': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'validation_t': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
            ]),
        'testing_t': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
} 

# TODO: Load the datasets with ImageFolder
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform = data_transforms['training_t']),
        'validation': datasets.ImageFolder(valid_dir, transform = data_transforms['validation_t']),
        'testing': datasets.ImageFolder(test_dir, transform = data_transforms['testing_t'])
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64)
}
    return data_transforms,image_datasets,dataloaders



def primaryloader_model(architecture="vgg13"):
    # Load Defaults if none specified
    if type(architecture) == type(None): 
        model = models.vgg13(pretrained=True)
        model.name = "vgg13"
        print("Network architecture specified as vgg13.")
    else: 
        a={}
        exec("model = models.{}(pretrained=True)".format(architecture),globals(),a)
        model=a["model"]
        print(model)
        model.name=architecture
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    return model
def classifier_initialise(model,hidden):
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier=classifier
    
    return classifier

def Validation(model, dataloaders, criterion):
    loss=0
    accuracy=0
    model.to('cuda')
    for images, labels in dataloaders['validation']:
        
        #images = Variable(images)
        #labels = Variable(labels)
        images, labels = images.to('cuda'), labels.to('cuda')
        
        output = model.forward(images)
        loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
    return loss, accuracy

def train_network(model,epochs,steps,print_every,dataloaders,optimizer,criterion,device):
    for e in range(epochs):
        #model.train()
        running_loss = 0
        model.to(device)
        for images, labels in dataloaders['training']:
            #images, labels = Variable(images), Variable(labels)
            steps += 1
            
            images = images.to(device) 
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                model.to(device)
                
                with torch.no_grad():
                    actual_loss, accuracy = Validation(model, dataloaders, criterion)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                     "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                     "Test Loss: {:.3f}.. ".format(actual_loss/len(dataloaders['validation'])),
                     "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['validation'])))
                
                running_loss = 0
                model.train()
                
    print("Training Process Completed!!! Now Proceed for Testing Process")
    return model

def check_accuracy(model,dataloaders):
    total=0
    c=0
    with torch.no_grad():
        for images,labels in dataloaders['testing']:
            images=images.to('cuda')
            labels=labels.to('cuda')
        
            output=model(images)
            _,pred=torch.max(output.data,1)
            total += labels.size(0)
            c=c+(pred==labels).sum().item()
    temp=c/total
    print('Accuracy is: %d%%' % (100 * temp))
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
   
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def save_checkpoint(model,image_datasets,optimizer,save_directory,epochs,struc):
    
    model.class_to_idx = image_datasets['training'].class_to_idx
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'opt_state': optimizer.state_dict,
                  'num_epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'architecture':struc
                  }
    torch.save(checkpoint, 'checkpoint.pth')


def main():
    data_dir = 'flowers'
    data_transforms,image_datasets,dataloaders=define_transforms(data_dir)
    args=argument_parser()
    
    struc=args.arch
    model = primaryloader_model(struc)
    hidden=args.hidden_units
    model.classifier=classifier_initialise(model,hidden)
    lr=args.learning_rate
    #args.learning_rate
    model.to('cuda')
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr)
    
    if type(args.epochs) == type(None):
        epoch = 6
        print("Epoch is set to 6")
    else: 
        epoch = args.epochs
        print(epoch)
    
    steps=0
    print_every=35
    
    
    if type(args.device)==type(None):
        device='cuda'
    else:
        device=args.device
    temp_model=train_network(model,epoch,steps,print_every,dataloaders,optimizer,criterion,device)
    check_accuracy(temp_model,dataloaders)
    
    save_checkpoint(temp_model,image_datasets,optimizer,"checkpoint.pth",epoch,struc)


if __name__ == '__main__': main()

    