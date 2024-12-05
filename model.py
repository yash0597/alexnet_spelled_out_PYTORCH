
"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""


import os
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from tensorboardX import SummaryWriter 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9 
LR_DECAY = 0.0095
LR_INIT = 0.01
IMAGE_DIM = 227 
NUM_CLASSES = 1000 
DEVICE_IDS = [0,1,2,3]
# modify this to point to your data directory
INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

Class AlexNet(nn.Module):
     """
    Neural network model consisting of layers proposed by AlexNet paper.
    """
    
    def __init__(self , num_classes = 10000):
        
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = 3 , out_channels = 96 , kernel_size = 11 , stride = 4 ),
            nn.ReLU(),
            nn.LocalResponseNorm(size =5 , alpha = 0.0001 , beta = 0.75 , k =2 )
            nn.MaxPool2d(kernel_size = 3 , stride=2)
            nn.Conv2d(96 , 256 , 5 , padding =2) , 
            nn.ReLU(),
            nn.LocalResponseNorm(size = 5 , alpha = 0.0001 , beta = 0.75 , k =2),
            nn.MaxPool2d(kernel_size = 3 , stride =2),
            nn.Conv2d(256 , 384 , 3, padding = 1),
            nn.ReLU(),
            nn.Con2d(384 , 384 , 3 , padding = 1),
            nn.ReLU(),
            nn.Conv2d(384 , 256 , 3 , padding = 1)
            nn.ReLu(),
            nn.MaxPool(kernel_size = 3 , stride = 2 )
                  
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5 , inplace = True) , 
            nn.Linear(in_features=(256 * 6*6) , out_features = 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5 , inplace = True) , 
            nn.Linear(in_features = 4096 , out_features = 4096),
            nn.ReLU(),
            nn.Linear(in_features = 4096 , out_features = num_classes)
        )
        
        
        self.init_bias()
        
        
    def init_bias(self):
        for layer in self.net:
            if isinstance(layer , nn.Conv2d):
                nn.init.nomral_(layer.weight , mean = 0 , std = 0.01)
                nn.init.constant_(layer.bias ,0)
                
            nn.init.constant_(self.net[4].bias , 1)
            nn.init.constant_(self.net[10].bias , 1)
            nn.init.constant_(self.net[12].bias , 1)
            
    def forward(self , x):
         x = self.net(x)
         x =x.view(-1 , 256 * 6 *6 )
         return self.classifier(x)
     
     

if __name__ == "__main__":
    
    seed = torch,initial_seed()
    
    print('Used seed : {}'.format(seed))
    
    tbwriter = SummaryWriter(log_dir = LOG_DIR)
    print("TensorBoardX summary writer created ")
    
    alexnet = AlexNet(num_classes = NUM_CLASSES).to(device)
    
    #Enabling multi gpu training process 
    
    alexnet = torch.nn.parallel.DataParallel(alexnet , device_ids = DEVICE_IDS)
    
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR , transforms.Compose([
        transforms.CenterCrop(IMAGE_DIM) , 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        
    ]))
    print("Datset created")
    
    dataloader = data.DataLoader(
      dataset,
      shuffle=True,
      pin_memory=True,
      num_workers=8,
      drop_last=True,
      batch_size=BATCH_SIZE)
    
    print("Dataloader created")
    
    
    optimizer = optim.Adam(params = alexnet.parameters() , lr = 0.0001)
    
    ### BELOW is the setting proposed by the original paper - which doesn't train....
    # optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=LR_INIT,
    #     momentum=MOMENTUM,
    #     weight_decay=LR_DECAY)
    
    print("optimizer created")

    lr_scheduler = optim.lr_scheduler.StepLR(optimzer , step_s9ze = 30 , gamma = 0.1)
    print('LR SCHEDULER CREATED')
    
    #START TRAINING NOW 
    
    print("Start training.... ")
    
    total_steps = 1
 
    for epoch in range(NUM_EPOCHS):
        lr_scheduler.step()
        
        for imgs , classes in dataloader:
            
            imgs , classes = imgs.to(device) , classes.to(device)
            
            
            output = alexnet(imgs)
            
            loss = F.cross_entropy(output , classes )
            
            optimizer.zero_grad()
            loss.bakcward()
            optimzer.step()
            
            
            
            #log the information and add to tensorbaord 
            
           if total_steps % 10 == 0:
                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    accuracy = torch.sum(preds == classes)

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                        .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)

            # print out gradient values and parameter average values
            if total_steps % 100 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter values
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                    parameter.grad.cpu().numpy(), total_steps)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                    parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

             total_steps += 1
             
             
         #Saves checkpoints 
         checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
         state = {
            'epoch': epoch,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }
        torch.save(state, checkpoint_path)
            
        
 
             

        
    
    


    
    
    
    




