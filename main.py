import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
torch.cuda.set_device(8)
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from prepare_dataset import *
from training_config import *
from model import *



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Speech Command Classification')

    # model hyper-parameter variables
    parser.add_argument('--lr', default=0.001, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=32, metavar='batch_size', type=float, help='batch_size')
    parser.add_argument('--itr', default=100, metavar='itr', type=int, help='Number of iterations')
    
    args = parser.parse_args()

    
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size         
    N_EPOCHS = args.itr           
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
x, y = load_data('/home/aminul/data/Speech_data/train/audio')
train_loader, val_loader = get_dataloader(x,y,batch_size=BATCH_SIZE)

net = NN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=LEARNING_RATE)


start_epoch=0
num_epochs=N_EPOCHS

for epoch in range(start_epoch, start_epoch+num_epochs):
    
    train(net,train_loader,optimizer,criterion,epoch,device)
    best_acc = test(net,val_loader,optimizer,criterion,epoch,device)