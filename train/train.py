import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score as accuracy_function
import yaml
import sys
sys.path.insert(0, '../model/')
from model import CustomModel
from dataset import TrainTestDataset

def loadDataset(batch_size, dataset_path):
    # -- loads the training and testing dataset from the dataset.py
    # -- returns: a train data loader and test data loader
    dataset = TrainTestDataset(path)

    train_dl = DataLoader(dataset=dataset.train_dataloader, batch_size=batch_size, 
                          shuffle=True, pin_memory=True, num_workers=16, drop_last=True)
    test_dl = DataLoader(dataset=dataset.test_dataloader, batch_size=batch_size, 
                          shuffle=True, pin_memory=True, num_workers=16, drop_last=True)

    return train_dl, test_dl

def validation(model, dataset, epoch, global_step, writer):
    
    model.eval() # -- set model to evaluation mode (makes weights updatable)

    for input, target in loader: # -- iterates the dataset in batches

        input = input.to(device) # -- sets tensor to cpu or gpu
        target = target.to(device) # -- sets tensor to cpu or gpu
        
        # -- predict output
        prediction = model(input)

        # -- calculate loss, this is done for evaluation purposes
        loss = loss_function(prediction, target)

        # -- check accuracy
        acc = accuracy_function(prediction.clone().detach(), target.clone().detach())

        # -- plot to tensorboard
        writer.add('In-validation loss', loss.item(), global_step=global_step)
        writer.add('Val_acc', acc, global_step=global_step)

        global_step += 1

        return writer, global_step

def train(model, dataset, epoch, writer):

    model.train() # -- set model to train mode (makes weights updatable)

    for input, target in dataset: # -- iterates the dataset in batches
        
        # -- set gradients to zero
        for param in model.parameters():
            param.grad=None

        input = input.to(device) # -- sets tensor to cpu or gpu
        target = target.to(device) # -- sets tensor to cpu or gpu
        
        # -- predict output
        prediction = model(input)

        # -- calculate loss
        loss = loss_function(prediction, target)

        #-- optimizer step
        optimizer.step()

        # -- check accuracy
        acc = accuracy_function(prediction.clone().detach(), target.clone().detach())

        # -- plot to tensorboard
        writer.add('In-training loss', loss.item(), global_step=global_step)
        writer.add('Train_acc', acc, global_step=global_step)

        global_step += 1

        return writer, global_step

def loadConfig(path='./train_config.yaml'):
    # -- loads the training configuration
    return yaml.safe_load(open(path,'r'))

def loadModel(model_param):
    # -- loads model according to parameter list in train_config.yaml
    p = list(model_param['MODEL_PARAM'].values())
    return CustomModel(p[1], p[2], p[3],p[4])

if __name__=='__main__':
    
    # -- choose gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model config
    train_config = loadConfig()
    # -- load dataset
    train_dataset, test_dataset = loadDataset(batch_size=train_config['MODEL_PARAM']['BATCH_SIZE'],
                                              dataset_path=train_config['MODEL_CFG']['DS_PATH'])
    # -- tensorboard output
    writer = SummaryWriter(train_config['MODEL_CFG']['TB_PATH'], comment=train_config['MODEL_CFG']['TB_COMMENT'])
    # -- load model according to model config
    model = loadModel(train_config['MODEL_PARAM']).to(device)
    # -- set loss function
    loss_function = nn.MSELoss().to(device())
    # -- set optimizer
    optimizer = optim.AdamW(model.parameters(), lr=train_config['TRAIN_PARAM']['LEARNING_RATE'],
                            weight_decay=train_config['TRAIN_PARAM']['WEIGHT_DECAY'])
    # -- set number of eepochs
    number_epochs = train_config['TRAIN_PARAM']['EPOCHS']
    
    # -- training loop
    for epoch in range(num_epochs):
        # -- training step
        writer, train_step = train(model, dataset, epoch, writer)
        
        # -- validation step
        writer, test_step = validation(model, dataset, epoch, writer)

        # -- checkpoint
        if epoch % 25 == 0:
            torch.save()

    writer.close()
