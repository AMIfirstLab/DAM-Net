from model import DAM_NET
from data_loaders import DataPrep, ValDataPrep
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from utils import scores


device = torch.device('cuda:0')

model = DAM_NET(3, 3).to(device)

epochs = 50
base_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = base_lr, weight_decay = 0.0001)


loss_function = nn.CrossEntropyLoss(label_smoothing = 0.2)

data_path = 'E:\\Research\\covid_research\\total_covid_data_3classes.npy'
training_set = DataPrep(data_path)
validation_set = ValDataPrep(data_path)

train_loader = DataLoader(training_set, batch_size=12, shuffle=True, pin_memory=True)
val_loader = DataLoader(validation_set, batch_size=12, shuffle=True, pin_memory=True)

for epoch in range(epochs):
    train_accuracy = []
    train_sensitivity = []
    train_specivity = []
    train_precision = []
    
    train_accuracy_1 = []
    train_precision_1 = []
    train_sensitivity_1 = []
    train_specificity_1 = []
    
    train_accuracy_2 = []
    train_precision_2 = []
    train_sensitivity_2 = []
    train_specificity_2 = []
    
    for sample in train_loader:
        image, label = sample
        image, label = image.to(device), label.to(device)
        label = torch.squeeze(label, dim = 1)
        
        output = model(image.float())

        label = torch.squeeze(label, dim = 1)
        loss = loss_function(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output = torch.argmax(output, dim = 1)
        label = torch.argmax(label, dim = 1) 

        label = label.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        
        cm = confusion_matrix(label, output)
        accuracy_1, precision_1, sensitivity_1, specificity_1, accuracy_2, precision_2, sensitivity_2, specificity_2 = scores(cm)
        train_accuracy_1.append(accuracy_1)
        train_precision_1.append(precision_1)
        train_sensitivity_1.append(sensitivity_1)
        train_specificity_1.append(specificity_1)
        
        train_accuracy_2.append(accuracy_2)
        train_precision_2.append(precision_2)
        train_sensitivity_2.append(sensitivity_2)
        train_specificity_2.append(specificity_2)
        
    
    all_acc = (np.mean(train_accuracy_1) + np.mean(train_accuracy_2)) / 2
    all_prec = (np.mean(train_precision_1) + np.mean(train_precision_2)) / 2
    all_sens = (np.mean(train_sensitivity_1) + np.mean(train_sensitivity_2)) / 2
    all_spec = (np.mean(train_specificity_1) + np.mean(train_specificity_2)) / 2
    
    print('Training Accuracy: ', all_acc)
    print('Training Precision: ', all_prec)
    print('Training Sensitivity: ', all_sens)
    print('Training Specivity: ', all_spec)
    
    
    val_accuracy = []
    val_sensitivity = []
    val_specivity = []
    val_precision = []
    
    val_accuracy_1 = []
    val_precision_1 = []
    val_sensitivity_1 = []
    val_specificity_1 = []
    
    val_accuracy_2 = []
    val_precision_2 = []
    val_sensitivity_2 = []
    val_specificity_2 = []
    
    for sample in val_loader:
        image, label = sample
        image, label = image.to(device), label.to(device)

        output = model(image.float())

        label = torch.squeeze(label, dim = 1)
        loss = loss_function(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output = torch.argmax(output, dim = 1)
        label = torch.argmax(label, dim = 1)
        
        label = label.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
        
        cm = confusion_matrix(label, output)
        accuracy_1, precision_1, sensitivity_1, specificity_1, accuracy_2, precision_2, sensitivity_2, specificity_2 = scores(cm)
        val_accuracy_1.append(accuracy_1)
        val_precision_1.append(precision_1)
        val_sensitivity_1.append(sensitivity_1)
        val_specificity_1.append(specificity_1)
        
        val_accuracy_2.append(accuracy_2)
        val_precision_2.append(precision_2)
        val_sensitivity_2.append(sensitivity_2)
        val_specificity_2.append(specificity_2)

        
    all_acc = (np.mean(val_accuracy_1) + np.mean(val_accuracy_2)) / 2
    all_prec = (np.mean(val_precision_1) + np.mean(val_precision_2)) / 2
    all_sens = (np.mean(val_sensitivity_1) + np.mean(val_sensitivity_2)) / 2
    all_spec = (np.mean(val_specificity_1) + np.mean(val_specificity_2)) / 2
    
    print('Validation Accuracy: ', all_acc)
    print('Validation Precision: ', all_prec)
    print('Validation Sensitivity: ', all_sens)
    print('Validation Specivity: ', all_spec)
    
    torch.save(model.state_dict(), 'E:\\Research\\covid_research\\saved_models\\binary_only\\dam_net{}.pth'.format(epoch+1))
