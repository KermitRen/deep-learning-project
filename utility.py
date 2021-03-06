from numpy import NaN
import torch
import visualizing as Vis
import matplotlib.pyplot as plt

def fit_AutoEncoder(data_loader, model, loss_func, optimizer, epochs, val_loader, graph_loss = False):

    #Setup showcase batch
    batch, target_batch = next(iter(val_loader))
    batch, target_batch = batch[0:5], target_batch[0:5]
    history = torch.cat((target_batch.cpu(), batch.cpu()),0)

    #Loss Variables
    train_losses = []
    val_losses = [NaN]

    for epoch in range(epochs):

        #Training Parameters
        model.train()
        train_loss = 0
        for img, target_img in data_loader:
            loss = loss_func(model(img), target_img)
            train_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #Evaluating
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for img, target_img in val_loader:
                val_loss += loss_func(model(img), target_img)

        #Print Progress
        avgTrainLoss = (train_loss.item()/(len(data_loader.dataset)))*100
        avgValLoss = (val_loss.item()/(len(val_loader.dataset)))*100
        train_losses.append(avgTrainLoss)
        val_losses.append(avgValLoss)
        print("Completed epoch " + str(epoch + 1) + " out of " + str(epochs))
        print("The new model has training loss: " + "{:.6f}".format(avgTrainLoss))
        print("The new model has validation loss: " + "{:.6f}".format(avgValLoss) + "\n")
        history = torch.cat(((history,) + (model(batch).cpu(),)), 0)
    
    #Show loss graph
    if(graph_loss):
        graphLossHistory(train_losses, val_losses)

    return history

def fit_Classifier(data_loader, model, loss_func, optimizer, epochs, val_loader, graph_loss = False):

    #Loss Variables
    train_losses = []
    val_losses = [NaN]

    for epoch in range(epochs):

        #Training Parameters
        model.train()
        train_loss = 0
        for img, label in data_loader:
            loss = loss_func(model(img), label) 
            train_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #Evaluating
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for img, label in val_loader:
                val_loss += loss_func(model(img), label)

        #Print Progress
        avgTrainLoss = (train_loss.item()/(len(data_loader.dataset)))*100
        avgValLoss = (val_loss.item()/(len(val_loader.dataset)))*100
        train_losses.append(avgTrainLoss)
        val_losses.append(avgValLoss)
        print("Completed epoch " + str(epoch + 1) + " out of " + str(epochs))
        print("The new model has training loss: " + "{:.6f}".format(avgTrainLoss))
        print("The new model has validation loss: " + "{:.6f}".format(avgValLoss) + "\n")
    
    #Show loss graph
    if(graph_loss):
        graphLossHistory(train_losses, val_losses)


def fit_DeepDream(model, loss_func, optimizer, epochs, history_size = 5):

    ending_encoding = model.getEndingEncoding()
    loss_history = []
    weight_history = []

    for epoch in range(epochs):

        #Training Parameters
        model.train()
        loss = loss_func(model(), ending_encoding)
        loss_history.append(loss.item())
        weight_history.append(model.getWeights())
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        #Evaluating
        if(epoch%10 == 0):
            print("Completed epoch " + str(epoch + 1) + " out of " + str(epochs) + " with loss " + str(loss) + "\n")
    
    history = createHistory(loss_history, weight_history, history_size)
    return history

def createHistory(loss_history, weight_history, history_size):
    #Needs to be done
    loss_diff = loss_history[0]-loss_history[len(loss_history)-1]
    loss_interval = loss_diff/(history_size-1)
    history = weight_history[0]
    for i in range(history_size-1):
        target_loss = loss_history[0]-((i+1)*loss_interval)
        smallest_diff = float('inf')
        best_epoch = 0
        for j in range(len(loss_history)):
            targetloss_diff = abs(loss_history[j]-target_loss)
            if targetloss_diff < smallest_diff:
                smallest_diff = targetloss_diff
                best_epoch = j
        history = torch.cat(((history, )+(weight_history[best_epoch],)),0)
    return history

def graphLossHistory(train_hist, val_hist):
    plt.figure(figsize=(20,6))
    plt.plot(train_hist)
    plt.plot(val_hist)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def getInterpolations(img1, img2, size = 5):

    hybrids = []
    interpolationFactor = 1/(size - 1)

    for i in range(size - 2):
        hybrids.append(torch.lerp(img1, img2, (i + 1) * interpolationFactor))

    interpolationTuple = (img1,) + tuple(hybrids) + (img2,)
    return torch.cat(interpolationTuple,0)

def interpolationMerge(img1, img2, model, size = 5):
    encoding1 = model.encode(img1)
    encoding2 = model.encode(img2)
    interpolatedEncodings = getInterpolations(encoding1, encoding2, size)
    interpolatedDecodings = model.decode(interpolatedEncodings)
    #Vis.displayImages(interpolatedDecodings)
    return interpolatedDecodings

def save_model_state(model, opt, path):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            }, path)

def load_model_state(model, opt, path):
    data = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(data['model_state_dict'])
    opt.load_state_dict(data['optimizer_state_dict'])

def labelMapping(animal):
    return {
        'cat': 0,
        'dog': 1,
        'fox': 2,
        'lion': 3,
        'wolf': 4,
        'tiger': 5,
        'leopard': 6,
        'cheetah': 7,
    }[animal]