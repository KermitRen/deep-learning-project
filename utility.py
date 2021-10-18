import torch

def fit_AutoEncoder(data_loader, model, loss_func, optimizer, epochs, val_loader):

    history = []
    display_batch = next(iter(val_loader))[0:5]
    history.append(display_batch.cpu())

    for epoch in range(epochs):

        #Training Parameters
        model.train()
        for batch in data_loader:
            loss = loss_func(model(batch), batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #Evaluating
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                val_loss += loss_func(model(batch), batch)
            print("Completed epoch " + str(epoch + 1) + " out of " + str(epochs))
            print("The new model has validation loss: " + str(val_loss))
            history.append(model(display_batch).cpu())
    
    return history

def fit_DeepDream(model, loss_func, optimizer, epochs):

    ending_encoding = model.getEndingEncoding()
    for epoch in range(epochs):

        #Training Parameters
        model.train()
        loss = loss_func(model(), ending_encoding)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        #Evaluating
        print("Completed epoch " + str(epoch + 1) + " out of " + str(epochs))

def save_model_state(model, opt, path):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            }, path)

def load_model_state(model, opt, path):
    data = torch.load(path)
    model.load_state_dict(data['model_state_dict'])
    opt.load_state_dict(data['optimizer_state_dict'])
    return model, opt