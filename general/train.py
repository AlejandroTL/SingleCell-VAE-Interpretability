# def libraries
import torch.optim as optim
import torch.utils.data
from datetime import datetime
import matplotlib.pyplot as plt
from .data_preprocessing import *
from .models import *
import numpy as np


# Setup Learning Rate, optimizer, criterion... of the model
def setup(model, learning_rate=0.0001, option='mse'):
    if option == 'bce':
        criterion = nn.BCELoss(reduction='sum')
    else:
        criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return optimizer, criterion, device


# Final loss function with beta as hyperparameters
def final_loss(mu, logvar, reconstruction_loss, beta, c):
    kl_divergence = 0.5 * torch.sum(torch.exp(logvar) - logvar - 1 + torch.pow(mu, 2))
    channel_loss = torch.abs(kl_divergence - c)
    reconstruction = reconstruction_loss

    return beta * channel_loss + reconstruction, kl_divergence


# Fitting function
def fit(model, dataloader, beta, c, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    rec_loss = 0.0
    kl_loss = 0.0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar, _ = model(data)
        reconstruction_loss = criterion(reconstruction, data)
        loss, kl = final_loss(mu, logvar, reconstruction_loss, beta, c)
        rec_loss += reconstruction_loss.item()
        kl_loss += kl.item()
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    train_loss = running_loss/len(dataloader.dataset)
    kl_div_loss = kl_loss/len(dataloader.dataset)
    reconstructed_loss = rec_loss/len(dataloader.dataset)
    return train_loss, kl_div_loss, reconstructed_loss


# Validation over test dataset
def validate(model, dataloader, beta, c, criterion, device):
    model.eval()  # network in evaluation mode
    running_loss = 0.0
    rec_loss = 0.0
    kl_loss = 0.0
    with torch.no_grad():  # in validation we don't want to update weights
        for data in dataloader:
            data = data.to(device)
            reconstruction, mu, logvar, _ = model(data)
            reconstruction_loss = criterion(reconstruction, data)
            loss, kl = final_loss(mu, logvar, reconstruction_loss, beta, c)
            rec_loss += reconstruction_loss.item()
            kl_loss += kl.item()
            running_loss += loss.item()
            
    val_loss = running_loss / len(dataloader.dataset)
    kl_div_loss = kl_loss / len(dataloader.dataset)
    reconstructed_loss = rec_loss / len(dataloader.dataset)
    return val_loss, kl_div_loss, reconstructed_loss


def data2tensor(train_data, test_data, batch_size):

    train_data = pd.DataFrame(train_data)
    train_dataset = torch.tensor(train_data.values).float()

    test_data = pd.DataFrame(test_data)
    test_dataset = torch.tensor(test_data.values).float()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, test_loader


def loss_plots(train_loss, test_loss, kl_loss_train, kl_loss_test, rec_loss_train, rec_loss_test, dt_string):

    f = plt.figure(1)
    plt.title('Train Loss vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.linspace(1, len(train_loss), len(train_loss)), train_loss)
    plt.plot(np.linspace(1, len(test_loss), len(test_loss)), test_loss)
    f.savefig(f"TrainingReports/Plots/Train_VS_Test_{dt_string}.png")

    g = plt.figure(2)
    plt.title('Rec Train Loss vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.linspace(1, len(rec_loss_train), len(rec_loss_train)), rec_loss_train)
    plt.plot(np.linspace(1, len(rec_loss_test), len(rec_loss_test)), rec_loss_test)
    g.savefig(f"TrainingReports/Plots/Rec_Train_VS_Test_{dt_string}.png")

    h = plt.figure(3)
    plt.title('KL Train Loss vs Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.linspace(1, len(kl_loss_train), len(kl_loss_train)), kl_loss_train)
    plt.plot(np.linspace(1, len(kl_loss_test), len(kl_loss_test)), kl_loss_test)
    h.savefig(f"TrainingReports/Plots/KL_Train_VS_Test_{dt_string}.png")


def reporting(date, epoch, epochs, train_loss, test_loss, kl_train, kl_test, rec_train, rec_test, beta, c, c_launcher):

    f = open(f"TrainingReports/TrainingReport{date}_beta_{beta}_channel_{c}.txt", "a")
    f.write(f"\n Epoch {epoch + 1} of {epochs} \n")
    f.write("------------\n")
    f.write(f"Train Loss: {train_loss:.4f}\n")
    f.write(f"Train KL Loss: {kl_train:.4f}\n")
    f.write(f"Train Rec Loss: {rec_train:.4f}\n")
    f.write("------------\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test KL Loss: {kl_test:.4f}\n")
    f.write(f"Test Rec Loss: {rec_test:.4f}\n")
    f.write("------------\n")
    f.write(f"C parameter: {c_launcher: 4f}\n")
    f.write("------------\n\n")
    f.close()


# Callback for parameter C
def c_callback(epoch, total_epochs, final_c):

    c = epoch * (final_c / (0.75*total_epochs))  # 75% of the epochs with full C
    return min(final_c, c)


# Defined cyclical training
def training(mid_dim, features, dropout, data_path, batch_size, scaling, epochs=110, beta=1.00, c=0.00,
             learning_rate=0.0001):

    train_loss = []
    test_loss = []
    kl_loss_train = []
    kl_loss_test = []
    rec_loss_train = []
    rec_loss_test = []

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%H_%M")

    # Data
    train_data, test_data = data_preprocessing(data_path, scaling)
    loader_train, loader_test = data2tensor(train_data, test_data, batch_size)

    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=2000, mid_dim=mid_dim, features=features, drop=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum')

    print(device)
    print(torch.cuda.is_available())

    # Training
    print("Starting Training ")
    for epoch in range(epochs):

        c_launcher = c_callback(epoch, epochs, c)
        print(c, c_launcher)
        train_epoch_loss, kl_train_loss, rec_train_loss = fit(model, loader_train, beta, c_launcher,
                                                              optimizer, criterion, device)
        test_epoch_loss, kl_test_loss, rec_test_loss = validate(model, loader_test, beta, c_launcher,
                                                                criterion, device)
        train_loss.append(train_epoch_loss)
        test_loss.append(test_epoch_loss)
        kl_loss_train.append(kl_train_loss)
        kl_loss_test.append(kl_test_loss)
        rec_loss_train.append(rec_train_loss)
        rec_loss_test.append(rec_test_loss)
        reporting(dt_string, epoch, epochs, train_epoch_loss, test_epoch_loss, kl_train_loss, kl_test_loss,
                  rec_train_loss, rec_test_loss, beta, c, c_launcher)

    # Save the model trained

    path = f'trained_models/model{dt_string}_beta_{beta}_channel_{c}.pth'
    torch.save(model.state_dict(), path)

    return train_loss, test_loss, kl_loss_train, kl_loss_test, rec_loss_train, rec_loss_test, dt_string, device,

