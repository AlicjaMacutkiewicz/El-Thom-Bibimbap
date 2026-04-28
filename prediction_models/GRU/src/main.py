import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from GRU_model import GRU

# Check for current best hardware options: 
#   mps - apple gpu, 
#   cuda - nvidia gpu, 
#   xpu - intel gpu
#   cpu - other/cpu options

def get_best_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        try:
            import intel_extension_for_pytorch
            if torch.xpu.is_available():
                dev = torch.device("xpu")
            else:
                dev = torch.device("cpu")
        except ImportError:
            dev = torch.device("cpu")
    
    print(f"running on {dev}")
    return dev

# Load data from num_of_flights parquet files into a single numpy array
# containing X, Y, Z acceleration data

# Structure of file paths from which the function reads:
# base_path - directory containing the flight parquet files + base file name
# flight_start_flight - a number added at the end of base_path to identify a single flight
# .parquet - all the data is read from parquet files

def read_flight_data(start_flight, num_of_flights, base_path = "../../../demo_flight_data/flight_"):
    flights = [] 
    
    for flight_number in range(start_flight, start_flight + num_of_flights):
        file_path = base_path + str(flight_number) + ".parquet"
        flight_data = pd.read_parquet(file_path) # read parquet file into a dataframe
        # extract only columns with acceleration data
        acc_data = flight_data[["Acceleration_X", "Acceleration_Y", "Acceleration_Z"]].values
        
        flights.append(acc_data)

    # "flights" is now a list of numpy arrays where each element contains
    # acceleration data (from X, Y, Z axis) from one flight
    # flights shape: (timesteps, 3)
    return flights


# Split list of flights into training and testing sets for the model
# where split_ratio is the fraction of flights used for training
def split_flights(flights, split_ratio=0.8):
    split_idx = int(len(flights) * split_ratio)

    train_flights = flights[:split_idx]
    test_flights  = flights[split_idx:]

    return train_flights, test_flights

# Apply normalization to each flight independently using only the statistical
# values (mean, std) for the training dataset passed as a 1D array 
def normalize_flights(flights, array):
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    return [(flight - mean) / std for flight in flights]

# Convert flight time-series data into shorter learning 
# samples for the GRU model
# Each sample consists of:

# X: past seq_len time steps - model input sequence
# y: next pred_len time steps - future values 
# that the model is trained to predict

# X shape: (num_samples, seq_len, 3)
# y shape: (num_samples, pred_len, 3)
def make_sequences(flights, seq_len, pred_len):
    X, y = [], []

    # for every flight take a sequence of seq_len next time steps
    # so that the model can predcit pred_len values
    for flight in flights:
        for i in range(len(flight) - seq_len - pred_len):
            # take seq_len values from past observations
            X.append(flight[i:i+seq_len])
            # take pred_len future values to be predicted
            y.append(flight[i+seq_len:i+seq_len+pred_len])
    
    # convert to numpy arrays bcs apparently creating a tensor from 
    # a normal list of numpy arrays is slow af
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # convert to tensors (required for model training)
    return torch.from_numpy(X), torch.from_numpy(y)


# Train the GRU model using small-batch gradient descent
# one training_round = one full pass over dataset

def train_model(model, X_train, y_train, X_test, y_test, loss, optimizer,
                pred_len, batch_size=64, training_rounds=10):

    train_losses = []
    test_losses = []
    print("started training")

    for training_round in range(training_rounds):
        round_loss = 0.0 # cumulated prediction error for the whole round
        total_samples = 0
        
        model.train() # set the mode to train (some layers behave
        # differently during training and evaluation)

        # iterate over the training dataset in smaller batches
        for i in range(0, len(X_train), batch_size):
            # take a slice of successive input sequences
            # starting from the current time stamp (i)

            # X_batch: (batch_size, seq_len, 3)
            # where 3 = [Acc_x, Acc_y, Acc_z]
            X_batch = X_train[i:i+batch_size].to(device) # batch input
            
            # y_batch shape: (batch_size, pred_len, 3)
            y_batch = y_train[i:i+batch_size].to(device) # correct future values to be predicted (targets)

            # pass input sequence through GRU to get predictions for each time step
            # outputs shape: (batch_size, seq_len, 3)
            outputs, _ = model(X_batch)
            preds = outputs[:, -pred_len:, :] # take only the part of the sequence
            # that was predicted in the current iteration (so the last pred_len values)

            # calculate loss between predictions and targets
            batch_loss = loss(preds, y_batch)

            optimizer.zero_grad()  # reset gradients from previous step
            
            # calculate how should weights change to reduce loss (backpropagation)
            batch_loss.backward()
            # update weights using calculated gradients
            optimizer.step()

            round_loss += batch_loss.item() # accumulate loss for this round
            total_samples += len(X_batch)

        # calcutate average loss over all training batches in this round
        avg_loss = round_loss / total_samples
        
        train_losses.append(avg_loss)

        # calcutate average loss over all test batches in this round
        avg_test_loss = evaluate_model(model, X_test, y_test, loss, batch_size, pred_len)
        test_losses.append(avg_test_loss)
        
        print(f"Iteration {training_round+1}, train loss: {avg_loss:.5f}, test loss: {avg_test_loss:.5f}")

    return train_losses, test_losses

# Evaluate model using test data
def evaluate_model(model, X_test, y_test, loss, batch_size=64, pred_len=3):

    model.eval() # set the mode to evaluation mode
    test_loss = 0.0
    with torch.no_grad():
        # iterate over the test dataset in smaller batches    
        for i in range(0, len(X_test), batch_size):
            # select one batch of test inputs
            X_batch = X_test[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device)

            # pass input sequence through GRU to get predictions for each time step
            outputs, _ = model(X_batch)
            # take only the last pred_len timesteps from the output sequence
            preds = outputs[:, -pred_len:, :]
            # compute loss between predictions and true values
            curr_loss = loss(preds, y_batch)

            # calculate total loss value
            test_loss += curr_loss.item()

    # return average loss over all test batches    
    return test_loss / (len(X_test) / batch_size)


def plot_prediction(model, X_test, y_test, pred_len, sample_idx=0, axis=0):   
    model.eval() # set the mode to evaluation mode
    
    with torch.no_grad():
        # X_test shape: (num_samples, seq_len, 3)
        # y_test shape: (num_samples, pred_len, 3)
    
        # select a single test sample (batch size = 1)
        # input_seq shape = (1, seq_len, 3)
        input_seq = X_test[sample_idx:sample_idx+1].to(device)

        # select corresponding correct future values to be predicted (targets)
        # target shape: (pred_len, 3)
        target = y_test[sample_idx]

        # pass the input sequence through the GRU model
        # output shape: (1, seq_len, 3)
        # hidden state (_) is ignored
        output, _ = model(input_seq)

        # extract only the last pred_len time steps from the sequence
        # prediction shape: (pred_len, 3)
        prediction = output[0, -pred_len:, :].cpu().numpy()    

        # define time axes for past (input) and future (prediction)
        seq_len =  input_seq.shape[1] # length of input sequence

        # past_time: numpy array [0, 1, ..., seq_len-1]
        # represents time indices of the input sequence
        past_time = np.arange(seq_len)
        # future_time: numpy array [seq_len, ..., seq_len+pred_len-1]
        # Represents time indices of the future (prediction)
        future_time = np.arange(seq_len, seq_len + pred_len)

        plt.figure(figsize=(10, 5))
        axes_labels = ["X", "Y", "Z"]
        # plot historical data used as input
        plt.plot(past_time, X_test[sample_idx, :, axis], label="Historia (Input)", color='blue', marker='o')
        # plot the actual future values
        plt.plot(future_time, target[:, axis], label="Prawda (Target)", color='green', marker='s')
        # plot the values predicted by the GRU model
        plt.plot(future_time, prediction[:, axis], label="Predykcja", color='red', linestyle='--', marker='x')
    
        # add a vertical line seperating past and future
        plt.axvline(x=seq_len-0.5, color='gray', linestyle='--')
    
        plt.title(f"Predykcja Przyspieszenia {axes_labels[axis]} (Próbka {sample_idx})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(test_losses, label='Testing Loss', color='orange')
    plt.xlabel("Round")
    plt.ylabel("Loss (MSE)")
    plt.title("Model Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


batch_size = 64
training_rounds = 10 # iterations over the entire dataset during training
seq_len = 40  # length of the input time sequence (how many past steps the model sees)
pred_len = seq_len  # number of future time steps the model predicts
loss = nn.MSELoss() # loss function used for training the model

flights = read_flight_data(10, 20)
print("data loaded")

train_flights, test_flights = split_flights(flights)

# flatten all training flights into a single array to calculate statistical values
train_array = np.concatenate(train_flights, axis=0)

train_flights = normalize_flights(train_flights, train_array)
test_flights  = normalize_flights(test_flights, train_array)

X_train, y_train = make_sequences(train_flights, seq_len, pred_len)
X_test, y_test = make_sequences(test_flights, seq_len, pred_len)

print("data preprocessing done")
device = get_best_device()
model = GRU(input_size=3, hidden_size=64, output_size=3,  num_layers=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

#adjusted the arguments passed to the function to process them properly
train_losses, test_losses = train_model(model, X_train, y_train, X_test, y_test, loss, optimizer,
                           batch_size = batch_size, training_rounds = training_rounds, pred_len = pred_len)

plot_losses(train_losses, test_losses)

plot_prediction(model, X_test, y_test, pred_len, sample_idx=np.random.randint(0, len(X_test)))
