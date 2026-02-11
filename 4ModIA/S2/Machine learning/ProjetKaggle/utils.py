import torch 
import tqdm 
from plotly.subplots import make_subplots
import plotly.graph_objects as go



def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm.tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc



def plot_training_metrics(train_accuracies, test_accuracies, train_losses, test_losses, epoch_times, num_epochs):
    fig_acc = make_subplots(rows=1, cols=1, subplot_titles=("Train and Test Accuracy",))
    trace_train_acc = go.Scatter(x=list(range(1, num_epochs + 1)), y=train_accuracies, mode='lines+markers', name='Train Accuracy')
    trace_test_acc = go.Scatter(x=list(range(1, num_epochs + 1)), y=test_accuracies, mode='lines+markers', name='Test Accuracy')
    fig_acc.add_trace(trace_train_acc)
    fig_acc.add_trace(trace_test_acc)
    fig_acc.update_layout(title='Train and Test Accuracy Over Epochs', xaxis_title='Epoch', yaxis_title='Accuracy')

    fig_loss = make_subplots(rows=1, cols=1, subplot_titles=("Train and Test Loss",))
    trace_train_loss = go.Scatter(x=list(range(1, num_epochs + 1)), y=train_losses, mode='lines+markers', name='Train Loss')
    trace_test_loss = go.Scatter(x=list(range(1, num_epochs + 1)), y=test_losses, mode='lines+markers', name='Test Loss')
    fig_loss.add_trace(trace_train_loss)
    fig_loss.add_trace(trace_test_loss)
    fig_loss.update_layout(title='Train and Test Loss Over Epochs', xaxis_title='Epoch', yaxis_title='Loss')

    fig_time = make_subplots(rows=1, cols=1, subplot_titles=("Epoch Time",))
    trace_epoch_time = go.Scatter(x=list(range(1, num_epochs + 1)), y=epoch_times, mode='lines+markers', name='Epoch Time')
    fig_time.add_trace(trace_epoch_time)
    fig_time.update_layout(title='Epoch Time Over Training', xaxis_title='Epoch', yaxis_title='Time (seconds)')

    fig_acc.show()
    fig_loss.show()
    fig_time.show()
