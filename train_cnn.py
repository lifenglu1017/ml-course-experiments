import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MNISTDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx].reshape(28, 28).astype(np.float32)
        image = image / 255.0
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).unsqueeze(0)
        
        if self.labels is not None:
            label = int(self.labels[idx])
            return image, label
        return image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        return x

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_data = train_df.iloc[:, 1:].values
    train_labels = train_df.iloc[:, 0].values
    test_data = test_df.values
    
    return train_data, train_labels, test_data

def train_model(model, train_loader, val_loader, config, device):
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    if config.get('scheduler', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        scheduler = None
    
    early_stopping = EarlyStopping(patience=5) if config.get('early_stopping', False) else None
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if scheduler:
            scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{config["epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, train_accs, val_accs, best_val_acc

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    
    return predictions

def save_submission(predictions, output_path):
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f'Submission saved to {output_path}')

def run_experiment(exp_name, config, train_data, train_labels, test_data, device):
    print(f'\n{"="*60}')
    print(f'Running {exp_name}')
    print(f'Config: {config}')
    print(f'{"="*60}\n')
    
    X_train, X_val, y_train, y_val = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42
    )
    
    if config.get('augmentation', False):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
    
    val_transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = MNISTDataset(X_train, y_train, transform=train_transform)
    val_dataset = MNISTDataset(X_val, y_val, transform=val_transform)
    test_dataset = MNISTDataset(test_data, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = CNN().to(device)
    
    train_losses, val_losses, train_accs, val_accs, best_val_acc = train_model(
        model, train_loader, val_loader, config, device
    )
    
    predictions = predict(model, test_loader, device)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'predictions': predictions,
        'model_state': model.state_dict(),
        'converged_epoch': len(train_losses)
    }

def plot_loss_curves(results, output_path):
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (exp_name, result) in enumerate(results.items()):
        epochs = range(1, len(result['train_losses']) + 1)
        plt.plot(epochs, result['train_losses'], color=colors[idx], 
                label=f'{exp_name} - Train', linestyle='-', linewidth=2)
        plt.plot(epochs, result['val_losses'], color=colors[idx], 
                label=f'{exp_name} - Val', linestyle='--', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('训练过程中的Loss曲线对比', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Loss curves saved to {output_path}')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_path = 'train.csv'
    test_path = 'test.csv'
    
    print('Loading data...')
    train_data, train_labels, test_data = load_data(train_path, test_path)
    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')
    
    experiments = {
        'Exp1 (SGD, lr=0.01, bs=64)': {
            'optimizer': 'SGD',
            'lr': 0.01,
            'batch_size': 64,
            'epochs': 30,
            'augmentation': False,
            'early_stopping': False,
            'scheduler': False
        },
        'Exp2 (Adam, lr=0.001, bs=64)': {
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 64,
            'epochs': 30,
            'augmentation': False,
            'early_stopping': False,
            'scheduler': False
        },
        'Exp3 (Adam, lr=0.001, bs=128, ES)': {
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 128,
            'epochs': 50,
            'augmentation': False,
            'early_stopping': True,
            'scheduler': False
        },
        'Exp4 (Adam, lr=0.001, bs=64, Aug, ES)': {
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 64,
            'epochs': 50,
            'augmentation': True,
            'early_stopping': True,
            'scheduler': False
        }
    }
    
    results = {}
    for exp_name, config in experiments.items():
        results[exp_name] = run_experiment(
            exp_name, config, train_data, train_labels, test_data, device
        )
    
    plot_loss_curves(results, 'loss_curves.png')
    
    print('\n' + '='*60)
    print('实验结果汇总')
    print('='*60)
    for exp_name, result in results.items():
        print(f'\n{exp_name}:')
        print(f'  最佳验证准确率: {result["best_val_acc"]:.4f}')
        print(f'  收敛Epoch: {result["converged_epoch"]}')
        print(f'  最终训练Loss: {result["train_losses"][-1]:.4f}')
        print(f'  最终验证Loss: {result["val_losses"][-1]:.4f}')
    
    print('\n' + '='*60)
    print('训练最终模型（用于Kaggle提交）')
    print('='*60)
    
    final_config = {
        'optimizer': 'Adam',
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 50,
        'augmentation': True,
        'early_stopping': True,
        'scheduler': True
    }
    
    final_result = run_experiment(
        'Final Model', final_config, train_data, train_labels, test_data, device
    )
    
    torch.save(final_result['model_state'], 'model.pth')
    print('Model saved to model.pth')
    
    save_submission(final_result['predictions'], 'submission.csv')
    
    print('\n' + '='*60)
    print('实验完成！')
    print('='*60)
    print('生成的文件：')
    print('  - loss_curves.png: Loss曲线对比图')
    print('  - model.pth: 训练好的模型权重')
    print('  - submission.csv: Kaggle提交文件')

if __name__ == '__main__':
    main()
