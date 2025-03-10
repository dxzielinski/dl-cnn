import torch

def _calc_conv_output_size(size=32, kernel_size=3, stride=1, padding=0, pooling_size=2):
    conv_out = (size - kernel_size + 2 * padding) // stride + 1
    pool_out = conv_out // pooling_size
    return pool_out

class BasicCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=64*_calc_conv_output_size(_calc_conv_output_size())**2, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=10)
        )
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    
    def forward(self, x):
        return self.model(x)
    
    def fit(self, train_loader, device="cuda", epochs=10):
        self.to(device)
        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self(inputs.to(device))
                labels = labels.to(device)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
    
    def evaluate(self, test_loader, device="cuda"):
        self.to(device)
        self.eval()
        classes = test_loader.get_classes()
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                labels = labels.to(device)
                images = images.to(device)
                outputs = self(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
        
        return correct_pred, total_pred