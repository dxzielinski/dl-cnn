import torch
import mlflow


def _calc_conv_output_size(size=32, kernel_size=3, stride=1, padding=0, pooling_size=2):
    conv_out = (size - kernel_size + 2 * padding) // stride + 1
    pool_out = conv_out // pooling_size
    return pool_out


class BasicCNN(torch.nn.Module):
    def __init__(self, learning_rate):
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
            torch.nn.Linear(in_features=64 * _calc_conv_output_size(_calc_conv_output_size()) ** 2, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=10),
        )
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def fit(self, train_dataset, val_dataset, device="cuda", epochs=10):
        self.to(device)
        self.train()
        mlflow.start_run()
        mlflow.log_param("learning_rate", self.learning_rate)
        mlflow.log_param("epochs", epochs)
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for i, data in enumerate(train_dataset.get_dataloader(), 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self(inputs.to(device))
                labels = labels.to(device)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 1000 == 999:
                    avg_train_loss = running_loss / 1000
                    mlflow.log_metric("train_loss", avg_train_loss)
                    running_loss = 0.0
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_dataset.get_dataloader():
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataset.get_dataloader())
            mlflow.log_metric("val_loss", avg_val_loss)

        mlflow.pytorch.log_model(self, "model")
        mlflow.end_run()

    def evaluate(self, test_dataset, device="cuda"):
        self.to(device)
        self.eval()
        classes = test_dataset.get_classes()
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        with torch.no_grad():
            for data in test_dataset.get_dataloader():
                images, labels = data
                labels = labels.to(device)
                images = images.to(device)
                outputs = self(images)
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        return correct_pred, total_pred
