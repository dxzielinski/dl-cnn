import torch
import lightning as L
import torchmetrics
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from torchvision.models.mobilenet import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

from model_config import TASK, NUM_CLASSES
from kan_convolutional.KANConv import KAN_Convolutional_Layer
from efficient_kan.kan import KANLinear


def _calc_conv_output_size(size=32, kernel_size=2, stride=1, padding=0, pooling_size=2):
    conv_out = (size - kernel_size + 2 * padding) // stride + 1
    pool_out = conv_out // pooling_size
    return pool_out


class BasicCNN(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=64 * _calc_conv_output_size(_calc_conv_output_size()) ** 2,
                out_features=64,
            ),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        return self.layers(x)


class Model(L.LightningModule):
    def __init__(self, hyperparameters):
        """
        Args:
            hyperparameters: dict containing model hyperparameters - it must contain the
                following keys:
                - learning_rate: float, learning rate for the optimizer
                - dropout: float, dropout rate before the last linear layer
                - weight_decay: float, weight decay for the optimizer
        """
        super().__init__()
        self.model = BasicCNN(hyperparameters["dropout"])
        self.learning_rate = hyperparameters["learning_rate"]
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "f1_macro": torchmetrics.F1Score(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
                "precision": torchmetrics.Precision(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
                "recall": torchmetrics.Recall(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
                "auroc": torchmetrics.AUROC(
                    task=TASK, num_classes=NUM_CLASSES, average="macro"
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.train_batch_outputs = []
        self.val_batch_outputs = []
        self.test_batch_outputs = []
        self.hyperparameters = hyperparameters

    def on_train_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams(self.hyperparameters)

    def on_test_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams(self.hyperparameters)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probabilities = torch.softmax(logits, dim=1)
        self.train_batch_outputs.append({"probabilities": probabilities, "y": y})
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        probabilities = torch.cat(
            [x["probabilities"] for x in self.train_batch_outputs]
        )
        y = torch.cat([x["y"] for x in self.train_batch_outputs])
        metrics = self.train_metrics(probabilities, y)
        self.log_dict(metrics)
        self.train_metrics.reset()
        self.train_batch_outputs.clear()

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probabilities = torch.softmax(logits, dim=1)
        self.val_batch_outputs.append({"probabilities": probabilities, "y": y})
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        probabilities = torch.cat([x["probabilities"] for x in self.val_batch_outputs])
        y = torch.cat([x["y"] for x in self.val_batch_outputs])
        metrics = self.val_metrics(probabilities, y)
        self.log_dict(metrics)
        self.val_metrics.reset()
        self.val_batch_outputs.clear()

    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        probabilities = torch.softmax(logits, dim=1)
        self.test_batch_outputs.append({"probabilities": probabilities, "y": y})

    def on_test_epoch_end(self):
        probabilities = torch.cat([x["probabilities"] for x in self.test_batch_outputs])
        y = torch.cat([x["y"] for x in self.test_batch_outputs])
        metrics = self.test_metrics(probabilities, y)
        self.log_dict(metrics)
        self.test_metrics.reset()
        self.test_batch_outputs.clear()

    def configure_optimizers(self):
        base_lr = self.learning_rate / 10
        max_lr = self.learning_rate * 10
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.hyperparameters["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer, base_lr=base_lr, max_lr=max_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


default_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        # hue shifts color around the RGB color wheel. +-0.05 * 365deg ~= +-18deg
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.5, scale=(0.005, 0.01), ratio=(1.2, 1.8)
        ),  # cutout
        # 32*32 = 1024; 0.005 * 1024 ~= 5; 3x2 has ratio 1.5, so ratio can be 1.2-1.8
        transforms.Normalize(
            mean=[0.4789, 0.4723, 0.4305], std=[0.2421, 0.2383, 0.2587]
        ),
    ]
)

no_augmentation = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4789, 0.4723, 0.4305], std=[0.2421, 0.2383, 0.2587]
        ),
    ]
)


class ClassificationData(L.LightningDataModule):
    def __init__(self, data_dir="../data", batch_size=16, transform=default_transforms):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = ImageFolder(
                root=f"{self.data_dir}/train", transform=self.transform
            )
            self.val_dataset = ImageFolder(
                root=f"{self.data_dir}/valid", transform=self.transform
            )
        if stage == "test":
            self.test_dataset = ImageFolder(
                root=f"{self.data_dir}/test", transform=self.transform
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )


inception_transforms = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.005, 0.01), ratio=(1.2, 1.8)),
        transforms.Normalize(
            mean=[0.4789, 0.4723, 0.4305], std=[0.2421, 0.2383, 0.2587]
        ),
    ]
)

inception_no_augmentation = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4789, 0.4723, 0.4305], std=[0.2421, 0.2383, 0.2587]
        ),
    ]
)


class PretrainedModelInception(Model):
    def __init__(
        self, batch_size=64, learning_rate=0.01, dropout=0.2, weight_decay=0.001
    ):
        super().__init__(
            hyperparameters={
                "learning_rate": learning_rate,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
            }
        )
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.aux_logits = False
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, NUM_CLASSES)


class PretrainedModelMobileNet(Model):
    def __init__(
        self, batch_size=64, learning_rate=0.01, dropout=0.2, weight_decay=0.001
    ):
        super().__init__(
            hyperparameters={
                "learning_rate": learning_rate,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
            }
        )
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.model.classifier[3] = torch.nn.Linear(1280, NUM_CLASSES)


class KANCNN(torch.nn.Module):
    """
    Due to implementation issues with the KANConvolutionalLayer, this model is not working.
    """

    def __init__(self, dropout):
        super().__init__()
        self.layers = torch.nn.Sequential(
            KAN_Convolutional_Layer(
                in_channels=3,
                out_channels=32,
                kernel_size=(2, 2),
                stride=(1, 1),
                grid_size=1,
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            KAN_Convolutional_Layer(
                in_channels=32,
                out_channels=64,
                kernel_size=(2, 2),
                stride=(1, 1),
                grid_size=1,
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=64 * _calc_conv_output_size(_calc_conv_output_size()) ** 2,
                out_features=64,
            ),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        return self.layers(x)


class ModelKANCNN(Model):
    """
    Due to implementation issues with the KANConvolutionalLayer, this model is not working.
    """

    def __init__(
        self, batch_size=64, learning_rate=0.01, dropout=0.2, weight_decay=0.001
    ):
        super().__init__(
            hyperparameters={
                "learning_rate": learning_rate,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
            }
        )
        self.model = KANCNN(dropout)


class KANLinearBase(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            KANLinear(
                64 * _calc_conv_output_size(_calc_conv_output_size()) ** 2,
                64,
            ),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            KANLinear(64, 10),
        )

    def forward(self, x):
        return self.layers(x)


class ModelKANLinear(Model):
    def __init__(
        self, batch_size=64, learning_rate=0.01, dropout=0.2, weight_decay=0.001
    ):
        super().__init__(
            hyperparameters={
                "learning_rate": learning_rate,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
            }
        )
        self.model = KANLinearBase(dropout)


vit_transforms = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.005, 0.01), ratio=(1.2, 1.8)),
        transforms.Normalize(
            mean=[0.4789, 0.4723, 0.4305], std=[0.2421, 0.2383, 0.2587]
        ),
    ]
)

vit_no_augmentation = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4789, 0.4723, 0.4305], std=[0.2421, 0.2383, 0.2587]
        ),
    ]
)


class PretrainedVisionTransformer(Model):
    def __init__(
        self, batch_size=64, learning_rate=0.01, dropout=0.2, weight_decay=0.001
    ):
        super().__init__(
            hyperparameters={
                "learning_rate": learning_rate,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
            }
        )
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.model.heads.head = torch.nn.Linear(
            self.model.heads.head.in_features, NUM_CLASSES
        )

    def forward(self, x):
        return self.model(x)
