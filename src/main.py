from dataset import get_dataset, get_loaders
from modelling.train import trainer
from modelling.ProgressiveFusionUNET_V2 import create_progressive_fusion_unet
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

levels = 6
img_size = 128

model = create_progressive_fusion_unet(n_fusion_levels=levels)
train_loader, val_loader, test_loader = get_loaders(basedir=r'..\data\FusedDataset', size=img_size, levels=levels) #6,1,w,h # [0] is the images

trainer = trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-3,
    checkpoint_dir='checkpoints',
    img_size=img_size,
)

trainer.train(num_epochs=10)