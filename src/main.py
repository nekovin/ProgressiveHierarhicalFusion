#from full_framework import *
from get_dataset import *
from oct_trainer import *
from ProgressiveFusionUNET import create_progressive_fusion_unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

levels = 6
img_size = 256

model = create_progressive_fusion_unet(n_fusion_levels=levels)
dataset = get_dataset(basedir='../FusedDataset', size=img_size, levels=levels)
train_loader, val_loader = get_loaders(dataset) #6,1,w,h # [0] is the images
#train_loader = val_loader
#visualise_loader(train_loader)

trainer = OCTDenoiseTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-3,
    checkpoint_dir='checkpoints',
    img_size=img_size,
)

trainer.train(num_epochs=30)
#trainer.predict(model, val_loader) # doesnt work rn

