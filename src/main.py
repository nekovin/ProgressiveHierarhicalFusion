#from full_framework import *
from get_dataset import *
from oct_trainer import *
from ssn2v_like import create_progressive_fusion_unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = create_progressive_fusion_unet(n_fusion_levels=6)
dataset = get_dataset(basedir='../FusedDataset', size=512)
train_loader, val_loader = get_loaders(dataset) #6,1,w,h
#visualise_loader(train_loader)

trainer = OCTDenoiseTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-4,
    checkpoint_dir='checkpoints'
)

#trainer.train(num_epochs=5)

trainer.predict(model, val_loader)

