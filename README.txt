# Progressive hierarchical fusion

## Structure

### src
You can find
- main.py which runs the program
- get_dataset.py which extracts / performs initial fusion on a structured OCT dataset
- oct_trainer.py which houses the training operations, functions, etc
- ssn2v_like.py which includes the custom Unet model

### outputs
I have included one example output from my training. 
- I am wrestling with how to evaluate this. The outputs are clearly of extremely high feature retention, however, without any true clean images certain evaluation metrics such as SSIM and PSNR are less useful. 

## Data Structure
FusedDataset/{PATIENT}/{FUSEDIMAGELEVEL0, FUSEDIMAGELEVEL1,.... FUSEDIMAGELEVELN}