import torch
import os
from pathlib import Path
def save_checkpoint(epoch,run_number,val_accuracy, model,optimizer,scheduler, filename):
    state = {
        'epoch': epoch,
        'run_number': run_number,
        'val_accuracy':val_accuracy,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(state, filename)

def load_checkpoint(model,optimizer,scheduler,filename:str|Path):
    filename = Path(filename)
    assert filename.is_absolute(), 'filename in load_checkpoint MUST be an absolute path'
    assert os.path.isfile(filename), 'filename in load_checkpoint is not a valid file'
    checkpoint = torch.load(filename)

    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    run_number = checkpoint['run_number']
    print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")

    return start_epoch, run_number