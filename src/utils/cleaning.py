import os
from safetensors.torch import load_file, save_file
def clean_accelerate_checkpoint(checkpoint_dir):
    """
    Remove num_batches_tracked parameters from Accelerate checkpoint files
    """
    print(f"Cleaning checkpoint in: {checkpoint_dir}")
    
    # Check for different possible checkpoint file names
    possible_files = [
        'model_1.safetensors','model_2.safetensors','model.safetensors'
    ]
    
    cleaned_files = []
    
    for filename in possible_files:
        file_path = os.path.join(checkpoint_dir, filename)
        
        if os.path.exists(file_path):
            print(f"Processing {filename}...")
            
            try:
                state_dict = load_file(file_path)
                
                # Find keys to remove
                keys_to_remove = [k for k in state_dict.keys() if 'num_batches_tracked' in k]
                
                if keys_to_remove:
                    print(f"Removing {len(keys_to_remove)} num_batches_tracked keys...")
                    for key in keys_to_remove:
                        del state_dict[key]
                    
                    # Save back
                    save_file(state_dict, file_path)
                    
                    cleaned_files.append(filename)
                else:
                    print(f"No num_batches_tracked keys found in {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    if cleaned_files:
        print(f"Successfully cleaned: {cleaned_files}")
    else:
        print("No files were cleaned")
    
    return len(cleaned_files) > 0