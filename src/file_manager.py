import os
import shutil
import pathlib

def organize_files(file_paths, labels, output_dir, mode='copy'):
    """
    Organizes files into folders based on cluster labels.
    
    Args:
        file_paths (list): List of original file paths.
        labels (list): List of cluster labels corresponding to file_paths.
        output_dir (str): Root directory for organized files.
        mode (str): 'copy' (default) or 'move'. 'copy' is safer.
    """
    
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, file_path in enumerate(file_paths):
        label = labels[i]
        
        # Create cluster directory: e.g., <output_dir>/Cluster_0
        cluster_dir = os.path.join(output_dir, f"Cluster_{label}")
        pathlib.Path(cluster_dir).mkdir(exist_ok=True)
        
        # Get filename
        filename = os.path.basename(file_path)
        dest_path = os.path.join(cluster_dir, filename)
        
        try:
            if mode == 'copy':
                shutil.copy2(file_path, dest_path) # copy2 preserves metadata
                # print(f"Copied: {filename} -> Cluster_{label}")
            elif mode == 'move':
                shutil.move(file_path, dest_path)
                # print(f"Moved: {filename} -> Cluster_{label}")
            else:
                print(f"Unknown mode: {mode}")
        except Exception as e:
            print(f"Error organizing {filename}: {e}")

    print(f"\nOrganization complete! Files are in: {output_dir}")
