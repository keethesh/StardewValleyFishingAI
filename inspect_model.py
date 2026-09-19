
import torch

def inspect_checkpoint(path):
    print(f"Inspecting {path}...")
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        # Determine if it's a full checkpoint or just state dict
        if 'local_state_dict' in checkpoint:
            state_dict = checkpoint['local_state_dict']
            print("Found Checkpoint with keys: ", checkpoint.keys())
        else:
            state_dict = checkpoint
            print("Found raw State Dict")
            
        print("\nLayer Shapes:")
        for key, value in state_dict.items():
            if 'feature_layer' in key or 'value_stream' in key or 'advantage_stream' in key:
                 print(f"{key}: {value.shape}")
                 
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_checkpoint("models/model_v1_old_json.pth")
