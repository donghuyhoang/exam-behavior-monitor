import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="Modify pre-trained weights for custom num_classes")
    parser.add_argument('--input', default='r50_deformable_detr-checkpoint.pth', help='Input checkpoint path')
    parser.add_argument('--output', default='r50_deformable_detr_custom.pth', help='Output checkpoint path')
    args = parser.parse_args()

    # Load the checkpoint
    print(f"Loading weights from {args.input}")
    ckpt = torch.load(args.input, map_location='cpu', weights_only=False)

    # Remove the class_embed weights because the number of classes changed
    keys_to_remove = [key for key in ckpt['model'].keys() if 'class_embed' in key]

    for key in keys_to_remove:
        print(f"Removing {key} from checkpoint")
        del ckpt['model'][key]

    # Remove optimizer and epoch data so we start fresh
    for key in ['optimizer', 'lr_scheduler', 'epoch']:
        if key in ckpt:
            print(f"Removing {key} state from checkpoint")
            del ckpt[key]

    # Save the modified checkpoint
    torch.save(ckpt, args.output)
    print(f"Saved modified weights to {args.output}")
    print("You can now start training with: --resume r50_deformable_detr_custom.pth (or using it as --pretrained)")

if __name__ == '__main__':
    main()
