import torch

from TextDataset import TextDataset
from CustomTransformerGenerator import CustomTransformerGenerator
from TrainModel import train_model

import warnings


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


if __name__ == "__main__":
    # Sample training text (you can replace with any text file)
    sample_text = """
    Once upon a time in a small village nestled between rolling hills and a sparkling river,
    there lived a young girl named Elara. She was known throughout the village for her
    boundless curiosity and her love of exploring the ancient forest that bordered their home.
    Every day after finishing her chores, Elara would venture into the woods, discovering
    hidden glades, talking to woodland creatures, and collecting interesting stones and feathers.
    Her grandmother often told her stories of a magical kingdom hidden deep within the forest,
    accessible only to those with a pure heart and unwavering courage. These stories fascinated
    Elara, and she dreamed of one day finding this mythical place. One afternoon, while following
    an unfamiliar path she had never noticed before, Elara stumbled upon a peculiar golden key
    lying beneath an old oak tree. As she picked it up, the key began to glow warmly in her hand,
    and she heard a faint whisper on the wind, inviting her to unlock the greatest adventure
    of her life.
    """ * 10  # Repeat to have more data

    # Create dataset
    print("Creating dataset...")
    dataset = TextDataset(sample_text, seq_length=64)
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Total characters: {len(dataset.data)}")

    # Create model
    print("\nCreating model...")
    model = CustomTransformerGenerator(
        vocab_size=dataset.vocab_size,
        d_model=256,      # Smaller for faster training
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("\n" + "="*50)
    losses = train_model(
        model=model,
        dataset=dataset,
        num_epochs=50,  # Adjust based on your needs
        batch_size=32,
        learning_rate=3e-4
    )

    # Final generation example
    print("\n" + "="*50)
    print("Final text generation examples:")
    print("="*50)

    model.eval()
    test_prompts = ["The", "Once", "She", "In the"]

    for prompt in test_prompts:
        start_tokens = dataset.encode(prompt)
        generated = model.generate(
            start_tokens,
            max_new_tokens=150,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        generated_text = dataset.decode(generated)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated_text}")
        print("-"*50)

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'char_to_idx': dataset.char_to_idx,
        'idx_to_char': dataset.idx_to_char,
        'model_config': {
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'd_ff': 1024,
            'max_seq_len': 512,
            'dropout': 0.1
        }
    }, 'custom_transformer_model.pth')
    print("\nModel saved to 'custom_transformer_model.pth'")
