import torch
import torch.nn.functional as F

from CustomTransformerGenerator import CustomTransformerGenerator
from TextDataset import TextDataset


device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(
    model: CustomTransformerGenerator,
    dataset: TextDataset,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    warmup_steps: int = 1000
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))

    # Create causal mask (prevents looking at future tokens)
    def create_causal_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(device)

    model.train()
    losses = []

    print("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = len(dataset.data) // batch_size // dataset.seq_length

        for step in range(num_batches):
            # Get batch
            x, y = dataset.get_batch(batch_size)

            # Create causal mask
            mask = create_causal_mask(x.size(1))

            # Forward pass
            logits = model(x, mask)
            loss = F.cross_entropy(logits.view(-1, dataset.vocab_size), y.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()

            # Print progress
            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step}/{num_batches}, Loss: {avg_loss:.4f}")

        avg_epoch_loss = total_loss / num_batches
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Generate sample text every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                start_text = "The"
                start_tokens = dataset.encode(start_text)
                generated = model.generate(
                    start_tokens,
                    max_new_tokens=100,
                    temperature=0.8,
                    top_k=50
                )
                generated_text = dataset.decode(generated)
                print(f"\n--- Sample generation at epoch {epoch+1} ---")
                print(generated_text)
                print("---\n")
            model.train()

    return losses
