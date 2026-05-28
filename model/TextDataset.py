import torch

"""
Data Preperation for training
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextDataset:
    def __init__(self, text: str, seq_length: int = 128):
        self.text = text
        self.seq_length = seq_length
        
        # Create character mapping with special tokens
        chars = sorted(list(set(text)))
        
        # Add special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        
        # Ensure special tokens are in the vocabulary
        special_tokens = [self.pad_token, self.unk_token, self.start_token, self.end_token]
        all_chars = special_tokens + chars
        
        self.vocab_size = len(all_chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(all_chars)}
        
        # Set token IDs
        self.pad_token_id = self.char_to_idx[self.pad_token]
        self.unk_token_id = self.char_to_idx[self.unk_token]
        self.start_token_id = self.char_to_idx[self.start_token]
        self.end_token_id = self.char_to_idx[self.end_token]
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Special tokens - PAD:{self.pad_token_id}, UNK:{self.unk_token_id}, START:{self.start_token_id}, END:{self.end_token_id}")
        
        # Prepare data with safe encoding
        encoded_text = []
        for ch in text:
            idx = self.char_to_idx.get(ch, self.unk_token_id)
            encoded_text.append(idx)
        
        self.data = torch.tensor(encoded_text, dtype=torch.long)
        
    def safe_encode(self, text: str) -> torch.Tensor:
        """Safely encode text with unknown token handling"""
        encoded = []
        for ch in text:
            idx = self.char_to_idx.get(ch, self.unk_token_id)
            encoded.append(idx)
        return torch.tensor(encoded, dtype=torch.long)
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text and add start token"""
        encoded = self.safe_encode(text)
        # Add start token
        encoded = torch.cat([torch.tensor([self.start_token_id]), encoded])
        return encoded.unsqueeze(0).to(device)
    
    def decode(self, tokens: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode tokens to text, optionally skipping special tokens"""
        tokens = tokens.squeeze().cpu().numpy()
        chars = []
        for idx in tokens:
            ch = self.idx_to_char.get(int(idx), self.unk_token)
            if skip_special_tokens and ch in [self.pad_token, self.start_token, self.end_token]:
                continue
            chars.append(ch)
        return ''.join(chars)
    
    def get_batch(self, batch_size: int):
        """Get a batch of training data"""
        # Random sampling of sequences
        max_start = len(self.data) - self.seq_length - 1
        if max_start <= 0:
            # Not enough data, use what we have
            self.seq_length = len(self.data) // 2
        
        ix = torch.randint(0, max(1, len(self.data) - self.seq_length - 1), (batch_size,))
        x = torch.stack([self.data[i:i+self.seq_length] for i in ix])
        y = torch.stack([self.data[i+1:i+self.seq_length+1] for i in ix])
        
        # Ensure indices are within bounds
        x = torch.clamp(x, 0, self.vocab_size - 1)
        y = torch.clamp(y, 0, self.vocab_size - 1)
        
        return x.to(device), y.to(device)