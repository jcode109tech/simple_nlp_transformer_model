import torch
import sys
import time 

from chatbot.pymodel import TransformerBlock, CustomTransformerGenerator


class Chatbot:
    def __init__(self, model_path: str = 'custom_transformer_model_fixed.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conversation_history = []
        self.max_history = 8
        
        # Default generation parameters
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9
        self.max_response_length = 150
        
        print(f"🤖 Loading model from {model_path}...")
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load the trained model and vocabulary"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load vocabulary
            self.vocab_size = checkpoint['vocab_size']
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_char = checkpoint['idx_to_char']
            
            # Handle special tokens (with defaults if not found)
            self.pad_token_id = checkpoint.get('pad_token_id', 0)
            self.unk_token_id = checkpoint.get('unk_token_id', 1)
            self.start_token_id = checkpoint.get('start_token_id', 2)
            self.end_token_id = checkpoint.get('end_token_id', 3)
            
            # Load model config
            model_config = checkpoint['model_config']
            
            # Create model
            self.model = CustomTransformerGenerator(
                vocab_size=self.vocab_size,
                d_model=model_config['d_model'],
                num_heads=model_config['num_heads'],
                num_layers=model_config['num_layers'],
                d_ff=model_config['d_ff'],
                max_seq_len=model_config['max_seq_len'],
                dropout=model_config['dropout'],
                pad_token_id=self.pad_token_id,
                unk_token_id=self.unk_token_id
            ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"   📊 Vocabulary size: {self.vocab_size}")
            print(f"   🎯 Device: {self.device}")
            print(f"   ⚙️  Config: d_model={model_config['d_model']}, layers={model_config['num_layers']}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\n💡 Tips:")
            print("   1. Make sure the model file exists")
            print("   2. Train the model first using transformer_model_fixed.py")
            print("   3. Check that the model architecture matches")
            sys.exit(1)
    
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to token IDs"""
        tokens = []
        for ch in text:
            idx = self.char_to_idx.get(ch, self.unk_token_id)
            tokens.append(idx)
        
        # Add start token if it exists
        if hasattr(self, 'start_token_id'):
            tokens = [self.start_token_id] + tokens
        
        return torch.tensor([tokens], dtype=torch.long).to(self.device)
    
    def decode(self, tokens: torch.Tensor, skip_special: bool = True) -> str:
        """Convert token IDs back to text"""
        tokens = tokens.squeeze().cpu().numpy()
        chars = []
        for idx in tokens:
            ch = self.idx_to_char.get(int(idx), '?')
            if skip_special and ch in ['<PAD>', '<START>', '<END>', '<UNK>']:
                continue
            chars.append(ch)
        return ''.join(chars).strip()
    
    def format_prompt(self, user_input: str) -> str:
        """Format conversation history into a prompt"""
        self.conversation_history.append(f"Human: {user_input}")
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # Build prompt
        prompt = "\n".join(self.conversation_history) + "\nAssistant:"
        return prompt
    
    def generate_response(self, user_input: str) -> str:
        """Generate a response to user input"""
        try:
            # Format prompt with history
            prompt = self.format_prompt(user_input)
            
            # Encode
            input_tokens = self.encode(prompt)
            
            # Limit input length
            max_context = 400
            if input_tokens.size(1) > max_context:
                input_tokens = input_tokens[:, -max_context:]
            
            # Generate
            with torch.no_grad():
                output_tokens = self.model.generate(
                    start_tokens=input_tokens,
                    max_new_tokens=self.max_response_length,
                    temperature=self.temperature,
                    top_k=self.top_k if self.top_k > 0 else None,
                    top_p=self.top_p if self.top_p > 0 else None,
                    stop_token=self.end_token_id if hasattr(self, 'end_token_id') else None
                )
            
            # Decode full response
            full_response = self.decode(output_tokens, skip_special=True)
            
            # Extract only the new part (remove the prompt)
            prompt_text = self.decode(input_tokens, skip_special=True)
            response = full_response[len(prompt_text):].strip()
            
            # Clean up response
            if not response:
                response = "I'm not sure how to respond to that."
            
            # Add to history
            self.conversation_history.append(f"Assistant: {response}")
            
            return response
            
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return "I encountered an error. Please try again."
    
    def chat(self):
        """Start interactive chat session"""
        print("\n" + "="*60)
        print("🤖 CHAT WITH YOUR CUSTOM AI MODEL")
        print("="*60)
        print("\n💡 Commands:")
        print("   /quit or /exit - End conversation")
        print("   /clear         - Clear conversation history")
        print("   /temp <val>    - Set temperature (0.1-2.0)")
        print("   /topk <val>    - Set top-k (1-100)")
        print("   /topp <val>    - Set top-p (0.1-1.0)")
        print("   /settings      - Show current settings")
        print("   /help          - Show this help")
        print("\n" + "="*60)
        print("Start chatting!\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    cmd_parts = user_input.split()
                    cmd = cmd_parts[0].lower()
                    
                    if cmd in ['/quit', '/exit']:
                        print("\n👋 Goodbye! Thanks for chatting!")
                        break
                    
                    elif cmd == '/clear':
                        self.conversation_history = []
                        print("✅ Conversation history cleared!")
                        continue
                    
                    elif cmd == '/temp' and len(cmd_parts) > 1:
                        try:
                            self.temperature = float(cmd_parts[1])
                            print(f"✅ Temperature set to {self.temperature}")
                        except:
                            print("❌ Invalid temperature value")
                        continue
                    
                    elif cmd == '/topk' and len(cmd_parts) > 1:
                        try:
                            self.top_k = int(cmd_parts[1])
                            print(f"✅ Top-k set to {self.top_k}")
                        except:
                            print("❌ Invalid top-k value")
                        continue
                    
                    elif cmd == '/topp' and len(cmd_parts) > 1:
                        try:
                            self.top_p = float(cmd_parts[1])
                            print(f"✅ Top-p set to {self.top_p}")
                        except:
                            print("❌ Invalid top-p value")
                        continue
                    
                    elif cmd == '/settings':
                        print(f"\nCurrent Settings:")
                        print(f"  🌡️  Temperature: {self.temperature}")
                        print(f"  📊 Top-k: {self.top_k}")
                        print(f"  🎯 Top-p: {self.top_p}")
                        print(f"  📝 Max response length: {self.max_response_length}")
                        print(f"  💬 History size: {len(self.conversation_history)} exchanges")
                        continue
                    
                    elif cmd == '/help':
                        print("\nAvailable Commands:")
                        print("  /quit, /exit     - End conversation")
                        print("  /clear           - Clear conversation history")
                        print("  /temp <value>    - Set temperature (0.1=conservative, 1.5=creative)")
                        print("  /topk <value>    - Sample from top K tokens only")
                        print("  /topp <value>    - Nucleus sampling threshold")
                        print("  /settings        - Show current settings")
                        print("  /help            - Show this help")
                        continue
                    
                    else:
                        print(f"❌ Unknown command. Type /help for commands.")
                        continue
                
                # Generate and display response with typing effect
                print("\nAssistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                
                # Typing effect
                for char in response:
                    print(char, end="", flush=True)
                    time.sleep(0.008)
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")