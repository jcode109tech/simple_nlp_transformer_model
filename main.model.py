import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


from SafeTextDataset import SafeTextDataset


if __name__ == '__main__':
    # Sample training text
    n = 10
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
    """ * n  

    print("Creating datasets ....")
    datasets = SafeTextDataset(sample_text, seq_length=64)
    print(f"Dataset size: {len(dataset.data)} characters")


