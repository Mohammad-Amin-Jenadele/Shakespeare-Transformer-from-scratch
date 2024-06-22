from collections import defaultdict
import nltk
from nltk.tokenize import RegexpTokenizer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class TextTokenizer:
    def __init__(self, repetition_threshold):
        self.repetition_threshold = repetition_threshold
        self.word_freq = defaultdict(int)
        self.token_dict = {'<SOS>': 0, '<UNK>': 1, '<EOS>': 2}
        self.reverse_token_dict = {0: '<SOS>', 1: '<UNK>', 2: '<EOS>'}
        self.next_index = 3
        self.tokenizer = RegexpTokenizer(r"[\w']+|[.,!?;:&()\n]")

    def tokenize(self, text):
        """Tokenizes the text into words."""
        return self.tokenizer.tokenize(text)

    def track_word_frequencies(self, tokens):
        """Tracks the frequency of each word."""
        for token in tokens:
            self.word_freq[token] += 1

    def filter_words_by_frequency(self):
        """Filters words that exceed the specified repetition threshold and adds them to the dictionary."""
        for word, count in self.word_freq.items():
            if count > self.repetition_threshold:
                if word not in self.token_dict:
                    self.token_dict[word] = self.next_index
                    self.reverse_token_dict[self.next_index] = word
                    self.next_index += 1

    def process_text(self, text):
        """Main method to process text and build the dictionary."""
    
        # Step 1: Tokenize the text
        tokens = self.tokenize(text)

        # Step 2: Track word frequencies
        self.track_word_frequencies(tokens)

        # Step 3: Filter words by frequency and build the dictionary
        self.filter_words_by_frequency()

    def get_token_dict(self):
        """Returns the token dictionary."""
        return self.token_dict

    def get_reverse_token_dict(self):
        """Returns the reverse token dictionary."""
        return self.reverse_token_dict
    
    def tokens_to_text(self, tokens):
        """Converts a list of tokens back to the original text."""
        if tokens[0] == self.token_dict['<SOS>']:
            tokens = tokens[1:]
        if tokens[-1] == self.token_dict['<EOS>']:
            tokens = tokens[:-1]
        words = [self.reverse_token_dict.get(token, '<UNK>') for token in tokens]
        return ' '.join(words)
    
    def text_to_tokens(self, text):
        """Converts a given text to a list of tokens."""
        tokens = self.tokenize(text)
        token_list = [self.token_dict.get(token, 1) for token in tokens]
        token_list.insert(0, self.token_dict['<SOS>'])
        token_list.append(self.token_dict['<EOS>'])
        return token_list



def get_batch(dataset , batch_size , block_size):
    ix = torch.randint(len(dataset) - block_size , size = (batch_size,))
    x = torch.stack([dataset[i : i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1 : i + block_size + 1] for i in ix])
    x , y = x.to(device) , y.to(device)
    return x , y

    