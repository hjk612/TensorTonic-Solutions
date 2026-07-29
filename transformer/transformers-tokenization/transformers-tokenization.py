import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        vocab = []
        for text in texts:
            for word in text.split():
                if word not in vocab:
                    vocab.append(word)
                    
        self.vocab_size = len(special_tokens) + len(vocab)
        final_vocab = special_tokens + sorted(vocab)

        for i in range(self.vocab_size):
            self.word_to_id[final_vocab[i]] = i
            self.id_to_word[i] = final_vocab[i]
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        encoding = []
        for word in text.split():
            encoding.append(self.word_to_id.get(word.lower(), 1))
        return encoding

    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoding = []
        for id in ids:
            decoding.append(self.id_to_word.get(id, '<UNK>'))
        return " ".join(decoding)
