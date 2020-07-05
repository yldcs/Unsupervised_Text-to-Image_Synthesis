import os 
class Vocabulary(object):
    def __init__(self,
                 vocab_file,
                 start_word="<S>",
                 end_word="</S>",
                 unk_word="<UNK>"):
        """Initializes the vocabulary."""
        assert os.path.exists(vocab_file), 'can not find file: %s' % vocab_file

        with open(vocab_file, 'r') as f:
            reverse_vocab = list(f.readlines())
        reverse_vocab = [line.split()[0] for line in reverse_vocab] 
        assert start_word in reverse_vocab 
        assert end_word in reverse_vocab
        if unk_word not in reverse_vocab:
            reverse_vocab.append(unk_word)
        vocab = dict([(x,y) for y,x in enumerate(reverse_vocab)])

        self.vocab = vocab
        self.reverse_vocab = reverse_vocab # reverse_vocab[id] = word
        # Save special word ids.
        self.start_id = vocab[start_word]
        self.end_id = vocab[end_word]
        self.unk_id = vocab[unk_word]
             
    def word_to_id(self, word):
        """Returns the intger id of a word string"""
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.unk_id
    def id_to_word(self, word_id):
        """Returns the word string of an intger word id."""
        if word_id >= len(self.reverse_vocab):
            return self.reverse_vocab[self.unk_id]
        else:
            return self.reverse_vocab[word_id]
