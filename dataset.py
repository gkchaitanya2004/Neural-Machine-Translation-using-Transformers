from datasets import load_dataset
import spacy

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        self.dataset = load_dataset("bentrevett/multi30k", split=split)
        self.en_nlp = spacy.load("en_core_web_sm")
        self.de_nlp = spacy.load("de_core_news_sm")


    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        en_dict = dict()
        de_dict = dict()

        en_dict['<unk>'] = 0
        en_dict['<pad>'] = 1
        en_dict['<sos>'] = 2
        en_dict['<eos>'] = 3

        de_dict['<unk>'] = 0
        de_dict['<pad>'] = 1
        de_dict['<sos>'] = 2
        de_dict['<eos>'] = 3

        for item in self.dataset:
            en_words = self.en_nlp(item['en'])
            de_words = self.de_nlp(item['de'])


            for words in en_words:
                if words.text not in en_dict:
                    en_dict[words.text] = len(en_dict)

            for words in de_words:
                if words.text not in de_dict:
                    de_dict[words.text] = len(de_dict)

        self.en_vocab = en_dict
        self.de_vocab = de_dict

        self.en_vocab_rev = {v: k for k, v in en_dict.items()}
        self.de_vocab_rev = {v: k for k, v in de_dict.items()}



    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices

        en_tokenized = []
        de_tokenized = []

        for item in self.dataset:
            en_tokens = self.en_nlp(item['en'])
            de_tokens = self.de_nlp(item['de'])

            cur_en_tokenized = [self.en_vocab['<sos>']]
            cur_de_tokenized = [self.de_vocab['<sos>']]

            for token in en_tokens:
                cur_en_tokenized.append(self.en_vocab.get(token.text, self.en_vocab['<unk>']))

            for token in de_tokens:
                cur_de_tokenized.append(self.de_vocab.get(token.text, self.de_vocab['<unk>']))

            cur_en_tokenized.append(self.en_vocab['<eos>'])
            cur_de_tokenized.append(self.de_vocab['<eos>'])

            en_tokenized.append(cur_en_tokenized)
            de_tokenized.append(cur_de_tokenized)

        self.en_tokenized = en_tokenized
        self.de_tokenized = de_tokenized


        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.de_tokenized[idx], self.en_tokenized[idx]
