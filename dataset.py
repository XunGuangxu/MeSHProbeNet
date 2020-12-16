import torch
import torchtext
import six
from torchtext.data.pipeline import Pipeline
from torchtext.data.utils import get_tokenizer

class Vocabulary(object):
    def __init__(self, vocab_path):
        self.itos = ["<pad>"]
        self.stoi = {"<pad>": 0}
        for line in open(vocab_path):
            wid, word = line.strip().split('\t')
            if word == "<pad>": continue
            self.stoi[word] = len(self.stoi)
            self.itos.append(word)
            

class NumJrnlField(torchtext.data.RawField):
    def __init__(self, tensor_type=torch.LongTensor, preprocessing=None, postprocessing=None,
            tokenize=(lambda s: s.split()), batch_first=True, pad_id=0):
        self.tensor_type = tensor_type
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.tokenize = get_tokenizer(tokenize)
        self.batch_first = batch_first
        self.pad_id = pad_id

    def preprocess(self, x):
        if (six.PY2 and isinstance(x, six.string_types) and not
                isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if isinstance(x, six.text_type):
            x = self.tokenize(x.rstrip('\n'))
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device, train):
        tensor = self.make_variable(batch, device=device, train=train)
        return tensor

    def make_variable(self, arr, device=None, train=True):
        arr = self.tensor_type(arr)
        if not self.batch_first:
            arr.t_()
        if device == -1:
            arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
        return arr


class NumMeshField(torchtext.data.RawField):
    def __init__(self, tensor_type=torch.FloatTensor, preprocessing=None, postprocessing=None,
            tokenize=(lambda s: s.split()), batch_first=True, pad_id=0, vocab_size=0):
        self.tensor_type = tensor_type
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.tokenize = get_tokenizer(tokenize)
        self.batch_first = batch_first
        self.pad_id = pad_id
        self.vocab_size = vocab_size

    def preprocess(self, x):
        if (six.PY2 and isinstance(x, six.string_types) and not
                isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if isinstance(x, six.text_type):
            x = self.tokenize(x.rstrip('\n'))
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device, train):
        label = self.make_label(batch)
        tensor = self.make_variable(label, device=device, train=train)
        return tensor

    def make_label(self, minibatch):
        labels = torch.zeros(len(minibatch), self.vocab_size)
        for row, meshdoc in enumerate(minibatch):
            for col in meshdoc:
                labels[row, col] = 1
        return labels

    def make_variable(self, arr, device=None, train=True):
        arr = self.tensor_type(arr)
        if not self.batch_first:
            arr.t_()
        if device == -1:
            arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
        return arr


class NumWordField(torchtext.data.RawField):
    def __init__(self, tensor_type=torch.LongTensor, preprocessing=None, postprocessing=None,
            tokenize=(lambda s: s.split()), include_lengths=False, batch_first=True, pad_id=0, pre_max_len=1000):
        self.tensor_type = tensor_type
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.tokenize = get_tokenizer(tokenize)
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_id = pad_id
        self.pre_max_len = pre_max_len

    def preprocess(self, x):
        if (six.PY2 and isinstance(x, six.string_types) and not
                isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if isinstance(x, six.text_type):
            x = self.tokenize(x.rstrip('\n'))
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device, train):
        padded = self.pad(batch)
        tensor = self.make_variable(padded, device=device, train=train)
        return tensor

    def pad(self, minibatch):
        minibatch = list(minibatch)
        max_len = max(len(x) for x in minibatch)
        max_len = min(max_len, self.pre_max_len)
        padded, lengths = [], []
        for x in minibatch:
            padded.append(list(x[:max_len]) +
                [self.pad_id] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded

    def make_variable(self, arr, device=None, train=True):
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)
        if self.postprocessing is not None:
            arr = self.postprocessing(arr)

        arr = self.tensor_type(arr)
        if not self.batch_first:
            arr.t_()
        if device == -1:
            arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
            if self.include_lengths:
                lengths = lengths.cuda(device)
        if self.include_lengths:
            return arr, lengths
        return arr