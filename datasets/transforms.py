import numpy as np

def encode(input, tokenizer, max_length=128):
    encoded = tokenizer(input, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    x = encoded['input_ids'].squeeze()
    if 'attention_mask' in encoded:
        mask_x = encoded['attention_mask'].squeeze()
    try: 
        return list(zip(x.numpy(), mask_x.numpy()))
    except:
        return x.numpy()
    
    
class Tokenize(object):
    def __init__(self, tokenizer, max_length=128):
        self.encode = lambda x : encode(x, tokenizer, max_length)
        
    def __call__(self, data):
        inputs, targets = zip(*data)
        targets = np.array(targets)
        inputs = self.encode(list(inputs))
        return np.array([np.array(a, dtype=object) for a in zip(inputs, targets)], dtype=object)