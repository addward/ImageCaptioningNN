import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models.inception import Inception3
from warnings import warn
from torch.utils.model_zoo import load_url


class BeheadedInception3(Inception3):
    """ Like torchvision.models.inception.Inception3 but the head goes separately """
    
    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        else: warn("Input isn't transformed")
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x_for_capt = x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        return x_for_attn, x_for_capt, x
    

class CaptionNet(nn.Module):
    def __init__(self, vocab_size, emb_size, cnn_feature_size=2048, hidden_dim=500, num_layers=2):
        super(self.__class__, self).__init__()
        self.num_layers = num_layers

        self.fc_h = nn.Linear(in_features  = cnn_feature_size,
                             out_features = hidden_dim)
        self.fc_c = nn.Linear(in_features  = cnn_feature_size,
                             out_features = hidden_dim)
        
        self.emb = nn.Embedding(num_embeddings = vocab_size, 
                                embedding_dim  = emb_size)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=0.2)
        
        self.fc_out = nn.Linear(in_features = hidden_dim,
                                out_features = vocab_size)
        
    def forward(self, image_vectors, captions_ix):
        """ 
        Apply the network in training mode. 
        :param image_vectors: torch tensor, that is the output of Inception_v3 model
            shape: [batch, cnn_feature_size]
        :param captions_ix: captions of the image (indexed) 
            shape:[batch, sent_len]
        :returns: output logits in the vocabulary emb space 
            shape: [sent_len, batch, vocab_size]
        """

        h_0 = torch.tanh(self.fc_h(image_vectors)).unsqueeze(0) # [1, batch, hidden_dim] 
        c_0 = torch.tanh(self.fc_c(image_vectors)).unsqueeze(0) # [1, batch, hidden_dim]
        # h_0, c_0 should be (num_layers, batch, hidden_dim)
        h_0 = h_0.repeat(self.num_layers, 1, 1)
        c_0 = c_0.repeat(self.num_layers, 1, 1)

        h_0 = self.dropout(h_0)
        c_0 = self.dropout(c_0)
        
        input = self.emb(captions_ix) # [batch, sent_len, emb_size]
        input = torch.permute(input, (1,0,2)) # [sent_len, batch, emb_size]

        output, _ = self.lstm(input, (h_0, c_0)) # [sent_len, batch, hidden_size]
        output = self.fc_out(output)             # [sent_len, batch, vocab_size]
        
        return output


class CaptionGenerator:
    def __init__(self, vocab_file, capNet_file):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        inception = BeheadedInception3(transform_input=True).train(False)
        inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        inception.load_state_dict(load_url(inception_url))

        self.vocab = Vocabulary(corpus=None, load=vocab_file)
        
        self.inception = inception.to(self.device)

        self.capNet = CaptionNet(vocab_size=len(self.vocab.words), 
                                 emb_size=300, 
                                 cnn_feature_size=2048, 
                                 hidden_dim=500, 
                                 num_layers=3).to(self.device)

        self.capNet.load_state_dict(torch.load(capNet_file))
        self.capNet.train(False)
        for layer in self.capNet.parameters():
            layer.requires_grad_(False)
        
    def generate_caption(self, image, caption_prefix='<bos>', t=0.3, sample=True, max_len=15):
        assert isinstance(image, np.ndarray) and np.max(image) <= 1\
           and np.min(image) >=0 and image.shape[-1] == 3
    
        with torch.no_grad():
            image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32).to(self.device)

            vectors_8x8, vectors_neck, logits = self.inception(image[None])
            caption_prefix = self.vocab.sent_to_idx([caption_prefix])

            for _ in range(max_len):

                inwordseq = torch.tensor([caption_prefix]).to(self.device) # [batch, sent_len] = [1,caption_prefix_len]
                outword = self.capNet(vectors_neck, inwordseq)             # [sent_len, batch, vocab_size] = [caption_prefix_len+1,1,vocab_size]
                outword = outword.to('cpu')[-1].squeeze()/t                # [vocab_size]
                outword = F.softmax(outword, dim=0)
                if sample == False:
                    idx = outword.argmax().item()
                else:
                    idx = torch.multinomial(outword, 1)
                caption_prefix.append(idx)

                if self.vocab.words[idx] == self.vocab.eos:
                    break
                
        return self.vocab.idx_to_sent(caption_prefix)

    def get_caption_message(self, image_file):
        img = Image.open(image_file).convert('RGB').resize((299,299))
        img = np.array(img).astype('float32') / 255.
        output = ''
        for i in range(4):
            output += '({}).'.format(i+1) + ' '.join(self.generate_caption(img, t=0.2, max_len=25, sample=True)[1:-1]) + '\n'
        return output


class Vocabulary:

    def __init__(self, corpus, thr=5, trash_words=["..", "...", "\\", "`", ">"], load=None):

        self.bos = '<bos>'
        self.eos = '<eos>'
        self.pad = '<pad>'
        if load == None:
            words = [self.split_sentence(sent) for sent in corpus]
            # flatten the array
            words = [item for sent in words for item in sent]
            words = np.array(words)
            w, n = np.unique(words, return_counts=True)
            w = w[(n > thr) & ~np.isin(w, trash_words)]
            self.words = np.insert(w, 0, (self.bos, self.eos, self.pad))

            self.word2idx = dict(zip(self.words, range(self.words.size)))
        else:
            self.load_dict(load)

    def load_dict(self, path):
        self.words = np.load(path)
        self.word2idx = dict(zip(self.words, range(self.words.size)))

    def save_dict(self, path):
        np.save(path, self.words)

    @staticmethod
    def split_sentence(sents):
        return [item for sent in sents for item in sent.split(' ')]

    def sent_to_idx(self, sent):
        return [self.word2idx[word] for word in sent]

    def idx_to_sent(self, sent):
        return [self.words[idx] for idx in sent]