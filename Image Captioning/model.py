import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features   

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, batch_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # (n_layers, batch_size, hidden_dim)
        return (torch.zeros(1, self.batch_size, self.hidden_size).cuda(),
                torch.zeros(1, self.batch_size, self.hidden_size).cuda()) 
    
    def forward(self, features, captions):
        captions = captions[:, 0:-1]
        embedded_captions = self.embedding(captions)
        # (seq_len, batch, input_size)
        features = features.unsqueeze(1) #([10, 1, 256])
        embedded_captions = torch.cat((features, embedded_captions), dim=1)                
        output, self.hidden = self.lstm(embedded_captions, self.hidden)
        out = self.fc(output)#if it fails, it could be because this is missing
        return out    
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # make sure the hidden state is clean for every sample
        self.hidden = self.init_hidden()        
        description = list()
        word, self.hidden = self.lstm(inputs, self.hidden)
        out = self.fc(word)
        out_ps = F.softmax(out, dim=2)
        top_p, top_class = out_ps.topk(1, dim=2)        
        wordIndex = top_class[0, 0, 0]
        description.append(int(wordIndex))                
        for i in range(max_len):
            x = wordIndex.unsqueeze(0).unsqueeze(0)
            embedded_word = self.embedding(x)
            word, self.hidden = self.lstm(embedded_word, self.hidden)
            out = self.fc(word)
            out_ps = F.softmax(out, dim=2)
            top_p, top_class = out_ps.topk(1, dim=2)        
            wordIndex = top_class[0, 0, 0]
            description.append(int(wordIndex))
            if(wordIndex == 1):
                return description       
        return description