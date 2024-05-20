import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import re
import unicodedata
from io import open

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:7" if USE_CUDA else "cpu")

SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

model_name = 'cb_model'
attn_model = 'dot'
#``attn_model = 'general'``
#``attn_model = 'concat'``
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
learning_rate = 0.0002

batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = '/mnt/disk1/Gulshan/rnn/data/save/cb_model/movie-corpus/2-2_500/15000_checkpoint.tar'
MAX_LENGTH= 10

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
    
if loadFilename:
    checkpoint = torch.load(loadFilename)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc_new = checkpoint['voc_dict']

print("wait dude I'm getting ready")
print('loading encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc_new['num_words'], hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size,voc_new['num_words'], decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
encoder = encoder.to(device)
decoder = decoder.to(device)

encoder.eval()
decoder.eval()

print("I'm ready to go!")

clip = 50.0
teacher_forcing_ratio = 1.0
decoder_learning_ratio = 5.0


# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# # If you have CUDA, configure CUDA to call
# for state in encoder_optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()

# for state in decoder_optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # print("input_seq",input_seq) # -> input_seq tensor([[54], [ 2]], device='cuda:7')
        # print("input_seq shape",input_seq.shape) # -> input_seq shape torch.Size([2, 1])
        # print("input_length",input_length)  # -> tensor([2])
        # print("input_length shape",input_length.shape) # -> input_length shape torch.Size([1])
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # print("encoder_outputs shape :",encoder_outputs.shape) # encoder_outputs shape : torch.Size([2, 1, 500])
        # print("encoder_hidden shape ", encoder_hidden.shape) # encoder_hidden shape  torch.Size([4, 1, 500])
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # print("decoder_hidden shape ",decoder_hidden.shape) # -> decoder_hidden shape  torch.Size([2, 1, 500])
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # print("decoder_input ",decoder_input) #  tensor([[1]], device='cuda:7')
        # print("decoder_input shape ",decoder_input.shape)  # decoder_input shape  torch.Size([1, 1])      
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        #this output at the end of the loop so 10 since max lenght 
        # print("\n")
        for _ in range(max_length):
            # print("decoder_input",decoder_input) #> decoder_input tensor([[2]], device='cuda:7')
            # print("decoder_input shape ",decoder_input.shape) #> decoder_input shape  torch.Size([1, 1])
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # print("decoder_output shape",decoder_output.shape) # -> decoder_output shape  torch.Size([1, 7836])
            # print("decoder_output ",decoder_output) # -> decoder_output  tensor([[1.0476e-13, 3.3556e-13, 9.0631e-01,  ..., 3.9148e-13, 9.8597e-14, 2.3664e-12]], device='cuda:7', grad_fn=<SoftmaxBackward0>)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # print("decoder_scores ",decoder_scores) # -> decoder_scores  tensor([0.9063], device='cuda:7', grad_fn=<MaxBackward0>)
            # print("decoder_scores shape",decoder_scores.shape) #-> decoder_scores shape torch.Size([1])
            # print("decoder_input ",decoder_input) #-> decoder_input  tensor([2], device='cuda:7')
            # print("decoder_input shape",decoder_input.shape) # -> decoder_input shape torch.Size([1])
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        #     print("decoder_input :",decoder_input) #-> decoder_input : tensor([[2]], device='cuda:7')
        #     print("decoder_input shape :",decoder_input.shape)#->decoder_input shape : torch.Size([1, 1])
        #     print("\n")
        # print("all_tokens ",all_tokens) # all_tokens  tensor([ 54,  14,   2,  14, 175,  17, 801,  10,   2,   2], device='cuda:7')
        # print("all_tokens shape: ",all_tokens.shape) # ll_tokens shape:  torch.Size([10])
        # print("all_scores ",all_scores) # all_scores  tensor([0.9662, 0.6055, 0.8332, 0.9344, 0.3115, 0.5430, 0.5852, 0.9921, 1.0000, 0.9063], device='cuda:7', grad_fn=<CatBackward0>)
        # print("all_scores shape ",all_scores.shape) # -> all_scores shape  torch.Size([10])
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # print(sentence) # -> hi
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # print(voc['index2word'][54])  # -> hi
    # print(indexes_batch)  # -> [[54, 2]]) # 54 word index , 2 is eos token
     # no shape for indexes_batch (not a tensor)
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # print("lengths",lengths)  # lengths-> tensor([2])
    # print("lengths shape",lengths.shape) # lengths shape torch.Size([1])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1) 
    # print("input_batch shape",input_batch.shape) # input_batch shape -> torch.Size([2, 1]) ie (2,54)
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc['index2word'][token.item()] for token in tokens]
    # print("decoded_words :",decoded_words) #>['hi', '.', 'EOS', '.', 'how', 's', 'business', '?', 'EOS', 'EOS']
    return decoded_words

def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            input_sentence = input('> ')
            if input_sentence == 'BYE' or input_sentence == 'quit': break
            input_sentence = normalizeString(input_sentence)
            # print(input_sentence)
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # print("output_words",output_words) #> ['hi', '.', 'EOS', '.', 'how', 's', 'business', '?', 'EOS', 'EOS']
            # print(output_words[:])
            output_words = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            # print(output_words) #-> ['hi', '.', '.', 'how', 's', 'business', '?']
            
            print('Bot:', ' '.join(output_words))
        except KeyError:
            print("Error: Encountered unknown word.")


def indexesFromSentence(voc, sentence):
    # print([voc['word2index'][word] for word in sentence.split(' ')])
    return [voc['word2index'][word] for word in sentence.split(' ')] + [EOS_token]

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)
evaluateInput(encoder, decoder, searcher, voc_new)