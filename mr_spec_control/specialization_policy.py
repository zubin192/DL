import torch
import torch.nn as nn
import torch.optim as optim


class TransformerModel(nn.Module):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 seq_len,
                 d_model=128,
                 nhead=4,
                 num_layers=2,
                 dropout=0.1
                 ):
        super(TransformerModel, self).__init__()

        # Embeddings, positional encoding
        self.embedding_obs = nn.Linear(obs_dim, d_model)
        self.embedding_act = nn.Linear(act_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)

        # Output
        self.output_layer = nn.Linear(d_model, act_dim)

    def forward(self, obs, act):
        obs = self.embedding_obs(obs) + self.positional_encoding  # Add positional encoding
        act = self.embedding_act(act) + self.positional_encoding  # Add positional encoding

        memory = self.transformer_encoder(obs)
        output = self.transformer_decoder(act, memory)
        output = self.output_layer(output)

        return output

# Example usage
obs_dim = 16  # Example observation dimension
act_dim = 8   # Example action dimension
seq_len = 10  # Example sequence length

model = TransformerModel(obs_dim, act_dim, seq_len)

# Example input (batch_size=32, seq_len=10, obs_dim=16)
example_obs = torch.randn(32, seq_len, obs_dim)
example_act = torch.randn(32, seq_len, act_dim)
output = model(example_obs, example_act)
print(output.shape)  # Expected output shape: (32, seq_len, act_dim)



class MultiHeadPolicy(nn.Module):

    def __init__(self,
                 obs_size,
                 num_agents, # num heads
                 num_modes, # num modalities
                 hidden_size=16,
                 device='cpu',
                 ):

        super(MultiHeadPolicy, self).__init__()
        self.num_agents = num_agents
        self.num_modes = num_modes

        # Define the network layers
        self.fc1 = nn.Linear(obs_size, hidden_size, device=device)
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, hidden_size, device=device)
        # Multi-head policy here
        self.heads = [nn.Linear(hidden_size, num_modes, device=device) for _ in range(num_agents)]

        self.softmax0 = nn.Softmax(dim=0)
        self.softmax1 = nn.Softmax(dim=1)

        self.fitness = None

    def forward(self, obs):

        # Pass data through the layers
        x = self.fc1(obs)  # Linear transformation
        x = self.relu(x)  # Apply ReLU activation
        # x = self.fc2(x)  # Final linear transformation
        # x = self.relu(x)  # Apply ReLU activation
        heads_x = [head(x) for head in self.heads]
        # print("Head shape:", len(heads_x[0].shape))
        if len(heads_x[0].shape) > 1: # for batch dim > 1
            actions = [self.softmax1(i) for i in heads_x]
        else:
            actions = [self.softmax0(i) for i in heads_x]

        # print("Raw heads:\n", heads_x, "\nActions:\n", actions)

        return actions
