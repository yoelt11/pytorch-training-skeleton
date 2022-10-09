import torch
from torch import nn

class CustomModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        
        # initialize layers
        self.MLP = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, num_classes)
                    )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = self.MLP(X)
        return self.softmax(X)

# -- testing zone -- #
if __name__ == '__main__':

    # -- set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- parameters
    batch_size, input_dim, hidden_dim, num_classes = 64, 10, 20, 8
    
    #-- create dummy input
    X = torch.randn(batch_size, 25, input_dim).to(device) # [B, Dim1, Dim2]
    
    # -- intiialize model
    model = CustomModel(input_dim, hidden_dim, num_classes).to(device)

    # -- run model
    output = model(X)

    # -- output debug
    print("output shape: ", output.shape)

