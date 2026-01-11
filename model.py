import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    # setting up the model
    def __init__(self, action_dim, hidden_dim=256, observation_shape=None):
        super(Model, self).__init__()

        # CNN layers - 3 is enough
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)

        conv_output_size = self.calculate_conv_output(observation_shape)
        
        print("conv_output_size: ", conv_output_size)

        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, action_dim)

        # initializing weights
        self.apply(self.weights_init)

    # gets called and returns results
    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        output = self.output(x)

        return output

    def calculate_conv_output(self, observation_shape):
        x = torch.zeros(1, *observation_shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x.view(-1).shape[0]

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def save_the_model(self, filename='models/latest.pt'):
        torch.save(self.state_dict(), filename)

    def load_the_model(self, filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(filename))
            print(f"Loaded weights from {filename}")
        except FileNotFoundError:
            print(f"No weights file found at {filename}")

def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        



# CNN - image recognition
# FC layers - 
