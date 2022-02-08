import torch
from torch import nn
import torchvision.models as models
import copy

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=48, high_res=False):
        super(EmbeddingNet, self).__init__()
        self.high_res = high_res
        if not high_res:
            print("Using the Small architecture for the EmbeddingNet")
            self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(32, 64, 5), nn.PReLU(),
                                        nn.MaxPool2d(2, stride=2))

            self.fc = nn.Sequential(nn.Linear(3136, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, embedding_dim)
                                    )
        else:
            print("Using the AlexNet architecture for the EmbeddingNet")
            model = models.alexnet(pretrained=True)

            pretrained_alexnet_state_dict = copy.deepcopy(model.state_dict())

            # Replaces the fully connected layers
            model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 2048),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, embedding_dim),
            )

            model_sd = model.state_dict()

            # Initialize the convolution layer params with that of the pretrained model
            # except for the first convolution layer due to the different channel nums
            for k, v in model_sd.items():
                if k.startswith("features"):
                    model_sd[k] = pretrained_alexnet_state_dict[k]

            model.load_state_dict(model_sd)
            self.net = model



            # self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
            #                             nn.MaxPool2d(2, stride=2),
            #                             nn.Conv2d(32, 64, 5), nn.PReLU(),
            #                             nn.MaxPool2d(2, stride=2),
            #                             nn.Conv2d(64, 128, 5), nn.PReLU(),
            #                             nn.MaxPool2d(2, stride=2))

            # self.fc = nn.Sequential(nn.Dropout(),
            #                         nn.Linear(128 * 9 * 9, 512),
            #                         nn.PReLU(),
            #                         nn.Dropout(),
            #                         nn.Linear(512, 256),
            #                         nn.PReLU(),
            #                         nn.Linear(256, embedding_dim)
            #                         )

            # self.convnet = nn.Sequential(nn.Conv2d(3, 64, 5, stride=1 ), 
            #                             nn.PReLU(),
            #                             nn.MaxPool2d(3, stride=2),
            #                             nn.Conv2d(64, 192, 5), nn.PReLU(),
            #                             nn.MaxPool2d(2, stride=2),
            #                             nn.Conv2d(192, 256, 3), nn.PReLU(),
            #                             nn.MaxPool2d(2, stride=2))

            # self.fc = nn.Sequential(nn.Dropout(),
            #                         nn.Linear(256 * 9 * 9, 1024),
            #                         nn.PReLU(),
            #                         nn.Dropout(),
            #                         nn.Linear(1024, 256),
            #                         nn.PReLU(),
            #                         nn.Linear(256, embedding_dim)
            #                         )


        

    def forward(self, x):
        if self.high_res:
            output = self.net(x)
        else:
            output = self.convnet(x)
            # print(output.shape)
            output = output.view(output.size()[0], -1)
            output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class CostNet(nn.Module):
    def __init__(self, embedding_dim=48):
        super(CostNet, self).__init__()

        self.fc = nn.Sequential(nn.Linear(embedding_dim, 32),
                                nn.PReLU(),
                                nn.Linear(32, 32),
                                nn.PReLU(),
                                nn.Linear(32, 16),
                                nn.PReLU(),
                                nn.Linear(16, 1)
                                )

    def forward(self, x):
        return self.fc(x)

class PreferenceNet(nn.Module):
    def __init__(self, embedding_dim=48):
        super(PreferenceNet, self).__init__()

        self.fc = nn.Sequential(nn.Linear(embedding_dim * 2, 32),
                                nn.PReLU(),
                                nn.Linear(32, 16),
                                nn.PReLU(),
                                nn.Linear(16, 1)
                                )

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=1)
        return self.fc(concat)

class FullPreferenceNet(nn.Module):
    def __init__(self, embedding_net, preference_net):
        super(FullPreferenceNet, self).__init__()
        self.embedding_net = embedding_net
        self.preference_net = preference_net

    def forward(self, x, y):
        emb_x = self.embedding_net(x)
        emb_y = self.embedding_net(y)
        return self.preference_net(emb_x, emb_y)

# Combines a CostNet with an EmbeddingNet
class FullCostNet(nn.Module):
    def __init__(self, embedding_net, cost_net):
        super(FullCostNet, self).__init__()

        self.embedding_net = embedding_net
        self.cost_net = cost_net

    def forward(self, x):
        emb = self.embedding_net(x)
        return self.cost_net(emb)
class DirectCostNet(nn.Module):
    def __init__(self):
        super(DirectCostNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(3136, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 32),
                                nn.PReLU(),
                                nn.Linear(32, 16),
                                nn.PReLU(),
                                nn.Linear(16, 1)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        return self.fc(output)
