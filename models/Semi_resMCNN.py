import torch
import torch.nn as nn
import torch.nn.functional as F

class Semi_resMCNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, depth = 32, height = 256, width = 256,training=True):
        super(Semi_resMCNN, self).__init__()
        self.training = training
        # Encoder convolutional layers
        self.encoder1_1 = nn.Conv3d(in_channel, 16, 3, stride=1, padding=1)
        self.encoder1_2 = nn.Conv3d(16, 32, 3, stride=1, padding=1) 
        
        self.encoder2_1 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.encoder2_2 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.encoder2_3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        
        self.encoder3_1 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.encoder3_2 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.encoder3_3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        
        self.encoder4_1 = nn.Conv3d(128, 128, 3, stride=1, padding=1)
        self.encoder4_2 = nn.Conv3d(128, 128, 3, stride=1, padding=1)    
        self.encoder4_3 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
        
        # Decoder convolutional layers
        self.decoder1_upconv = nn.ConvTranspose3d(256, 128, 2, 2)
        self.decoder2_upconv = nn.ConvTranspose3d(128, 64, 2, 2)
        self.decoder3_upconv = nn.ConvTranspose3d(64, 32, 2, 2)
        
        self.decoder1_1 = nn.Conv3d(256, 128, 3, stride=1,padding=1)
        self.decoder1_2 = nn.Conv3d(128, 128, 3, stride=1,padding=1)
        self.decoder1_3 = nn.Conv3d(128, 128, 3, stride=1,padding=1)
        
        self.decoder2_1 = nn.Conv3d(128, 64, 3, stride=1,padding=1)
        self.decoder2_2 = nn.Conv3d(64, 64, 3, stride=1,padding=1)
        self.decoder2_3 = nn.Conv3d(64, 64, 3, stride=1,padding=1)
        
        self.decoder3_1 = nn.Conv3d(64, 32, 3, stride=1,padding=1)
        self.decoder3_2 = nn.Conv3d(32, 32, 3, stride=1,padding=1)
        self.decoder3_3 = nn.Conv3d(32, 32, 3, stride=1,padding=1)

        # Group normalisation layers
        self.group_norm16 = nn.GroupNorm(8,16)
        self.group_norm32 = nn.GroupNorm(8,32)
        self.group_norm64 =nn.GroupNorm(8,64)
        self.group_norm128 =nn.GroupNorm(8,128)
        self.group_norm256 =nn.GroupNorm(8,256)

        # Main output path with resolution 256*256*32 
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4,8),
            nn.Conv3d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4,8),
            nn.Conv3d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4,8),
            nn.Conv3d(8, out_channel, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        # Auxiliary output path 1 with resolution 128*128*16 
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4,8),
            nn.Conv3d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, out_channel, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        # Auxiliary output path 2 with resolution 64*64*8 
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, out_channel, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        # Auxiliary output path 3 with resolution 32*32*4
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, out_channel, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        #Encoder1
        out = F.relu(self.group_norm16(self.encoder1_1(x)))
        out = F.relu(self.group_norm32(self.encoder1_2(out)))
        t1 = out
        
        out = F.max_pool3d(out,2,2)
        
        #Encoder2
        out_branch = out
        out = F.relu(self.group_norm32(self.encoder2_1(out)))
        out = F.relu(self.group_norm32(self.encoder2_2(out)))
        out = out_branch + out
        out = F.relu(self.group_norm64(self.encoder2_3(out)))
        t2 = out
        
        out = F.max_pool3d(out,2,2)
        
        #Encoder3
        out_branch = out        
        out = F.relu(self.group_norm64(self.encoder3_1(out)))
        out = F.relu(self.group_norm64(self.encoder3_2(out)))
        out = out_branch + out
        out = F.relu(self.group_norm128(self.encoder3_3(out)))
        t3 = out
        
        out = F.max_pool3d(out,2,2)
        
        #Encoder4(Bridge)
        out_branch = out        
        out = F.relu(self.group_norm128(self.encoder4_1(out)))
        out = F.relu(self.group_norm128(self.encoder4_2(out)))
        out = out_branch + out
        out = F.relu(self.group_norm256(self.encoder4_3(out)))
        
        output1 = self.map1(out) # 4*32*32 output path
        
        #Decoder1       
        out = F.relu(self.decoder1_upconv(out))
        out = torch.cat((out,t3),1)
        out = F.relu(self.group_norm128(self.decoder1_1(out)))
        out_branch = out    
        out = F.relu(self.group_norm128(self.decoder1_2(out)))
        out = self.group_norm128(self.decoder1_3(out))
        out = out_branch + out
        out = F.relu(out)
        output2 = self.map2(out) # 8*64*64 output path
        
        #Decoder2
        out = F.relu(self.decoder2_upconv(out))
        out = torch.cat((out,t2),1)
        out = F.relu(self.group_norm64(self.decoder2_1(out)))
        out_branch = out 
        out = F.relu(self.group_norm64(self.decoder2_2(out)))
        out = self.group_norm64(self.decoder2_3(out))
        out = out_branch + out
        out = F.relu(out)
        output3 = self.map3(out) # 16*128*128 output path
                
        #Decoder3  
        out = F.relu(self.decoder3_upconv(out))
        out = torch.cat((out,t1),1)
        out = F.relu(self.group_norm32(self.decoder3_1(out)))
        out_branch = out  
        out = F.relu(self.group_norm32(self.decoder3_2(out)))
        out = self.group_norm32(self.decoder3_3(out))
        out = out_branch + out
        out = F.relu(out)
        output4 = self.map4(out) # 32*256*256 output path
        
        if self.training is True:
            return [output1, output2, output3, output4] # multi-resolutional outputs
        else:
            return output4