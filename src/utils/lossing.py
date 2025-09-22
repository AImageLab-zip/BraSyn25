import torch
import torch.nn.functional as F
def dice_loss(reconstructed:torch.Tensor, original:torch.Tensor):
        '''
        reconstructed --> (B, 4, 240,240) 
        original --> (B, 1, 240,240)
        the four channels represent the 4 segmentation classes:
        0 --> background , useless
        1 --> tumor core
        2 --> edema
        3 --> enhancing

        the original is already argmaxed
        reconstructed must be softmaxed first
        '''
        num_classes = reconstructed.size(1)
        eps = 1e-3

        reconstructed = F.softmax(reconstructed, dim=1)

        #removing the background
        reconstructed = reconstructed[:,1:]

        original = F.one_hot(original.squeeze(1).long(), num_classes=num_classes)
        original = original.permute(0, 3, 1, 2).float() # now original is --> (B, 4, 240,240) 

        #removing the background
        original = original[:,1:]
        
        # Calculate dice for each class
        intersection = (reconstructed * original).sum(dim=(2, 3))
        union = reconstructed.sum(dim=(2, 3)) + original.sum(dim=(2, 3))
        
        dice = (2 * intersection + eps) / (union + eps)


        return 1 - dice
