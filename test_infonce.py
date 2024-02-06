import torch
import torch.nn.functional as F

feats = torch.randn(10,4)

feats = F.normalize(feats, p=2, dim=-1)
new_feats = feats + torch.randn(10,4)*0.00
new_feats = F.normalize(new_feats, p=2, dim=-1)

feats = torch.cat([feats, new_feats], dim=0)


print(feats)
# feats = self.convnet(imgs)
# Calculate cosine similarity
cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
print(cos_sim)
# Mask out cosine similarity to itself
self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool)
print(self_mask)
cos_sim.masked_fill_(self_mask, -9e15)
print(cos_sim)
# Find positive example -> batch_size//2 away from the original example
pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
print(pos_mask)
# InfoNCE loss
cos_sim = cos_sim / 1.0
nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
print(nll)
nll = nll.mean()