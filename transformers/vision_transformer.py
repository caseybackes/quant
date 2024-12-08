# https://www.youtube.com/watch?v=ovB0ddFtzzA

import torch 
import torch.nn as nn



class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.
    Parameters:
        img_size (int): input image size
        patch_size (int): patch size
        in_chans (int): number of input image channels
        embed_dim (int): number of linear projection output channels
    Attributes:
        n_patches (int): number of patches inside a single image
        proj (nn.Conv2d): the linear projection (convolution) layer
    """
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
            
    def forward(self, x):
        """Forward function.
        Args:
            x (tensor): input tensor.
        Returns:
            tensor: [batch_size, n_patches, embed_dim]
        """
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x 

class Attention(nn.Module):
    """
    Parameters:
        dim (int): 
            the number of input channels
        n_heads (int): 
            number of attention heads
        qkv_bias (bool): 
            if True, add bias to q, k, v
        attn_p (float): 
            dropout probability applied to attn
        proj_p (float): 
            dropout probability applied to output
    Attributes:
        scale (float): 
            scaling factor for attention
        qkv (nn.Linear): 
        linear layer for q, k, v
        proj (nn.Linear): 
        linear layer for output
        attn_drop, proj_drop (nn.Dropout): 
            dropout layers
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Forward function.
        Args:
            x (tensor): input tensor.
        Returns:
            tensor: [batch_size, n_patches + 1, embed_dim]
        """
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Input dim {dim} should match layer dim {self.dim}")
        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)
        dp = (q @ k_t) * self.scale # n_samples x n_heads x n_tokens x n_tokens
        attn = dp.softmax(dim=-1) # n_samples x n_heads x n_tokens x n_tokens
        attn = self.attn_drop(attn) # n_samples x n_heads x n_tokens x n_tokens
        weighted_avg = attn @ v # n_samples x n_heads x n_tokens x head_dim
        weighted_avg = weighted_avg.transpose(1, 2)
        weighted_avg = weighted_avg.flatten(2)
        x= self.proj(weighted_avg)
        x = self.proj_drop(x) # n_samples x n_tokens x dim
        return x 

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)    
    
    
    def forward(self, x):
        x = self.fc1(x) # n_samples x n_patches+1 x hidden_features
        x = self.act(x)  # n_samples x n_patches+1 x hidden_features
        x = self.drop(x) # n_samples x n_patches+1 x hidden_features
        x = self.fc2(x)  # n_samples x n_patches+1 x out_features
        x = self.drop(x) # n_samples x n_patches+1 x out_features
        return x
    
class Block(nn.Module):
    """
    Parameters:
        dim (int): 
            the number of input channels
        n_heads (int): 
            number of attention heads
        mlp_ratio (int): 
            ratio of mlp hidden dim to embedding dim, decides hidden dim of mlp
        qkv_bias (bool): 
            if True, add bias to q, k, v
        attn_p (float): 
            dropout probability applied to attn
        proj_p (float): 
            dropout probability applied to output

    Attributes:
        norm1, norm2 (nn.LayerNorm): 
            layer normalization
        attn (Attention): 
            attention module
        mlp (MLP): 
            mlp module
    """

    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=proj_p
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )
    
    def forward(self, x):
        """Forward function.
        Args:
            x (tensor): input tensor.
        Returns:
            tensor: [batch_size, n_patches + 1, embed_dim]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    """
    Parameters:
        img_size (int): 
            input image size
        patch_size (int): 
            patch size
        in_chans (int): 
            number of input image channels
        class_dim (int): 
            number of classes
        embed_dim (int): 
            number of linear projection output channels
        depth (int): 
            number of transformer layers 
        n_heads (int): 
            number of attention heads
        mlp_ratio (int): 
            ratio of mlp hidden dim to embedding dim, decides hidden dim of mlp
        qkv_bias (bool): 
            if True, add bias to q, k, v
        attn_p (float): 
            dropout probability applied to attn
        proj_p (float): 
            dropout probability applied to output

    Attributes:
        patch_embed (PatchEmbed): 
            patch embedding module
        cls_token (nn.Parameter): 
            learnable position embedding
        pos_embed (nn.Parameter): 
            positional embedding
        pos_drop (nn.Dropout): 
            positional dropout
        blocks (nn.ModuleList): 
            transformer blocks
        norm (nn.LayerNorm): 
            layer normalization
        head (nn.Linear): 
            head layer
    """

    def __init__(
            self, 
            img_size=224, 
            patch_size=16,
            in_chans=3,
            n_classes=1000,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        self.init_weights()

    def forward(self, x):
        """Forward function.
        Args:
            x (tensor): input tensor.
            Shape: [batch_size, in_chans, img_size, img_size]
        Returns:
            tensor: [batch_size, n_classes]
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1) # n_samples x (n_patches + 1) x embed_dim 
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)  
        return x

    def init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for block in self.blocks:
            for p in block.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                

if __name__=="__main__":
    custom_config = {
        'img_size': 384,
        'in_chans': 3,
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'n_heads': 12,
        'qkv_bias': True,
        'mlp_ratio': 4
    }
    import timm 
    model = VisionTransformer(**custom_config)

    def get_n_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    param_count = get_n_params(model)
    
    if param_count >= 1e9:
        param_suffix = "B"
        param_count /= 1e9
    elif param_count >= 1e6:
        param_suffix = "M"
        param_count /= 1e6
    elif param_count >= 1e3:
        param_suffix = "K"
        param_count /= 1e3
    else:
        param_suffix = ""

    print(f"Number of parameters: {param_count:.0f}{param_suffix}")    

    model_official = timm.create_model('vit_base_patch16_384', pretrained=True)

    for (n_o, p_o), (n_c, p_c) in zip(model_official.named_parameters(), model.named_parameters()):
        assert p_o.numel() == p_c.numel()
        print(F"{n_o} | {n_c}")

        p_c.data[:] = p_o.data

    inp = torch.rand(1, 3, 384, 384)
    res_c = model(inp)
    res_o = model_official(inp)
    assert get_n_params(model) == get_n_params(model_official)

    k = 10 
    import json 
    from collections import OrderedDict
    with open('image_net_info.json', 'r') as f:
        data = json.load(f)
        labels = data['id2label']
        label_dict = OrderedDict({int(k): v for k, v in labels.items()})

    model = torch.load('model.pth')
    model.eval()
