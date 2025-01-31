import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, task, dropout=0.):
        super().__init__()
        self.task = task
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out_ff = self.net(x)
        return out_ff


class Attention(nn.Module):
    def __init__(self, dim, task, heads=8, dim_head=64, dropout=0., args=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.task = task

        self.attend = nn.Softmax(dim=-1)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_v = nn.Linear(dim, inner_dim * 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # self.chn = args.chn #

    def forward(self, x):
        qk = self.to_qk(x).chunk(2, dim=-1)
        v = self.to_v(x)
        # print("after to_v, v shape: {}".format(v.shape))
        list_qk = list(qk)
        list_qk.append(v)
        qkv = tuple(list_qk)
        # print("attention function")
        # print(len(qkv))
        # print("qkv shape: {}{}{}".format(qkv.shape))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # print("q shape: {}".format(q.shape))
        # print("k shape: {}".format(k.shape))
        # print("v shape: {}".format(v.shape))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # print("dots shape: {}".format(dots.shape))
        attn = self.attend(dots)
        # print("attn shape: {}".format(attn.shape))

        out = torch.matmul(attn, v)
        # print("out shape:{}".format(out.shape))
        # print(out[:,0,:3,:10])

        # print("control_head:")
        # print(control_head)

        # if control_head_fixed == True:
        #     print("<<<<<fixed killing head<<<<")
        #     out[:, 3, :, :] = 0

        # if control_head == True:
        #     print("~~~killing head~~~~")
        #     out[:, self.chn, :, :] = 0
        # print("out shape: {}".format(out.shape))

        # print(out[:, 0, :3, :10])
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print("rearrange: {}".format(out.shape))

        output = self.to_out(out)
        # print("to_out: {}".format(output.shape))

        return output


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, task, dropout=0., args=None):  # task 추가
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, task, heads=heads, dim_head=dim_head, dropout=dropout, args=args)),  # task 추가
                PreNorm(dim, FeedForward(dim, mlp_dim, task, dropout=dropout))  # task 추가
            ]))

        # self.chd = args.chd

    def forward(self, x):
        # global control_head  #
        # # global control_head_fixed  #
        # # global chd
        # # global chn
        # control_head = False  #
        # # control_head_fixed = False  #
        # layers_depth = 0  #
        # for attn, ff in self.layers:
        #     print("detph: {}".format(layers_depth))  #
        #     # if layers_depth == 0: #control_head_fixed depth
        #     #     print("found the depth_fixed") #
        #     #     control_head_fixed = True #
        #     if layers_depth == self.chd:  ### control_head depth
        #         print("found the depth")  #
        #         control_head = True  #
        #     x = attn(x) + x
        #     x = ff(x) + x
        #     control_head = False  #
        #     # control_head_fixed = False  #
        #     layers_depth += 1  #
        # return x

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

            # x = attn(x) # no shortcut ver.
            # x = ff(x) # no shortcut ver.
            # #print("no this is real no shortcut!!")

            # x1 = attn(x) # big skip ver.
            # x2 = ff(x1) + x # big skip ver.
            # x = ff(attn(x)) + x # big skip ver. (worked)
        return x


class ViT_designing_clear(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, task, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., args=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        #for imagenet(cropped 224*224)
        # image_height = 224
        # image_width = 224

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
            # nn.Linear(48, 48*2) # for concated case
        )

        #print(f"num_patches:{num_patches}, dim:{dim}")

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  # num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, task, dropout, args)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.to_image_again = nn.Sequential(  # patch embedding의 역관계 느낌으로...
            nn.Linear(dim, patch_dim),  # [256, 65, 512] --> [256, 65, p1*p2*c]
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_height // patch_height,
                      w=image_width // patch_width, p1=patch_height, p2=patch_width),
        )

        # for non negative denoising:
        self.relu_denoising_head = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )

        self.task = task

        # self.chd, self.chn = args.chd, args.chn

    def forward(self, img):
        #print("input img shape: {}".format(img.shape))
        x = self.to_patch_embedding(img)
        #print("x) after patch embedding: {}".format(x.shape))
        # x = self.use_concated_pe(x) # for concated pe
        # print(f"x) after patch embedding2: {x.shape}")
        b, n, _ = x.shape

        # print(f"x.shape (b, n,): {b}, {n}....{x.shape}")
        # print(f"pos_embedding[:, :(n+1)]: {self.pos_embedding[:,:(n+1)].shape}")
        # print(f"pos_embedding[:, :]: {self.pos_embedding[:, :].shape}")
        # print(f"pos_embedding[:, :(n)]: {self.pos_embedding[:, :(n)].shape}")
        # print(f"pos_embedding: {self.pos_embedding.shape}")


        x += self.pos_embedding[:, :n] # :(n + 1), #imagenet: :n
        #print("x) after pos_embedding: {}".format(x.shape))
        x = self.dropout(x)
        #print("x) after drop out: {}".format(x.shape))

        x = self.transformer(x)
        #print("x) after transformer: {}".format(x.shape))

        if self.task in ['cls', 'stl_10_cls', 'cifar10', 'cifar100', 'imagenet_cls']:  # 'cifar10':
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            #print("x) after mean: {}".format(x.shape))

            x = self.to_latent(x)
            #print("x) after mean: {}".format(x.shape))

            x = self.mlp_head(x)
            #print("x) after class head: {}".format(x.shape))

        elif self.task in ['clear', 'denoising100', 'imagenet_deno', 'clear2clear']:
            x = self.to_image_again(x)
            #print("x) after to_image_again: {}".format(x.shape))
            # check...
            x = self.relu_denoising_head(x)
            #print("x) after relu_denoising head: {}".format(x.shape))

        return x
