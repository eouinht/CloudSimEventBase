import torch.nn as nn
import torch
from .functional import _in_projection_packed, _scaled_dot_product_attention
class SplitMultiHeadAttention(nn.Module):
    """
    kế thừa logic Split Point đặc thù để xử lý các cụm máy chủ không đồng nhất.
    """
    def __init__(self, embed_dim=64, num_heads=8):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Bifurcated weights
        # Các bộ trọng số được phân nhánh (trong in_proj_weight1 và 2).
        
        self.i_proj_w1 = nn.Parameter(torch.empty(3*embed_dim, embed_dim))
        self.i_proj_b1 = nn.Parameter(torch.empty(3*embed_dim))
        self.i_proj_w2 = nn.Parameter(torch.empty(3*embed_dim, embed_dim))
        self.i_proj_b2 = nn.Parameter(torch.empty(3*embed_dim))

        self.o_proj1 = nn.Linear(embed_dim, embed_dim)
        self.o_proj2 = nn.Linear(embed_dim, embed_dim)
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.i_proj_w1)
        nn.init.xavier_uniform_(self.i_proj_w2)
    
    def forward(
        self,
        query,
        key,
        value,
        split_point,
        key_padding_mask=None):
        
        tgt_len , bsz, embed_dim = query.shape
        # Doing In-projection for two group PM 
        q1, k1, v1 = _in_projection_packed(
            query[:split_point],
            key[:split_point],
            value[:split_point],
            self.i_proj_w1,
            self.i_proj_b1)
        q2, k2, v2 = _in_projection_packed(
            query[:split_point],
            key[:split_point],
            value[:split_point],
            self.i_proj_w2,
            self.i_proj_b2)
        
        # Ghép và Reshape để tính Multi-head
        
        q = torch.cat([q1, q2], dim=0).contiguous().view(
            tgt_len,
            bsz*self.num_heads, 
            self.head_dim
        ).transpose(0, 1)
        
        k = torch.cat([k1, k2], dim=0).contiguous().view(
            -1,
            bsz*self.num_heads,
            self.head_dim
        ).transpose(0,1)
        
        v = torch.cat([v1, v2], dim=0).contiguous().view(
            -1,
            bsz*self.num_heads, 
            self.head_dim
        ).transpose(0,1)
        
        attn_output, _ =  _scaled_dot_product_attention(q, k, v)
        
        attn_output = attn_output.transpose(0,1).contiguous().view(
            tgt_len,
            bsz,
            embed_dim
        )
        
        res1 = self.o_proj1(attn_output[:split_point])
        res2 = self.o_proj2(attn_output[:split_point])
        
        return torch.cat([res1, res2], dim=0)
        