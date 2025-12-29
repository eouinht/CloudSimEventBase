import math
import torch
from torch import Tensor
from typing import Optional, Tuple, List
from torch.nn.functional import linear, softmax, dropout, pad

def _in_projection_packed(
    q: Tensor, 
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor]=None)-> List[Tensor]:
    """ Phép chiếu đóng gói (giúp tính toán song song Q, K, V hiệu quả hơn).

    Returns:
        List[Tensor]: _description_
    """
    E = q.size(-1)
    if k is v and q is k:
        return linear(q, w, b).chunk(3, dim=-1) # Self-attention
    w_q, w_k, w_v = w.chunk(3)
    b_q, b_k, b_v = b.chunk(3) if b is not None else (None, None, None)
    return [linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)]
    
def _scaled_dot_product_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float=0.0) -> Tuple[Tensor,Tensor]:
    """Tích vô hướng có định tỉ lệ (giúp ổn định giá trị Attention).

    Args:
        q (Tensor): _description_
        k (Tensor): _description_
        v (Tensor): _description_
        attn_mask (Optional[Tensor], optional): _description_. Defaults to None.
        dropout_p (float, optional): _description_. Defaults to 0.0.

    Returns:
        Tuple[Tensor,Tensor]: _description_
    """
    B, Nt, E = q.shape
    q = q/math.sqrt(E)
    attn = torch.bmm(q, k.transpose(-2, -1))                            
    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)
    if(dropout_p > 0.0):
        attn = dropout(attn, p=dropout_p)
    return torch.bmm(attn,v), attn