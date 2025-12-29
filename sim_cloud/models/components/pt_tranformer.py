import torch
import torch.nn as nn 
from .helper import Helper
from .multihead import SplitMultiHeadAttention

class CloudTransformer(nn.Module):
    """ Encoder dành cho bài toán điều phối VM.

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        input_dim=3,
        d_model=64,
        num_heads=8
    ):
        super(CloudTransformer, self).__init__()
        
        self.helper = Helper(
            input_dim=input_dim,
            d_model=d_model
        )
        self.attention = SplitMultiHeadAttention(
            embed_dim=d_model,
            num_heads=num_heads
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x,
        split_point=None,
        mask=None
        ):
        
        """
        Docstring for forward
        
        :param self: Description
        :param x:  Tensor đầu vào (Batch_size, Num_PMs, 3)
        :param split_point: Điểm phân tách các nhóm PM (ví dụ: máy rủi ro vs máy ổn định)
        :param mask: Padding mask cho các kịch bản có số lượng PM < 100
        
        """
        # Output shape: (Batch, Num_PMs, d_model)
        embedded = self.helper()
        
        # Bước 2: Chuyển đổi Shape sang (Sequence_length, Batch, d_model) cho Transformer
        # Trong bài toán này, Sequence_length chính là số lượng PMs (2-100)
        
        x_trans = embedded.transpose(0, 1)
        
        if split_point is None:
            split_point = x_trans.size(0) // 2
        
        # Bước 3: Split Multi-Head Attention
        # xử lý việc tính toán Q, K, V nội bộ
        attn_out = self.attention(
            query=x_trans,
            key=x_trans,
            value=x_trans,
            split_point=split_point,
            key_padding_mask=mask
        )
        
        # Bước 4: Residual Connection + Norm
        out = self.norm(x_trans + attn_out).transpose(0,1)
        
        return out