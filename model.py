import jax
import jax.numpy as jnp
from flax import linen as nn

class MultiHeadAttention(nn.Module):
    d_in: int
    d_out: int
    block_size: int
    dropout: float
    num_heads: int
    qkv_bias: bool = False

    @nn.compact
    def __call__(self, x):
        assert self.d_out % self.num_heads == 0,         
        head_dim = self.d_out // self.num_heads
        W_query = nn.Dense(self.d_out, use_bias=self.qkv_bias)
        W_key = nn.Dense(self.d_out, use_bias=self.qkv_bias)
        W_value = nn.Dense(self.d_out, use_bias=self.qkv_bias)
        out_proj = nn.Dense(self.d_out)

        b, num_tokens, _ = x.shape

        queries = W_query(x)
        keys = W_key(x) 
        values = W_value(x)

        queries = jnp.reshape(queries, (b, num_tokens, self.num_heads, head_dim))
        keys = jnp.reshape(keys, (b, num_tokens, self.num_heads, head_dim))
        values = jnp.reshape(values, (b, num_tokens, self.num_heads, head_dim))

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        attn_scores = jnp.einsum('bnqd,bnkd->bnqk', queries, keys)
        mask_bool = jnp.triu(jnp.ones((self.block_size, self.block_size), dtype=bool), k=1)
        mask_bool = mask_bool[:num_tokens, :num_tokens]
        mask_unsqueezed = jnp.expand_dims(jnp.expand_dims(mask_bool, 0), 0)
        attn_scores = jax.lax.select(mask_unsqueezed, jnp.full_like(attn_scores, -jnp.inf), attn_scores)

        attn_weights = jax.nn.softmax(attn_scores / jnp.sqrt(head_dim), axis=-1)
        attn_weights = nn.Dropout(self.dropout)(attn_weights, deterministic=False)

        context_vec = jnp.einsum('bnqk,bnkd->bnqd', attn_weights, values)
        context_vec = context_vec.transpose(0, 2, 1, 3)
        context_vec = jnp.reshape(context_vec, (b, num_tokens, self.d_out))

        context_vec = out_proj(context_vec)

        return context_vec
