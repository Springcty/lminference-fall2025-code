from transformers import LogitsProcessor
import torch

class OddSamplingLogitsProcessor(LogitsProcessor):
    """
    This LogitsProcessor masks out all tokens except those with odd-numbered ranks
    (1st, 3rd, 5th, ...) according to their logits.
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: (batch_size, vocab_size)
        sorted_indices = torch.argsort(scores, dim=-1, descending=True)
        # Create a mask for odd-numbered ranks (0-based: 0,2,4,...)
        odd_rank_mask = torch.zeros_like(scores, dtype=torch.bool)
        batch_size, vocab_size = scores.shape
        for i in range(batch_size):
            # Odd-numbered ranks: indices 0,2,4,...
            odd_indices = sorted_indices[i, ::2]
            odd_rank_mask[i].scatter_(0, odd_indices, True)
        # Set logits of even-ranked tokens to -inf
        scores = scores.masked_fill(~odd_rank_mask, float('-inf'))
        return scores