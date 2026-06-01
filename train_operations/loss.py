import hashlib
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from typing import Optional, List

try:
    import torch.distributed.nn
    has_distributed = True
except ImportError:
    has_distributed = False


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get the number of processes in distributed training."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of the current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class GatherFeatures(torch.autograd.Function):
    """
    Gather tensors from all processes and support backward propagation.
    
    Based on OpenCLIP's implementation for distributed contrastive learning.
    When gather_with_grad=True, gradients flow through all gathered tensors
    via the replacement trick in forward (local chunk replaced with original).
    When gather_with_grad=False, only local tensors receive gradients.
    """
    
    @staticmethod
    def forward(ctx, tensor, gather_with_grad=False):
        ctx.gather_with_grad = gather_with_grad
        
        if not is_dist_avail_and_initialized():
            return tensor
        
        world_size = get_world_size()
        rank = get_rank()
        
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        
        if gather_with_grad:
            # Replace the local tensor with the original to preserve gradients.
            # Autograd tracks this replacement, so backward gets correct grads
            # without needing any all_reduce.
            gathered_tensors[rank] = tensor
        
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.batch_size = tensor.shape[0]
        
        return torch.cat(gathered_tensors, dim=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        # In both cases, each rank only needs its own slice of the gradient.
        # For gather_with_grad=True, the autograd graph through the replaced
        # tensor handles gradient flow — no all_reduce needed.
        grad_input = grad_output.narrow(
            0,
            ctx.rank * ctx.batch_size,
            ctx.batch_size
        ).contiguous()
        return grad_input, None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
):
    """
    Gather image and text features from all distributed processes.
    Based on OpenCLIP's implementation for distributed contrastive learning.
    
    Args:
        image_features: (N, D) tensor of image features from local batch
        text_features: (N, D) tensor of text features from local batch
        local_loss: If True, only use local features for query side (memory efficient)
        gather_with_grad: If True, use torch.distributed.nn.all_gather which supports grad
        rank: Current process rank (global rank, not local)
        world_size: Total number of processes
        
    Returns:
        all_image_features: (N * world_size, D) gathered image features
        all_text_features: (N * world_size, D) gathered text features
    """
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        
        if not local_loss:
            # Replace local chunk with original tensor to preserve gradients
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
            
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


def _gather_mask(mask, world_size, gather_with_grad=False):
    """Gather boolean mask from all GPUs. Consolidates duplicated logic."""
    if world_size <= 1:
        return mask
    if gather_with_grad:
        return torch.cat(torch.distributed.nn.all_gather(mask), dim=0)
    else:
        gathered = [torch.zeros_like(mask) for _ in range(world_size)]
        dist.all_gather(gathered, mask)
        return torch.cat(gathered, dim=0)


def _deterministic_hash(s: str) -> int:
    """Deterministic hash that is consistent across processes and Python sessions.
    
    Python's built-in hash() is randomized via PYTHONHASHSEED, so two GPU
    processes hashing the same string can produce different integers.
    We use MD5 (truncated to 64 bits) for cross-process consistency.
    """
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:16], 16)


class ClipLoss(nn.Module):
    """
    Standard CLIP contrastive loss with distributed training support.
    Based on OpenCLIP's implementation with multi-positive extension.
    
    Computes bidirectional contrastive loss:
    - Image-to-text: For each image, find the matching text
    - Text-to-image: For each text, find the matching image
    
    With distributed training, features are gathered across all GPUs to enlarge
    the negative sample pool.
    
    Args:
        local_loss: If True, compute loss using local queries against global keys.
        gather_with_grad: If True, gradients flow through all gathered features.
        cache_labels: If True, cache ground truth labels for efficiency.
        rank: Current process global rank in distributed training.
        world_size: Total number of distributed processes.
    """

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        """Get ground truth labels for contrastive loss."""
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        """Compute similarity logits between image and text features."""
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
            )
            # print(f"[DEBUG rank={self.rank}] local: {image_features.shape[0]}, gathered: {all_image_features.shape[0]}, expected: {image_features.shape[0] * self.world_size}") # ===========================


            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            mask=None,
            concurrent_ids=None,
            output_dict=False,
    ):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale, logit_bias=logit_bias,
        )
        
        # Handle multi-positive case
        if concurrent_ids is not None:
            return self._forward_multi_positive(
                logits_per_image, logits_per_text,
                concurrent_ids, mask, device, output_dict
            )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        if mask is not None:
            # Gather mask once (consolidated from duplicated branches)
            all_mask = _gather_mask(mask, self.world_size, self.gather_with_grad)
            local_mask = mask  # For local_loss query filtering
            
            logits_per_image = torch.nan_to_num(logits_per_image, nan=-100.0)
            logits_per_text = torch.nan_to_num(logits_per_text, nan=-100.0)
            
            # Mask out invalid keys (columns)
            logits_per_image = logits_per_image.masked_fill(~all_mask.unsqueeze(0), -100.0)
            logits_per_text = logits_per_text.masked_fill(~all_mask.unsqueeze(0), -100.0)
            
            # Determine valid queries
            query_mask = local_mask if (self.world_size > 1 and self.local_loss) else all_mask
            
            if query_mask.sum() == 0:
                zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
                return {"contrastive_loss": zero_loss} if output_dict else zero_loss
            
            loss_img = F.cross_entropy(logits_per_image[query_mask], labels[query_mask], reduction='sum')
            loss_txt = F.cross_entropy(logits_per_text[query_mask], labels[query_mask], reduction='sum')
            
            num_valid = query_mask.sum()
            total_loss = (loss_img + loss_txt) / (2 * num_valid)
        else:
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

    def _forward_multi_positive(
            self,
            logits_per_image,
            logits_per_text,
            concurrent_ids,
            mask,
            device,
            output_dict=False,
    ):
        """
        Compute multi-positive contrastive loss (SupCon-style) for CLIP.
        
        Uses deterministic hashing for cross-GPU consistency.
        Empty/invalid IDs are assigned unique labels so they don't form
        false positive clusters.
        """
        batch_size = logits_per_image.shape[0]
        
        # Convert concurrent_ids to tensor labels
        if isinstance(concurrent_ids, torch.Tensor):
            local_labels = concurrent_ids
        else:
            # Deterministic hash for cross-GPU consistency
            # Empty strings get unique negative labels to avoid false positive clustering
            labels_list = []
            empty_counter = -1
            for cid in concurrent_ids:
                if cid and cid.strip():
                    labels_list.append(_deterministic_hash(cid) % (2**62))
                else:
                    # Assign unique label so empty IDs don't match each other
                    labels_list.append(empty_counter)
                    empty_counter -= 1
            local_labels = torch.tensor(labels_list, device=device, dtype=torch.long)
        
        # Gather labels across GPUs
        if self.world_size > 1:
            gathered_labels = [torch.zeros_like(local_labels) for _ in range(self.world_size)]
            dist.all_gather(gathered_labels, local_labels)
            all_labels = torch.cat(gathered_labels, dim=0)
        else:
            all_labels = local_labels
        
        # Build positive mask: mask[i,j] = True if labels[i] == labels[j]
        positive_mask = (all_labels.unsqueeze(0) == all_labels.unsqueeze(1)).float()
        num_positives = positive_mask.sum(dim=1)
        
        # Handle validity mask
        if mask is not None:
            all_mask = _gather_mask(mask, self.world_size, self.gather_with_grad)
                
            logits_per_image = torch.nan_to_num(logits_per_image, nan=-100.0)
            logits_per_text = torch.nan_to_num(logits_per_text, nan=-100.0)
            logits_per_image = logits_per_image.masked_fill(~all_mask.unsqueeze(0), -100.0)
            logits_per_text = logits_per_text.masked_fill(~all_mask.unsqueeze(0), -100.0)
            # Zero out positives involving invalid samples
            positive_mask = positive_mask * all_mask.unsqueeze(0).float() * all_mask.unsqueeze(1).float()
            num_positives = positive_mask.sum(dim=1)
            validity_mask = all_mask
        else:
            validity_mask = None
        
        loss_img = self._multi_positive_loss(logits_per_image, positive_mask, num_positives, validity_mask)
        loss_txt = self._multi_positive_loss(logits_per_text, positive_mask.T, num_positives, validity_mask)
        
        total_loss = (loss_img + loss_txt) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss

    def _multi_positive_loss(self, logits, positive_mask, num_positives, validity_mask=None):
        """Compute multi-positive contrastive loss."""
        # Numerical stability: subtract max
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stable = logits - logits_max.detach()
        
        # Log-softmax
        exp_logits = torch.exp(logits_stable)
        log_prob = logits_stable - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean log probability over positives
        num_positives_safe = torch.clamp(num_positives, min=1.0)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / num_positives_safe
        
        if validity_mask is not None:
            valid_anchors = validity_mask
        else:
            valid_anchors = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return -mean_log_prob_pos[valid_anchors].mean()


class InfoNCE(nn.Module):
    """
    InfoNCE contrastive loss with distributed training support.

    Args:
        temperature: Logits are divided by temperature before cross entropy.
        reduction: 'none', 'sum', or 'mean'.
        negative_mode: 'paired' or 'unpaired'.
        use_distributed: If True, gather features across all GPUs.
        local_loss: If True, local queries against global keys.
        gather_with_grad: If True, enable gradient flow through gathered features.
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired',
                 use_distributed=False, local_loss=False, gather_with_grad=False):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.use_distributed = use_distributed
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad

    def forward(self, query, positive_key, negative_keys=None, mask=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode,
                        mask=mask,
                        use_distributed=self.use_distributed,
                        local_loss=self.local_loss,
                        gather_with_grad=self.gather_with_grad)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', 
             negative_mode='unpaired', mask=None, use_distributed=False, local_loss=False, 
             gather_with_grad=False):
    """
    Compute InfoNCE contrastive loss with optional distributed training support.
    """
    # Input validation
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    
    # Distributed info
    world_size = get_world_size() if use_distributed else 1
    rank = get_rank() if use_distributed else 0
    local_batch_size = len(query)
    
    if negative_keys is not None:
        # Explicit negative keys
        if use_distributed and world_size > 1:
            if gather_with_grad:
                all_negative_keys = torch.cat(torch.distributed.nn.all_gather(negative_keys), dim=0)
            else:
                gathered_negatives = [torch.zeros_like(negative_keys) for _ in range(world_size)]
                dist.all_gather(gathered_negatives, negative_keys)
                all_negative_keys = torch.cat(gathered_negatives, dim=0)
        else:
            all_negative_keys = negative_keys

        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            negative_logits = query @ transpose(all_negative_keys)
        elif negative_mode == 'paired':
            query_expanded = query.unsqueeze(1)
            negative_logits = query_expanded @ transpose(all_negative_keys)
            negative_logits = negative_logits.squeeze(1)

        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Implicit negatives (off-diagonal)
        if use_distributed and world_size > 1:
            all_query, all_positive_key = gather_features(
                query, positive_key,
                local_loss=local_loss,
                gather_with_grad=gather_with_grad,
                rank=rank,
                world_size=world_size,
            )
            
            if local_loss:
                logits = query @ transpose(all_positive_key)
                labels = torch.arange(local_batch_size, device=query.device) + rank * local_batch_size
                
                if mask is not None:
                    all_mask = _gather_mask(mask, world_size, gather_with_grad)
                    logits = torch.nan_to_num(logits, nan=-100.0)
                    logits = logits.masked_fill(~all_mask.unsqueeze(0), -100.0)
                    
                    if mask.sum() == 0:
                        return torch.tensor(0.0, device=query.device, requires_grad=True)
                    loss = F.cross_entropy(logits[mask] / temperature, labels[mask], reduction='sum')
                    return loss / mask.sum()
            else:
                logits = all_query @ transpose(all_positive_key)
                labels = torch.arange(len(all_query), device=query.device)
                
                if mask is not None:
                    all_mask = _gather_mask(mask, world_size, gather_with_grad)
                    logits = torch.nan_to_num(logits, nan=-100.0)
                    logits = logits.masked_fill(~all_mask.unsqueeze(0), -100.0)
                    
                    if all_mask.sum() == 0:
                        return torch.tensor(0.0, device=query.device, requires_grad=True)
                    loss = F.cross_entropy(logits[all_mask] / temperature, labels[all_mask], reduction='sum')
                    return loss / all_mask.sum()
        else:
            logits = query @ transpose(positive_key)
            labels = torch.arange(len(query), device=query.device)
        
    if mask is not None:
        logits = torch.nan_to_num(logits, nan=-100.0)
        logits = logits.masked_fill(~mask.unsqueeze(0), -100.0)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=query.device, requires_grad=True)
            
        if reduction == 'mean':
            loss = F.cross_entropy(logits[mask] / temperature, labels[mask], reduction='sum')
            return loss / mask.sum()
        else:
            losses = F.cross_entropy(logits / temperature, labels, reduction='none')
            losses = losses * mask.float()
            return losses.sum() if reduction == 'sum' else losses
    
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]