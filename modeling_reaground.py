# -*- coding: utf-8 -*-
"""Single-pass region and point heads for Qwen2.5-VL image grounding."""
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration


class ReAGroundHeadModel(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL with dense region logits and point scores plus local offsets."""

    def __init__(self, config):
        super().__init__(config)
        self.region_layer = getattr(config, "region_layer", 8)
        self.region_head = nn.Linear(config.hidden_size, 1)
        self.point_head = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("ReAGroundHeadModel requires one image per sample")
        if input_ids.shape[0] != image_grid_thw.shape[0]:
            raise ValueError("ReAGroundHeadModel supports exactly one image per sample")
        if not 1 <= self.region_layer < self.config.num_hidden_layers:
            raise ValueError("region_layer must be between 1 and num_hidden_layers - 1")

        inputs_embeds = self.model.embed_tokens(input_ids)
        image_embeds = self.visual(pixel_values.type(self.visual.dtype),
                                   grid_thw=image_grid_thw)
        image_mask = input_ids == self.config.image_token_id
        if image_mask.sum().item() != image_embeds.shape[0]:
            raise ValueError("image features and image tokens do not match")
        mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype))
        position_ids, _ = self.get_rope_index(
            input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        outputs = self.model(
            input_ids=None, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, position_ids=position_ids,
            use_cache=False, output_hidden_states=True, return_dict=True)
        middle = outputs.hidden_states[self.region_layer]
        last = outputs.last_hidden_state

        result = {"region_logits": [], "point_scores": [],
                  "point_offsets": [], "grids": []}
        merge = self.config.vision_config.spatial_merge_size
        for i, grid in enumerate(image_grid_thw):
            t, h, w = (int(v) for v in grid.tolist())
            if t != 1 or h % merge or w % merge:
                raise ValueError("expected a single image with an even spatial grid")
            gh, gw = h // merge, w // merge
            n = gh * gw
            visual_middle = middle[i][image_mask[i]]
            visual_last = last[i][image_mask[i]]
            if visual_middle.shape[0] != n:
                raise ValueError("visual token count does not match image_grid_thw")
            region = self.region_head(visual_middle).squeeze(-1)
            point = self.point_head(visual_last)
            result["region_logits"].append(region)
            result["point_scores"].append(point[:, 0] + region)
            result["point_offsets"].append(point[:, 1:3])
            result["grids"].append((gh, gw))
        return result


def grid_targets(bbox, grid, device=None):
    """Normalized GT bbox -> dense region target, center cell and local offset."""
    bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
    gh, gw = grid
    if bbox.shape != (4,) or gh < 1 or gw < 1:
        raise ValueError("invalid bbox or grid")
    x1, y1, x2, y2 = bbox.unbind()
    if not bool((0 <= x1 < x2 <= 1) and (0 <= y1 < y2 <= 1)):
        raise ValueError("bbox must be normalized and non-empty")
    xs = (torch.arange(gw, device=bbox.device, dtype=torch.float32) + 0.5) / gw
    ys = (torch.arange(gh, device=bbox.device, dtype=torch.float32) + 0.5) / gh
    region = ((xs[None, :] >= x1) & (xs[None, :] <= x2)
              & (ys[:, None] >= y1) & (ys[:, None] <= y2))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    col = torch.clamp((cx * gw).long(), max=gw - 1)
    row = torch.clamp((cy * gh).long(), max=gh - 1)
    region[row, col] = True
    center = torch.stack(((col.float() + 0.5) / gw,
                          (row.float() + 0.5) / gh))
    point = torch.stack((cx, cy))
    offset = (point - center) * (2 * torch.tensor([gw, gh], device=bbox.device))
    return region.float().flatten(), row * gw + col, offset


def balanced_bce(logits, target):
    logits, target = logits.float(), target.float()
    positive = target.bool()
    pos_loss = F.binary_cross_entropy_with_logits(logits[positive], target[positive])
    if positive.all():
        return pos_loss
    neg_loss = F.binary_cross_entropy_with_logits(logits[~positive], target[~positive])
    return 0.5 * (pos_loss + neg_loss)


def head_loss(outputs, gt_bboxes, lambda_region=1.0, lambda_offset=1.0):
    losses = {"region": [], "point_ce": [], "offset": []}
    for i, grid in enumerate(outputs["grids"]):
        device = outputs["region_logits"][i].device
        region_target, point_target, offset_target = grid_targets(
            gt_bboxes[i], grid, device=device)
        losses["region"].append(balanced_bce(
            outputs["region_logits"][i], region_target))
        losses["point_ce"].append(F.cross_entropy(
            outputs["point_scores"][i].float().unsqueeze(0),
            point_target.reshape(1)))
        predicted_offset = torch.tanh(
            outputs["point_offsets"][i][point_target]).float()
        losses["offset"].append(F.smooth_l1_loss(
            predicted_offset, offset_target.float()))
    region = torch.stack(losses["region"]).mean()
    point_ce = torch.stack(losses["point_ce"]).mean()
    offset = torch.stack(losses["offset"]).mean()
    total = point_ce + lambda_offset * offset + lambda_region * region
    metrics = {name: float(torch.stack(values).mean().detach())
               for name, values in losses.items()}
    return total, metrics


def decode_point(outputs, image_size, sample_index=0):
    """Decode one sample to original-image pixels without text generation."""
    gh, gw = outputs["grids"][sample_index]
    region = outputs["region_logits"][sample_index]
    scores = outputs["point_scores"][sample_index]
    offsets = outputs["point_offsets"][sample_index]
    if region.numel() != gh * gw or scores.numel() != gh * gw:
        raise ValueError("head output does not match its grid")
    region_index = int(region.argmax())
    point_index = int(scores.argmax())
    row, col = divmod(point_index, gw)
    delta = torch.tanh(offsets[point_index]).float().cpu()
    width, height = image_size
    x = ((col + 0.5) / gw + float(delta[0]) / (2 * gw)) * width
    y = ((row + 0.5) / gh + float(delta[1]) / (2 * gh)) * height
    point = (min(float(width), max(0.0, x)),
             min(float(height), max(0.0, y)))
    info = {"grid": [gh, gw], "region_cell": list(divmod(region_index, gw)),
            "point_cell": [row, col]}
    return point, info
