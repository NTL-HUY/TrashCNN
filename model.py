import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, roi_align


# 1.  RESNET50 BACKBONE
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self.block(x) + identity)


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64,   64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(256,  128, blocks=4, stride=2)
        self.layer3 = self._make_layer(512,  256, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)

    def _make_layer(self, in_channels, mid_channels, blocks, stride):
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Bottleneck(in_channels, mid_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x  = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


# 2.  FPN
class FPN(nn.Module):
    def __init__(self, in_channels_list=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList([
            self._make_lateral(in_ch, out_channels)
            for in_ch in in_channels_list
        ])
        self.output = nn.ModuleList([
            self._make_output(out_channels, out_channels)
            for _ in in_channels_list
        ])
        self.p6_pool = nn.MaxPool2d(kernel_size=1, stride=2)

    def _make_lateral(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
        )

    def _make_output(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        laterals = [l(f) for l, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="nearest"
            )
        p2, p3, p4, p5 = [out(lat) for out, lat in zip(self.output, laterals)]
        p6 = self.p6_pool(p5)
        return p2, p3, p4, p5, p6


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RPN
# ══════════════════════════════════════════════════════════════════════════════
class RPNHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred  = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        for layer in [self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.conv(x)
        return self.cls_logits(x), self.bbox_pred(x)


class AnchorGenerator(nn.Module):
    def __init__(self,
                 sizes=((32,), (64,), (128,), (256,), (512,)),
                 ratios=((0.5, 1.0, 2.0),) * 5):
        super().__init__()
        self.sizes  = sizes
        self.ratios = ratios

    def _make_anchors_single(self, size, ratios, stride, feat_h, feat_w, device):
        scales   = torch.tensor(size,   dtype=torch.float32, device=device)
        ratios_t = torch.tensor(ratios, dtype=torch.float32, device=device)
        h_ratios = torch.sqrt(ratios_t)
        w_ratios = 1.0 / h_ratios
        ws = (scales[:, None] * w_ratios[None, :]).view(-1)
        hs = (scales[:, None] * h_ratios[None, :]).view(-1)
        shift_x = (torch.arange(feat_w, device=device) + 0.5) * stride
        shift_y = (torch.arange(feat_h, device=device) + 0.5) * stride
        y, x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        x, y = x.reshape(-1), y.reshape(-1)
        x1 = (x[:, None] - ws[None, :] / 2).reshape(-1)
        y1 = (y[:, None] - hs[None, :] / 2).reshape(-1)
        x2 = (x[:, None] + ws[None, :] / 2).reshape(-1)
        y2 = (y[:, None] + hs[None, :] / 2).reshape(-1)
        return torch.stack([x1, y1, x2, y2], dim=1)

    def forward(self, feature_maps, strides=(4, 8, 16, 32, 64)):
        all_anchors = []
        device = feature_maps[0].device
        for i, fm in enumerate(feature_maps):
            _, _, h, w = fm.shape
            anchors = self._make_anchors_single(
                self.sizes[i], self.ratios[i], strides[i], h, w, device
            )
            all_anchors.append(anchors)
        return torch.cat(all_anchors, dim=0)


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes (N,4) and (M,4) → (N,M)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def encode_boxes(proposals, gt_boxes):
    """Encode gt_boxes relative to proposals as (dx,dy,dw,dh)."""
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]
    pcx = (proposals[:, 0] + proposals[:, 2]) / 2
    pcy = (proposals[:, 1] + proposals[:, 3]) / 2
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]
    gcx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gcy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    dx = (gcx - pcx) / pw.clamp(min=1e-6)
    dy = (gcy - pcy) / ph.clamp(min=1e-6)
    dw = torch.log(gw.clamp(min=1e-6) / pw.clamp(min=1e-6))
    dh = torch.log(gh.clamp(min=1e-6) / ph.clamp(min=1e-6))
    return torch.stack([dx, dy, dw, dh], dim=1)


class RPN(nn.Module):
    def __init__(self, in_channels=256, num_anchors_per_level=3,
                 nms_thresh=0.7, pre_nms_top_n=2000, post_nms_top_n=1000,
                 min_size=1.0,
                 rpn_pos_iou=0.7, rpn_neg_iou=0.3, rpn_batch=256, rpn_pos_frac=0.5):
        super().__init__()
        self.head           = RPNHead(in_channels, num_anchors_per_level)
        self.anchor_gen     = AnchorGenerator()
        self.nms_thresh     = nms_thresh
        self.pre_nms_top_n  = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.min_size       = min_size
        self.rpn_pos_iou    = rpn_pos_iou
        self.rpn_neg_iou    = rpn_neg_iou
        self.rpn_batch      = rpn_batch
        self.rpn_pos_frac   = rpn_pos_frac

    @staticmethod
    def decode_boxes(anchors, deltas):
        wa = anchors[:, 2] - anchors[:, 0]
        ha = anchors[:, 3] - anchors[:, 1]
        cx = (anchors[:, 0] + anchors[:, 2]) / 2
        cy = (anchors[:, 1] + anchors[:, 3]) / 2
        dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
        pred_cx = dx * wa + cx
        pred_cy = dy * ha + cy
        pred_w  = torch.exp(dw.clamp(max=4.0)) * wa
        pred_h  = torch.exp(dh.clamp(max=4.0)) * ha
        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    def _get_proposals(self, anchors, cls_flat, bbox_flat, image_shape):
        H, W = image_shape
        proposals = []
        for b in range(cls_flat.shape[0]):
            scores = torch.sigmoid(cls_flat[b])
            deltas = bbox_flat[b]
            boxes  = self.decode_boxes(anchors, deltas)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H)
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            keep = (ws >= self.min_size) & (hs >= self.min_size)
            boxes, scores = boxes[keep], scores[keep]
            topk = min(self.pre_nms_top_n, scores.numel())
            _, idx = scores.topk(topk)
            boxes, scores = boxes[idx], scores[idx]
            keep = nms(boxes, scores, self.nms_thresh)[:self.post_nms_top_n]
            proposals.append(boxes[keep])
        return proposals

    def _compute_rpn_loss(self, anchors, cls_flat, bbox_flat, targets, image_shape):
        H, W = image_shape
        total_cls_loss  = torch.tensor(0.0, device=anchors.device)
        total_reg_loss  = torch.tensor(0.0, device=anchors.device)

        for b in range(cls_flat.shape[0]):
            gt_boxes = targets[b]["boxes"]
            if gt_boxes.numel() == 0:
                continue

            # clip anchors
            valid_mask = (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & \
                         (anchors[:, 2] <= W) & (anchors[:, 3] <= H)
            valid_anchors = anchors[valid_mask]
            valid_cls     = cls_flat[b][valid_mask]
            valid_bbox    = bbox_flat[b][valid_mask]

            iou = box_iou(valid_anchors, gt_boxes)          # (A, G)
            max_iou, gt_idx = iou.max(dim=1)

            labels = torch.full((valid_anchors.shape[0],), -1,
                                dtype=torch.long, device=anchors.device)
            labels[max_iou >= self.rpn_pos_iou] = 1
            labels[max_iou < self.rpn_neg_iou]  = 0
            # each gt must have at least one pos
            best_anchor_per_gt = iou.argmax(dim=0)
            labels[best_anchor_per_gt] = 1

            # sample
            n_pos = int(self.rpn_batch * self.rpn_pos_frac)
            pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
            neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
            if pos_idx.numel() > n_pos:
                pos_idx = pos_idx[torch.randperm(pos_idx.numel())[:n_pos]]
            n_neg = self.rpn_batch - pos_idx.numel()
            if neg_idx.numel() > n_neg:
                neg_idx = neg_idx[torch.randperm(neg_idx.numel())[:n_neg]]
            sampled = torch.cat([pos_idx, neg_idx])

            # cls loss
            cls_targets = (labels[sampled] == 1).float()
            total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                valid_cls[sampled], cls_targets, reduction="mean"
            )

            # reg loss (only positives)
            if pos_idx.numel() > 0:
                matched_gt   = gt_boxes[gt_idx[pos_idx]]
                encoded      = encode_boxes(valid_anchors[pos_idx], matched_gt)
                pred_deltas  = valid_bbox[pos_idx]
                total_reg_loss = total_reg_loss + F.smooth_l1_loss(
                    pred_deltas, encoded, reduction="mean"
                )

        n = cls_flat.shape[0]
        return total_cls_loss / n, total_reg_loss / n

    def forward(self, feature_maps, image_shape, targets=None):
        all_cls, all_bbox = [], []
        for fm in feature_maps:
            cls_l, box_l = self.head(fm)
            all_cls.append(cls_l)
            all_bbox.append(box_l)

        strides = (4, 8, 16, 32, 64)
        anchors = self.anchor_gen(feature_maps, strides)

        cls_flat  = torch.cat([c.permute(0,2,3,1).reshape(c.shape[0],-1)
                                for c in all_cls],  dim=1)
        bbox_flat = torch.cat([b.permute(0,2,3,1).reshape(b.shape[0],-1,4)
                                for b in all_bbox], dim=1)

        proposals = self._get_proposals(anchors, cls_flat, bbox_flat, image_shape)

        if self.training and targets is not None:
            rpn_cls_loss, rpn_reg_loss = self._compute_rpn_loss(
                anchors, cls_flat, bbox_flat, targets, image_shape
            )
            return proposals, rpn_cls_loss, rpn_reg_loss

        return proposals, None, None


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ROI HEAD
# ══════════════════════════════════════════════════════════════════════════════
class BoxHead(nn.Module):
    def __init__(self, in_channels=256, roi_size=7, num_classes=91):
        super().__init__()
        flat = in_channels * roi_size * roi_size
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 1024),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight,  std=0.001)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.zeros_(self.bbox_pred.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.cls_score(x), self.bbox_pred(x)


class RoIHead(nn.Module):
    def __init__(self, in_channels=256, roi_size=7, num_classes=91,
                 score_thresh=0.01, nms_thresh=0.5, detections_per_img=100,
                 roi_pos_iou=0.5, roi_neg_iou_hi=0.5, roi_neg_iou_lo=0.0,
                 roi_batch=128, roi_pos_frac=0.25):
        super().__init__()
        self.roi_size           = roi_size
        self.score_thresh       = score_thresh
        self.nms_thresh         = nms_thresh
        self.detections_per_img = detections_per_img
        self.box_head           = BoxHead(in_channels, roi_size, num_classes)
        self.num_classes        = num_classes
        self.roi_pos_iou        = roi_pos_iou
        self.roi_neg_iou_hi     = roi_neg_iou_hi
        self.roi_neg_iou_lo     = roi_neg_iou_lo
        self.roi_batch          = roi_batch
        self.roi_pos_frac       = roi_pos_frac

    @staticmethod
    def assign_fpn_level(boxes, k0=4, canonical_size=224):
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        scale = torch.sqrt(ws * hs)
        levels = torch.floor(k0 + torch.log2(scale / canonical_size + 1e-6))
        return levels.clamp(2, 5).long()

    def _extract_roi_features(self, fpn_features, proposals_per_image, b):
        spatial_scales = [1/4, 1/8, 1/16, 1/32]
        props = proposals_per_image
        levels = self.assign_fpn_level(props)
        all_feat = torch.zeros(
            props.shape[0], fpn_features[0].shape[1],
            self.roi_size, self.roi_size, device=props.device
        )
        for lvl_idx, lvl in enumerate([2, 3, 4, 5]):
            mask = (levels == lvl)
            if mask.sum() == 0:
                continue
            rois = props[mask]
            batch_idx = torch.zeros(rois.shape[0], 1, device=rois.device)
            rois_with_batch = torch.cat([batch_idx, rois], dim=1)
            feat = roi_align(
                fpn_features[lvl_idx][b:b+1], rois_with_batch,
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=spatial_scales[lvl_idx],
                aligned=True,
            )
            all_feat[mask] = feat
        return all_feat

    def forward(self, features, proposals, image_shape, targets=None):
        fpn_features = list(features[:4])
        all_cls_logits, all_bbox_preds, all_proposals_used = [], [], []

        for b, props in enumerate(proposals):
            if props.numel() == 0:
                all_cls_logits.append(torch.zeros(0, self.num_classes, device=props.device))
                all_bbox_preds.append(torch.zeros(0, self.num_classes * 4, device=props.device))
                all_proposals_used.append(props)
                continue

            # ── Training: sample proposals by IoU with GT ──────────────────
            if self.training and targets is not None:
                gt_boxes  = targets[b]["boxes"]
                gt_labels = targets[b]["labels"]
                if gt_boxes.numel() > 0:
                    iou = box_iou(props, gt_boxes)              # (P, G)
                    max_iou, gt_idx = iou.max(dim=1)

                    roi_labels = torch.zeros(props.shape[0], dtype=torch.long, device=props.device)
                    roi_labels[max_iou >= self.roi_pos_iou] = \
                        gt_labels[gt_idx[max_iou >= self.roi_pos_iou]]

                    pos_mask = max_iou >= self.roi_pos_iou
                    neg_mask = (max_iou < self.roi_neg_iou_hi) & (max_iou >= self.roi_neg_iou_lo)

                    n_pos = int(self.roi_batch * self.roi_pos_frac)
                    pos_idx = pos_mask.nonzero(as_tuple=True)[0]
                    neg_idx = neg_mask.nonzero(as_tuple=True)[0]
                    if pos_idx.numel() > n_pos:
                        pos_idx = pos_idx[torch.randperm(pos_idx.numel())[:n_pos]]
                    n_neg = self.roi_batch - pos_idx.numel()
                    if neg_idx.numel() > n_neg:
                        neg_idx = neg_idx[torch.randperm(neg_idx.numel())[:n_neg]]
                    sampled = torch.cat([pos_idx, neg_idx])
                    props = props[sampled]
                    # store for loss computation
                    sampled_roi_labels = roi_labels[sampled]
                    sampled_gt_idx     = gt_idx[sampled]
                    sampled_pos_mask   = pos_mask[sampled]
                else:
                    sampled_roi_labels = torch.zeros(props.shape[0], dtype=torch.long, device=props.device)
                    sampled_gt_idx     = torch.zeros(props.shape[0], dtype=torch.long, device=props.device)
                    sampled_pos_mask   = torch.zeros(props.shape[0], dtype=torch.bool,  device=props.device)
                    gt_boxes = torch.zeros((1, 4), device=props.device)

                all_proposals_used.append({
                    "props": props,
                    "roi_labels": sampled_roi_labels,
                    "gt_idx": sampled_gt_idx,
                    "pos_mask": sampled_pos_mask,
                    "gt_boxes": targets[b]["boxes"] if targets[b]["boxes"].numel() > 0
                                else torch.zeros((1,4), device=props.device),
                })
            else:
                all_proposals_used.append(props)

            # ── RoI features ───────────────────────────────────────────────
            p = props if not isinstance(props, dict) else props
            if self.training and targets is not None:
                p = all_proposals_used[-1]["props"]

            if p.numel() == 0:
                all_cls_logits.append(torch.zeros(0, self.num_classes, device=p.device))
                all_bbox_preds.append(torch.zeros(0, self.num_classes * 4, device=p.device))
                continue

            roi_feat = self._extract_roi_features(fpn_features, p, b)
            cls_logits, bbox_pred = self.box_head(roi_feat)
            all_cls_logits.append(cls_logits)
            all_bbox_preds.append(bbox_pred)

        # ── Compute RoI losses ─────────────────────────────────────────────
        if self.training and targets is not None:
            roi_cls_loss = torch.tensor(0.0, device=features[0].device)
            roi_reg_loss = torch.tensor(0.0, device=features[0].device)
            n_valid = 0
            for b, info in enumerate(all_proposals_used):
                if not isinstance(info, dict):
                    continue
                cls_l  = all_cls_logits[b]
                box_p  = all_bbox_preds[b]
                if cls_l.shape[0] == 0:
                    continue
                rl = info["roi_labels"]
                roi_cls_loss = roi_cls_loss + F.cross_entropy(cls_l, rl)
                pos = info["pos_mask"]
                if pos.sum() > 0:
                    matched_gt  = info["gt_boxes"][info["gt_idx"][pos]]
                    encoded     = encode_boxes(info["props"][pos], matched_gt)
                    cls_id      = rl[pos]
                    pred_deltas = box_p[pos].reshape(-1, self.num_classes, 4)
                    pred_deltas = pred_deltas[torch.arange(pos.sum()), cls_id]
                    roi_reg_loss = roi_reg_loss + F.smooth_l1_loss(
                        pred_deltas, encoded, reduction="mean"
                    )
                n_valid += 1
            if n_valid > 0:
                roi_cls_loss = roi_cls_loss / n_valid
                roi_reg_loss = roi_reg_loss / n_valid
            return all_cls_logits, all_bbox_preds, roi_cls_loss, roi_reg_loss

        return all_cls_logits, all_bbox_preds, None, None

    def postprocess(self, cls_logits, bbox_preds, proposals, image_shape):
        results = []
        H, W = image_shape
        for cls_l, box_p, props in zip(cls_logits, bbox_preds, proposals):
            p = props if not isinstance(props, dict) else props["props"]
            if p.numel() == 0:
                results.append({
                    "boxes": torch.zeros(0, 4, device=p.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=p.device),
                    "scores": torch.zeros(0, device=p.device)
                })
                continue

            scores = F.softmax(cls_l, dim=-1)
            scores_fg = scores[:, 1:]
            boxes_all, scores_all, labels_all = [], [], []

            for c in range(scores_fg.shape[1]):
                sc   = scores_fg[:, c]
                keep = sc > self.score_thresh
                if keep.sum() == 0:
                    continue
                b_keep  = p[keep]
                sc_keep = sc[keep]
                nms_keep = nms(b_keep, sc_keep, self.nms_thresh)
                boxes_all.append(b_keep[nms_keep])
                scores_all.append(sc_keep[nms_keep])
                labels_all.append(torch.full((nms_keep.numel(),), c + 1,
                                             dtype=torch.long, device=p.device))

            if boxes_all:
                all_b = torch.cat(boxes_all)
                all_s = torch.cat(scores_all)
                all_l = torch.cat(labels_all)
                k = min(self.detections_per_img, all_s.numel())
                _, topk_idx = all_s.topk(k)
                results.append({"boxes": all_b[topk_idx],
                                 "labels": all_l[topk_idx],
                                 "scores": all_s[topk_idx]})
            else:
                results.append({
                    "boxes":  torch.zeros(0, 4, device=p.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=p.device),
                    "scores": torch.zeros(0, device=p.device)
                })
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FASTER R-CNN  (unified forward for train & eval)
# ══════════════════════════════════════════════════════════════════════════════
class FasterRCNN(nn.Module):
    """
    Train mode : model(images, targets) → dict of losses
    Eval  mode : model(images)          → list of dicts {boxes, labels, scores}
    """

    def __init__(self, num_classes=91):
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.fpn      = FPN(in_channels_list=(256, 512, 1024, 2048), out_channels=256)
        self.rpn      = RPN(in_channels=256, num_anchors_per_level=3)
        self.roi_head = RoIHead(in_channels=256, roi_size=7, num_classes=num_classes)

    def forward(self, images, targets=None):
        # ── Stack images ──────────────────────────────────────────
        if isinstance(images, (list, tuple)):
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)
            padded = []
            for img in images:
                h, w = img.shape[1], img.shape[2]
                pad_h = max_h - h  # pad bottom
                pad_w = max_w - w  # pad right
                # F.pad format: (left, right, top, bottom)
                img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
                padded.append(img)
            images = torch.stack(padded, dim=0)
        _, _, H, W = images.shape

        # ── Backbone + FPN ────────────────────────────────────────
        c2, c3, c4, c5 = self.backbone(images)
        fpn_maps = self.fpn((c2, c3, c4, c5))   # (P2..P6)

        # ── RPN ───────────────────────────────────────────────────
        proposals, rpn_cls_loss, rpn_reg_loss = self.rpn(
            fpn_maps, (H, W), targets=targets
        )

        # ── RoI Head ──────────────────────────────────────────────
        cls_logits, bbox_preds, roi_cls_loss, roi_reg_loss = self.roi_head(
            fpn_maps, proposals, (H, W), targets=targets
        )

        if self.training and targets is not None:
            losses = {
                "loss_rpn_cls": rpn_cls_loss  if rpn_cls_loss  is not None else torch.tensor(0.0),
                "loss_rpn_reg": rpn_reg_loss  if rpn_reg_loss  is not None else torch.tensor(0.0),
                "loss_roi_cls": roi_cls_loss  if roi_cls_loss  is not None else torch.tensor(0.0),
                "loss_roi_reg": roi_reg_loss  if roi_reg_loss  is not None else torch.tensor(0.0),
            }
            return losses

        # ── Inference postprocess ─────────────────────────────────
        results = self.roi_head.postprocess(cls_logits, bbox_preds, proposals, (H, W))
        return results


def build_model(num_classes):
    return FasterRCNN(num_classes=num_classes)


if __name__ == "__main__":
    model = FasterRCNN(num_classes=6)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

    # ── Test training forward ──────────────────────────────────────
    model.train()
    images  = [torch.rand(3, 416, 416) for _ in range(2)]
    targets = [
        {"boxes":  torch.tensor([[50., 30., 200., 180.], [220., 100., 350., 300.]]),
         "labels": torch.tensor([1, 2]),
         "image_id": torch.tensor([0])},
        {"boxes":  torch.tensor([[10., 10., 100., 100.]]),
         "labels": torch.tensor([3]),
         "image_id": torch.tensor([1])},
    ]
    losses = model(images, targets)
    print("Train losses:", {k: f"{v.item():.4f}" for k, v in losses.items()})

    # ── Test eval forward ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds = model(images)
    for i, p in enumerate(preds):
        print(f"Image {i}: {len(p['boxes'])} detections")