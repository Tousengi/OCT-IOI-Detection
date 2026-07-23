def bbox_wise_iou(box1, box2, xywh=True, wise_iou_version=3, eps=1e-7):
  
    if xywh:
        (x1, y1, w1, h1) = box1.unbind(-1)
        (x2, y2, w2, h2) = box2.unbind(-1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area + eps
    iou = inter_area / union_area

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1) 
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1) 
    c2 = cw ** 2 + ch ** 2 + eps                           

    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
    ) / 4

    wiou_base = iou - rho2 / c2

    if wise_iou_version == 1:
        loss = 1 - wiou_base

    elif wise_iou_version == 2:
        beta = 1.0
        focal = torch.exp((1 - iou) ** 2 / beta)
        loss = focal * (1 - wiou_base)

    else:
        wise_ratio = (rho2 / c2).detach()

        r = wise_ratio / (wise_ratio.mean() + eps)
        beta = 2.0                         
        focal = torch.exp(-(r ** beta))     
        loss = focal * (1 - wiou_base)

    return loss
