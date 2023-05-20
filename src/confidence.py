import torch 
from torchvision.transforms.autoaugment import AutoAugment, _apply_op
import random 

def append_dropout(model, rate=0.2):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        if isinstance(module, torch.nn.ReLU):
            new = torch.nn.Sequential(module, torch.nn.Dropout2d(p=rate, inplace=True))
            setattr(model, name, new)

def mc_dropout(model, images, n_classes, n_iter): 
    append_dropout(model)
    model.train() # for bayesian inference
    model = model.cuda()
    confidence_scores = torch.zeros(images.size(0), n_iter, n_classes) # batchsize x num_iteration x n_classes
    for i in range(n_iter): 
        confidence_scores[:, i, :] = model(images).detach().cpu()
    model.eval()
    return confidence_scores

def mc_perturbation(model, images, n_classes, transforms): 
    ''' 
    images should be original images not normalized one.
    '''
    model.eval()
    model = model.cuda()
    augment = AutoAugment() 
    transforms_cands = augment.policies
    images = images.detach().cpu()
    confidence_scores = torch.zeros(images.size(0), len(transforms_cands), n_classes) # batchsize x num_iteration x n_classes

    bs, c, h, w = images.shape
    assert h == 224 and w == 224, f"invalid image size: {h}x{w} --> 224x224"
    op_meta = augment._augmentation_space(10, (224, 224))
    for i, img in enumerate(images):
        # Constrcut candidates of transformed images 
        img_cands = []
        for p in transforms_cands:
            tmp_img = img
            for op_name, _, magnitude_id in p:
                magnitudes, signed = op_meta[op_name]
                magnitude = float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                if signed and random.random() < 0.5:
                    magnitude *= -1.0
                tmp_img = _apply_op(tmp_img, op_name, magnitude, interpolation=augment.interpolation, fill=augment.fill)
            tmp_img = tmp_img.float() / 255.0
            img_cands.append(transforms(tmp_img)) # for normalized. 
        img_cands = torch.stack(img_cands, dim=0)
        img_cands = img_cands.cuda()
        # model forward 
        confidence_scores[i, :, :] = model(img_cands).detach().cpu()
    return confidence_scores 



                             