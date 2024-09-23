# config.py

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'min_dim': 224,
    'feature_maps': [28, 14, 7, 7, 4, 2],  # Feature map sizes
    'min_sizes': [20, 40, 80, 80, 140, 224],  # Minimum sizes of anchor boxes
    'max_sizes': [40, 80, 80, 140, 224, 245],
    'steps': [8, 16, 32, 32, 64, 224],  # Stride of the feature maps
    'aspect_ratios': [
        [2],  # Feature map 1 (28x28): aspect ratios
        [2, 3],  # Feature map 2 (14x14)
        [2, 3],  # Feature map 3 (7x7)
        [2, 3],  # Feature map 4 (7x7)
        [2],  # Feature map 5 (5x5)
        [2]  # Feature map 6 (1x1)
    ],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

cs = {
    'num_classes': 9,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'min_dim': 224,
    'feature_maps': [28, 14, 7, 7, 4, 2],  # Feature map sizes
    'min_sizes': [20, 40, 80, 80, 140, 224],  # Minimum sizes of anchor boxes
    'max_sizes': [40, 80, 80, 140, 224, 245],
    'steps': [8, 16, 32, 32, 64, 224],  # Stride of the feature maps
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

csfg = {
    'num_classes': 9,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
     'min_dim': 224,
    'feature_maps': [28, 14, 7, 7, 4, 2],  # Feature map sizes
    'min_sizes': [20, 40, 80, 80, 140, 224],  # Minimum sizes of anchor boxes
    'max_sizes': [40, 80, 80, 140, 224, 245],
    'steps': [8, 16, 32, 32, 64, 224],  # Stride of the feature maps
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

cscar = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [28, 14, 7, 7, 4, 2],  # Feature map sizes
    'min_sizes': [20, 40, 80, 80, 140, 224],  # Minimum sizes of anchor boxes
    'max_sizes': [40, 80, 80, 140, 224, 245],
    'steps': [8, 16, 32, 32, 64, 224],  # Stride of the feature maps
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
     'feature_maps': [28, 14, 7, 7, 4, 2],  # Feature map sizes
    'min_sizes': [20, 40, 80, 80, 140, 224],  # Minimum sizes of anchor boxes
    'max_sizes': [40, 80, 80, 140, 224, 245],
    'steps': [8, 16, 32, 32, 64, 224],  # Stride of the feature maps
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
