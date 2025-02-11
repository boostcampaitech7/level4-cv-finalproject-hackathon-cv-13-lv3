import numpy as np
from audiomentations import (
    Compose,
    Normalize,
)

def get_transforms():
    transforms = Compose(transforms=[
        Normalize(apply_to="all", p=1),
    ])
    
    return transforms
