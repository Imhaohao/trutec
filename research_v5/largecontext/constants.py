"""Sizes for the half-resolution, query-conditioned hatch U-Net."""

HALF_CROP = 1024
NATIVE_CROP = HALF_CROP * 2
QUERY_SIZE = 96
TILE = 1024
OVERLAP = 256
OUTPUT_DIVISOR = 32
NATIVE_PER_CELL = 4
KNOWN_FRACTION = 0.75
VEC_DIM = 128
QUERY_MAP = 12
INPUT_CHANNELS = 4
SEED = 1234
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
POINT_CAP = 4096

# Drawing-stack convs, in forward order: (stride, dilation). Five stride-2
# layers reach 1/32. Three dilation-2 layers at the bottom push the deepest
# receptive field past 1000 native pixels. Keep this tuple matched to the
# drawing convolutions in model.py.
ENCODER_OPS = (
    (1, 1),
    (2, 1), (1, 1),
    (2, 1), (1, 1),
    (2, 1), (1, 1),
    (2, 1), (1, 1),
    (2, 1), (1, 1),
    (1, 2), (1, 2), (1, 2),
)


def deepest_native_receptive_field(ops: tuple[tuple[int, int], ...] = ENCODER_OPS) -> int:
    """Receptive field of the deepest map, measured in native pixels.

    The drawing branch sees a half-resolution image, so a field of N
    half-resolution pixels covers 2N native pixels.
    """
    field, jump = 1, 1
    for stride, dilation in ops:
        field += (3 - 1) * dilation * jump
        jump *= stride
    return field * 2
