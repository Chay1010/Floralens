"""
FloraLens Configuration — Central configuration for model, data paths, and serving.
"""
import os
from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "experiments"
ONNX_MODEL_PATH = MODEL_DIR / "floralens_efficientnet_b2.onnx"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# === Model ===
MODEL_NAME = "efficientnet_b2"          # timm model identifier
NUM_CLASSES = 102                        # Oxford 102 Flowers
IMAGE_SIZE = 260                         # EfficientNet-B2 native resolution
PRETRAINED = True

# === Training Hyperparameters ===
LEARNING_RATE = 3e-4                     # AdamW LR (tuned range: 1e-5 to 1e-3)
WEIGHT_DECAY = 1e-4                      # L2 regularization
BATCH_SIZE = 32
NUM_EPOCHS = 25
PATIENCE = 5                             # Early stopping patience
LABEL_SMOOTHING = 0.1                    # Cross-entropy label smoothing
MIXUP_ALPHA = 0.2                        # Mixup augmentation

# === Data Split ===
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# === Reproducibility Seeds ===
RANDOM_SEED = 42

# === Serving ===
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_IMAGE_SIZE_MB = 10

# === W&B ===
WANDB_PROJECT = "floralens"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)

# === Oxford 102 Flowers class names ===
FLOWER_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke",
    "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff",
    "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
    "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus",
    "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress",
    "canna lily", "hippeastrum", "bee balm", "ball moss",
    "foxglove", "bougainvillea", "camellia", "mallow",
    "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily",
]
