"""Simple smoke test to verify imports and dataset paths."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

missing = []

print('Project root:', ROOT)

try:
    import config
    print('config: OK')
    print('PATH_TRAIN =', config.PATH_TRAIN)
except Exception as e:
    print('config import failed:', e)
    missing.append(('config', str(e)))

try:
    import importlib
    try:
        IPL = importlib.import_module('Unet_Architecture.Image_Painting.library')
    except Exception:
        # try legacy CamelCase fallback
        IPL = importlib.import_module('Unet_Architecture.Image_Painting.Library')
    print('Image_Painting.library: OK')
except Exception as e:
    import traceback
    print('Image_Painting.library import failed:')
    traceback.print_exc()
    missing.append(('Image_Painting.library', str(e)))

try:
    try:
        SRL = importlib.import_module('Unet_Architecture.Super_Resolution.library')
    except Exception:
        SRL = importlib.import_module('Unet_Architecture.Super_Resolution.Library')
    print('Super_Resolution.library: OK')
except Exception as e:
    import traceback
    print('Super_Resolution.library import failed:')
    traceback.print_exc()
    missing.append(('Super_Resolution.library', str(e)))

if missing:
    print('\nMissing or failing imports:')
    for name, err in missing:
        print(f'- {name}: {err}')
    print('\nTry: pip install -r requirements.txt')
else:
    print('\nAll smoke imports OK')

    # Verify dataset paths from config
    try:
        import config as cf
        t = Path(cf.PATH_TRAIN)
        v = Path(cf.PATH_VAL)
        print('\nDataset paths:')
        print(' TRAIN exists:', t.exists(), t)
        print('  VAL exists:', v.exists(), v)
    except Exception as e:
        print('Could not verify dataset paths:', e)

    # Optional: check if torch is available and run a tiny data loader sample
    try:
        import torch
        print('\ntorch available:', torch.__version__)
    except Exception:
        print('\nTorch not available; skip runtime dataset check')


'''
This script should run out something like this:

(unet_skull) E:\Research\2DSkullReconstructionInUNet>python run_smoke.py
Project root: E:\Research\2DSkullReconstructionInUNet
config: OK
PATH_TRAIN = E:\Research\2DSkullReconstructionInUNet\dataset\train
Image_Painting.library: OK
Super_Resolution.library: OK

All smoke imports OK

Dataset paths:
 TRAIN exists: True E:\Research\2DSkullReconstructionInUNet\dataset\train
  VAL exists: True E:\Research\2DSkullReconstructionInUNet\dataset\val

torch available: 2.8.0+cpu
'''