import glob
import torch
import numpy as np
import random
import re
from Pathlib import Path

from datetime import datetime
from zoneinfo import ZoneInfo

def generate_run_id(zone: ZoneInfo = ZoneInfo("Asia/Kathmandu")) -> str:
    """Generate a unique run ID using current UTC date and time.

    Args:
        zone (ZoneInfo, optional): Timezone information. Defaults to Indian Standard Time.

    Returns:
        str: A unique run ID in the format 'run-YYYY-MM-DD-HH-MM-SS'.
    """
    try:
        current_utc_time = datetime.utcnow().astimezone(zone)
        formatted_time = current_utc_time.strftime("%Y-%m-%d-%H-%M-%S")
        return f"run-{formatted_time}"
    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error generating run ID: {e}")
        return None  # Or raise an exception if appropriate
    
def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
