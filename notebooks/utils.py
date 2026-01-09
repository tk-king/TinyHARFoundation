# Add autoreload using % snytax
import IPython
ip = IPython.get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'autoreload')
    ip.run_line_magic('autoreload', '2')

import sys
from pathlib import Path
notebook_dir = Path().resolve()
project_root = notebook_dir.parent
sys.path.append(str(project_root))
