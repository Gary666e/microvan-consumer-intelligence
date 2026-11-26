"""Execute archived analyses in dependency order from any working directory."""
from pathlib import Path
import os, subprocess, sys
root=Path(__file__).resolve().parent
env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
for name in ["microvan_v3/analyze.py","microvan_v4/analysis_v4.py","microvan_v4/survey_extra.py","microvan_v5/advanced.py","microvan_v5/monitor.py"]:
    subprocess.run([sys.executable,str(root/name)],cwd=root,env=env,check=True)
print("Analysis pipeline completed; outputs are under microvan_v3, microvan_v4 and microvan_v5.")
