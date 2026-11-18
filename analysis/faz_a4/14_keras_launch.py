"""Faz A4 / A4.3(b) - ince baslatici (ana venv). tools/local_job.py her zaman ana venv python'unu kullanir; bu betik
<project>/.venv-tf python'u ile faz_a4/14_keras_train.py'yi calistirir (ortam degiskenleri korunur).
Kullanim: python faz_a4/14_keras_launch.py [14_keras_train.py argumanlari...]
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_TF = Path(os.environ.get("PYTHON_TF", sys.executable))  # interpreter of the TF 2.18 / Keras 3.9.2 environment
env = dict(os.environ)
env["OMP_NUM_THREADS"] = env.get("TF_INTRA", "4")
env["TF_CPP_MIN_LOG_LEVEL"] = "2"
env.setdefault("PYTHONUNBUFFERED", "1")
cmd = [str(PY_TF), str(ROOT / "faz_a4" / "14_keras_train.py")] + sys.argv[1:]
print("[launch]", " ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd, cwd=str(ROOT), env=env))
