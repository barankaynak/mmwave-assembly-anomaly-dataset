"""Faz A4 / A4.3(b) - uzak (Linux, RTX 3090) baslatici. tools/remote_start.sh ana uzak venv ile calistirir;
bu betik ~/mmwave-revizyon/.venv-tf (TensorFlow 2.18.0 GPU, Keras 3.9.2; yerel .venv-tf ile ayni surumler)
python'u ile faz_a4/14_keras_train.py'yi cagirir. Kullanim: python faz_a4/14_keras_remote.py [14_keras_train.py argumanlari]
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_TF = Path.home() / "mmwave-revizyon" / ".venv-tf" / "bin" / "python"
env = dict(os.environ)
env.setdefault("TF_INTRA", "8")
env.setdefault("TF_INTER", "2")
env["OMP_NUM_THREADS"] = env["TF_INTRA"]
env["TF_CPP_MIN_LOG_LEVEL"] = "2"
env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"   # GPU baska kullanicilarla paylasiliyor
env.setdefault("PYTHONUNBUFFERED", "1")
cmd = [str(PY_TF), str(ROOT / "faz_a4" / "14_keras_train.py")] + sys.argv[1:]
print("[launch-remote]", " ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd, cwd=str(ROOT), env=env))
