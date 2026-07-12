# Colab Checklist

Run this from a fresh Colab notebook:

```bash
!git clone <your-repo-url> dreamloop
%cd dreamloop
!pip install -r requirements.txt
```

Procgen is optional. For the CoinRun route only, run `!pip install -r requirements-procgen.txt`. If that install fails, the runtime is probably newer than Procgen's wheel support; stay on toy arena or use a Python 3.10 runtime/container.

Recommended game-like 128x128 toy arena run:

```bash
!python scripts/generate_toy_arena_npz.py --out data/toy_arena_128_50k.npz --steps 50000 --size 128
```

Train:

```bash
!python scripts/train_next_frame.py --data data/toy_arena_128_50k.npz --epochs 8 --batch-size 32 --out-dir runs/toy_arena_128
```

Preview:

```bash
!python scripts/rollout_preview.py --data data/toy_arena_128_50k.npz --checkpoint runs/toy_arena_128/best.pt --out runs/toy_arena_128/preview.gif --steps 48
```

Then display the GIF:

```python
from IPython.display import Image
Image(filename="runs/toy_arena_128/preview.gif")
```

CoinRun route, if you want to test Procgen later:

```bash
!python scripts/generate_coinrun_npz.py --out data/coinrun_20k.npz --steps 20000 --num-envs 8
```

Train:

```bash
!python scripts/train_next_frame.py --data data/coinrun_20k.npz --epochs 5 --batch-size 64
```

Preview:

```bash
!python scripts/rollout_preview.py --data data/coinrun_20k.npz --checkpoint runs/coinrun_next_frame/best.pt
```

Then display the GIF:

```python
from IPython.display import Image
Image(filename="runs/coinrun_next_frame/rollout.gif")
```

Once this works, scale in this order:

```text
20k transitions -> 100k -> 1M
5 epochs -> 20 epochs
teacher-forced preview -> closed-loop rollout
simple CNN -> ConvLSTM / latent model
```

Double pendulum variant:

```bash
!python scripts/generate_double_pendulum_npz.py --out data/double_pendulum_50k.npz --steps 50000
!python scripts/train_next_frame.py --data data/double_pendulum_50k.npz --epochs 10 --batch-size 128 --out-dir runs/double_pendulum
!python scripts/rollout_preview.py --data data/double_pendulum_50k.npz --checkpoint runs/double_pendulum/best.pt --out runs/double_pendulum/preview.gif --steps 48
```
