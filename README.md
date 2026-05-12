# MiniGrid RL Labyrinth

Bu repo, MiniGrid ortamlarında PPO ile eğitim (`ppotrain.py`) ve değerlendirme (`ppoevaulate.py`) yapar.

## 1) Kurulum

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Eğitim

```bash
python ppotrain.py
```

Eğitim sırasında model checkpoint'leri ve loglar aşağıya yazılır:

- `best_model/...` : en iyi model dosyaları (`*.zip`)
- `ppo_tensorboard/...` : TensorBoard event logları

## 3) Değerlendirme

> Dosya adı repoda bilerek `ppoevaulate.py` şeklindedir.

```bash
python ppoevaulate.py
```

Script varsayılan olarak modeli `./best_model/2/best_model` yolundan yükler.

## 4) TensorBoard

```bash
tensorboard --logdir ./ppo_tensorboard
```

Ardından tarayıcıdan verilen yerel adrese gidin (genellikle `http://localhost:6006`).

## 5) CUDA / CPU notu

- `ppotrain.py`, otomatik olarak `torch.cuda.is_available()` kontrolü yapar.
- CUDA varsa GPU kullanır, yoksa CPU ile çalışır.
- CPU'da eğitim daha yavaş olabilir.
- CUDA kurulumunda sorun yaşarsanız PyTorch'u kendi CUDA sürümünüze uygun resmi komutla yeniden kurun: https://pytorch.org/get-started/locally/
