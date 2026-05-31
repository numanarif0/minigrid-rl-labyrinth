# MiniGrid LavaCrossing — PPO vs DQN Karşılaştırması

Bu proje, **MiniGrid LavaCrossing** ortamında **Proximal Policy Optimization (PPO)** ve **Deep Q-Network (DQN)** algoritmalarının performansını karşılaştırmaktadır. Aynı CNN mimarisi ve BFS tabanlı potansiyel ödül şekillendirmesi kullanılarak adil bir kıyaslama yapılmıştır.

> Makine Öğrenmesi dersi proje çalışması
> Yazılım Mühendisliği Bölümü
> **Numan Arif Deniz** & **Utkuhan Bulut**

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Sonuçlar](#sonuçlar)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Dosya Yapısı](#dosya-yapısı)
- [Versiyonlar](#versiyonlar)

---

## Genel Bakış

**Ortam:** MiniGrid LavaCrossing — ajan, ızgara tabanlı bir dünyada lava engellerini aşarak hedef kareye ulaşmaya çalışır. Dört zorluk seviyesi kullanılmıştır:

| Ortam | Izgara | Lava Sayısı |
|-------|--------|-------------|
| S9N1  | 9×9    | 1 |
| S9N2  | 9×9    | 2 |
| S9N3  | 9×9    | 3 |
| S11N5 | 11×11  | 5 |

**Öne çıkan özellikler:**

- **BFS tabanlı potansiyel ödül şekillendirmesi** — seyrek ödül problemini çözer
- **Aynı CNN mimarisi** her iki algoritma için (KucukCNN)
- **İki eğitim stratejisi:** Curriculum Learning vs Çoklu Ortam (Multi-Env)
- **Genelleme testi** — hiç görülmemiş SimpleCrossing ortamlarında

---

## Sonuçlar

### Başarı Oranları (100 episode, deterministik)

| Versiyon | S9N1 | S9N2 | S9N3 | S11N5 |
|----------|------|------|------|-------|
| PPO V1 (Curriculum) | 20% | 24% | 44% | 40% |
| **PPO V2 (Multi-Env)** | **99%** | **74%** | **61%** | **50%** |
| DQN V1 (Curriculum) | 0% | 0% | 0% | 0% |
| DQN V2 (Multi-Env, eşit) | 39% | 17% | 18% | 2% |
| **DQN V3 (Multi-Env, ağırlıklı)** | **44%** | **26%** | **41%** | **10%** |

### Genelleme Testi (SimpleCrossing — hiç görülmemiş)

| Ortam | PPO V1 | PPO V2 | DQN V3 |
|-------|--------|--------|--------|
| S9N1  | 0%  | 18% | 6% |
| S9N3  | 0%  | 17% | 7% |
| S11N5 | 0%  | 17% | 0% |

**Temel bulgular:**
1. Çoklu ortam eğitimi, curriculum learning'i tüm metriklerde geçmiştir
2. PPO, görsel seyrek ödüllü ortamlarda DQN'den belirgin biçimde üstündür
3. PPO daha iyi genelleme kapasitesine sahiptir (~4 kat)

---

## Kurulum

```bash
# Sanal ortam oluştur
python -m venv .venv

# Aktif et (Windows)
.venv\Scripts\Activate.ps1

# Aktif et (Linux/Mac)
source .venv/bin/activate

# Bağımlılıkları yükle
pip install stable-baselines3[extra] minigrid gymnasium torch tensorboard
```

**Gereksinimler:**
- Python 3.10+
- CUDA destekli GPU (önerilen)
- ~4 GB GPU belleği

---

## Kullanım

### Eğitim

```bash
# PPO V1 — Curriculum (baseline)
python ppotrain.py

# PPO V2 — Çoklu ortam (en iyi)
python ppotrain_v2.py

# DQN V1 — Curriculum (başarısız)
python dqntrain.py

# DQN V2 — Çoklu ortam (eşit dağılım)
python dqntrain_v2.py

# DQN V3 — Çoklu ortam (S11N5 ağırlıklı, en iyi)
python dqntrain_v3.py
```

### Değerlendirme

```bash
# Eğitim ortamları + SimpleCrossing genelleme testi
python ppoevaulate.py
python ppoevaulate_v2.py
python dqnevaluate.py
python dqnevaluate_v2.py
python dqnevaluate_v3.py
```

### TensorBoard

```powershell
# Tüm logları karşılaştırma
.\tensorboard_compare.ps1
```

Veya manuel:
```bash
tensorboard --logdir_spec "PPO_V1:ppo_tensorboard,PPO_V2:ppo_tensorboard_v2,DQN_V2:dqn_tensorboard_v2,DQN_V3:dqn_tensorboard_v3"
```

---

## Dosya Yapısı

```
RL_PROJECT/
├── env_wrapper.py            # BFS ödül şekillendirme + Monitor wrapper
├── ppotrain.py               # PPO V1 (curriculum)
├── ppotrain_v2.py            # PPO V2 (çoklu ortam, en iyi)
├── ppoevaulate.py            # PPO V1 değerlendirme
├── ppoevaulate_v2.py         # PPO V2 değerlendirme
├── dqntrain.py               # DQN V1 (curriculum)
├── dqntrain_v2.py            # DQN V2 (çoklu ortam, eşit)
├── dqntrain_v3.py            # DQN V3 (çoklu ortam, ağırlıklı, en iyi)
├── dqnevaluate.py            # DQN V1 değerlendirme
├── dqnevaluate_v2.py         # DQN V2 değerlendirme
├── dqnevaluate_v3.py         # DQN V3 değerlendirme
├── tensorboard_compare.ps1   # Tüm TensorBoard loglarını birlikte aç
├── best_model/               # Eğitilmiş modeller
│   ├── ppo_runs_cf/          # PPO V1
│   ├── ppo_runs_v2/          # PPO V2
│   ├── dqn_runs_v2/          # DQN V2
│   └── dqn_runs_v3/          # DQN V3
└── README.md
```

---

## Versiyonlar

### PPO

| Versiyon | Strateji | Adım | Notlar |
|----------|----------|------|--------|
| V1 | Curriculum (4 aşama) | 1M | Son ortama özgünleşme sorunu |
| **V2** | Çoklu ortam (8 env) | 5M | **En iyi**, kararlı %95+ rollout |

### DQN

| Versiyon | Strateji | Adım | Notlar |
|----------|----------|------|--------|
| V1 | Curriculum | 7M | %0 başarı (aşamalı unutma) |
| V2 | Çoklu ortam (eşit) | 5M | İlk başarılı DQN konfigürasyonu |
| **V3** | Çoklu ortam (S11N5×3) | 7M | **En iyi**, zor ortam ağırlıklı |

---

## Yöntem Detayları

**BFS Ödül Şekillendirmesi:**

```
r_t = r_terminal + (d_t - d_{t+1}) × 0.1 - 0.001
```

- `d_t`: ajanın hedefe BFS mesafesi (lava engel olarak)
- Başarılı terminal: +1.0
- Başarısız terminal: -1.0
- Zaman cezası: -0.001 (her adım)

**CNN Mimarisi (paylaşılan):**

```
Conv2D(3→16, 2×2) → ReLU
Conv2D(16→32, 2×2) → ReLU
Conv2D(32→64, 2×2) → ReLU
Flatten → Linear(128) → ReLU
```

---

## Kaynaklar

- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*
- Chevalier-Boisvert et al. (2023) — *Minigrid & Miniworld: Modular & Customizable RL Environments* (NeurIPS)
- Raffin et al. (2021) — *Stable-Baselines3*

---

## Lisans

Akademik amaçlı proje çalışması. MIT lisansı altında.
