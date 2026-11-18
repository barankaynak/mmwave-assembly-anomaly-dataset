# Faz D — Sızıntı tablosu, bütün CNN-GRU koşuları RTX 3090'da

Plan: `faz_d/DESIGN_D.md`. Üreten: `faz_d/02_leakage_3090.py --source d`. Koşu: `faz_d/01_cnn_s0_seeds_3090.py`.
Ortam: NVIDIA GeForce RTX 3090, torch 2.6.0+cu124, CUDA 12.4, cuDNN 90100.

## Kapılar
- K1: her S0 OOF dosyasının test pencereleri ve y_true ilgili tohumun bölmesiyle birebir aynı (assert) — GEÇTİ.
- K2: tohum 1–4'te test pencereleri ve y_true, A2.2'nin GTX OOF dosyalarıyla birebir aynı (assert) — GEÇTİ.
- K3: yöntem kapısı ayrıca `--source a2` ile koşulur (A2.2 özetini birebir üretmeli).
- K4: GPU 'NVIDIA GeForce RTX 3090', 4 tohumun device'ı cuda — GEÇTİ.

## Koşu başına (CNN-GRU)
| bölme | koşu | n | doğruluk | macro-F1 |
|---|---|---|---|---|
| S1 | 0 | 3300 | 85.45 | 0.8275 |
| S1 | 1 | 3224 | 88.59 | 0.8583 |
| S1 | 2 | 3223 | 87.65 | 0.8533 |
| S1 | 3 | 3237 | 87.15 | 0.8367 |
| S1 | 4 | 3264 | 85.81 | 0.8282 |
| S0 | 42 | 3250 | 90.09 | 0.8781 |
| S0 | 1 | 3250 | 91.35 | 0.8995 |
| S0 | 2 | 3250 | 91.23 | 0.8969 |
| S0 | 3 | 3250 | 90.46 | 0.8858 |
| S0 | 4 | 3250 | 91.69 | 0.9040 |

## Özet (S0 beş tohum ortalaması − S1 beş kat ortalaması; Welch %95 GA)
| model | metrik | S0 ort ± sd | S1 ort ± sd | etki (puan) | %95 GA | A2.2 karışık-GPU etkisi |
|---|---|---|---|---|---|---|
| b1 | acc | 61.64 ± 0.67 | 61.30 ± 1.23 | **+0.34** | [-1.18, +1.87] | +0.34 |
| b1 | macro_f1 | 53.46 ± 0.59 | 52.65 ± 1.36 | **+0.81** | [-0.85, +2.47] | +0.81 |
| b2 | acc | 88.86 ± 0.28 | 88.41 ± 1.16 | **+0.45** | [-0.98, +1.87] | +0.45 |
| b2 | macro_f1 | 86.29 ± 0.41 | 85.64 ± 1.22 | **+0.65** | [-0.84, +2.14] | +0.65 |
| cnngru | acc | 90.97 ± 0.66 | 86.93 ± 1.30 | **+4.04** | [+2.44, +5.63] | +4.42 |
| cnngru | macro_f1 | 89.29 ± 1.07 | 84.08 ± 1.43 | **+5.21** | [+3.34, +7.07] | +5.59 |

## Okuma kuralı (DESIGN_D, koşudan önce)
- Başlık değeri (PyTorch portu, tamamen 3090, beş tohum): doğruluk **+4.04** puan [+2.44, +5.63], macro-F1 **+5.21** [+3.34, +7.07].
- Keras (tek koşu, 3090, A4.3b'den değişmeden): doğruluk +5.34, macro-F1 +6.39 puan.

## Önceden yazılan beklentiler (betimsel)
- (a) 3090 S0 tohum 1–4 doğrulukları %89,5–93,0 içinde: TUTTU (91.35, 91.23, 90.46, 91.69).
- (b) beş tohumlu doğruluk etkisi +2 ile +6 puan arasında: TUTTU (+4.04).
