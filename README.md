# Brent Petrol Forward Volatilite Tahmini

Bu proje, petrol piyasası volatilitesinin (`OVX`) ve jeopolitik risk göstergelerinin (`GPRD`, `GPRD_THREAT`) Brent petrol davranışını tahmin etmede yardımcı olup olamayacağını araştırıyor.

İlk denemeler, 1 günlük Brent fiyat/getiri tahmininin sağlam olmadığını gösterdi: XGBoost, Attention BiLSTM ve stacking modelleri, basit random-walk ya da ortalama-getiri (mean-return) baseline'larını geçemedi. Bu yüzden proje daha savunulabilir bir hedefe yeniden çerçevelendi:

**10 günlük forward Brent volatilitesini tahmin etmek.**

## Ana Bulgu

En iyi sonuç, basit bir hibrit ortalamadan geldi:

```text
Hybrid = 0.5 * XGBoost + 0.5 * Attention BiLSTM
```

`10 günlük forward volatilite` hedefi için.

| Model / Baseline | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| Simple Avg XGB + LSTM | 0.005990 | 0.004708 | 0.2055 |
| XGBoost | 0.006156 | 0.004867 | 0.1610 |
| Attention BiLSTM | 0.006184 | 0.004710 | 0.1533 |
| Train mean baseline | 0.006749 | 0.005266 | -0.0086 |
| Past volatility baseline | 0.007848 | 0.005921 | -0.3636 |
| Ridge stacking | 0.006769 | 0.005004 | -0.0145 |

Yorum: OVX ve jeopolitik risk değişkenleri, Brent volatilitesini tahmin etmede günlük Brent getiri yönünü tahmin etmekten daha kullanışlı.

## Veri

Beklenen girdi dosyası:

```text
work/veriseti.xlsx
```

Beklenen sayfa (sheet):

```text
data (1)
```

Gerekli sütunlar:

```text
Date
Brent_Petrol
OVX
GPRD
GPRD_THREAT
```

Ham Excel veri seti özel/gizli olabileceği için varsayılan olarak Git'e dahil edilmemiştir (`.gitignore`).

## Proje Yapısı

```text
.
├── README.md
├── requirements.txt
├── work/
│   ├── brentproject.py
│   ├── run_xgb_model.py
│   ├── run_xgb_optuna.py
│   ├── run_attention_bilstm.py
│   ├── run_attention_bilstm_optuna.py
│   ├── run_stacking_xgb_lstm.py
│   ├── run_multihorizon_xgb.py
│   └── run_volatility10_lstm_stacking.py
└── outputs/
    ├── xgb_multihorizon_summary.csv
    ├── volatility10_lstm_stacking_results.json
    └── üretilen tahminler, grafikler ve model dosyaları
```

## Metodoloji

### Ön İşleme (Preprocessing)

- Kronolojik train/validation/test bölünmesi.
- Rastgele karıştırma (shuffle) yapılmadı, çünkü bu bir zaman serisi verisi.
- Veri sızıntısını (leakage) önlemek için winsorization/scaling yalnızca train verisi üzerinden fit edildi.
- Non-causal wavelet denoising devre dışı bırakıldı, çünkü gelecekteki bilginin geçmiş gözlemlere sızmasına (leakage) yol açabiliyor.

### Özellik Mühendisliği (Feature Engineering)

Kullanılan özellikler:

- Brent lag ve EMA özellikleri.
- OVX lag, EMA, z-score, spike ve rejim (regime) özellikleri.
- GPRD ve GPRD Threat lag/EMA/z-score özellikleri.
- `OVX * GPRD` ve `OVX * GPRD_THREAT` gibi etkileşim (interaction) özellikleri.
- Takvim kontrolleri (calendar controls).
- Log getiri tabanlı realized volatility.

### Test Edilen Hedefler (Targets)

Projede test edilenler:

- 1 günlük forward log getiri.
- 5 günlük forward log getiri.
- 10 günlük forward log getiri.
- 5 günlük forward volatilite.
- 10 günlük forward volatilite.

Getiri (return) hedefleri zayıf kaldı ve baseline'ları sağlam biçimde geçemedi. Volatilite hedefleri daha güçlüydü, özellikle 10 günlük forward volatilite.

## Nasıl Çalıştırılır

Bağımlılıkları kur:

```bash
pip install -r requirements.txt
```

Veri setini şu konuma yerleştir:

```text
work/veriseti.xlsx
```

Çok-ufuklu (multi-horizon) XGBoost karşılaştırmasını çalıştır:

```bash
python work/run_multihorizon_xgb.py
```

Final 10 günlük volatilite hibrit deneyini çalıştır:

```bash
python work/run_volatility10_lstm_stacking.py
```

## Önemli Çıktılar

Önemli çıktı dosyaları:

```text
outputs/xgb_multihorizon_summary.csv
outputs/xgb_multihorizon_results.json
outputs/volatility10_lstm_stacking_results.json
outputs/volatility10_lstm_stacking_predictions.csv
outputs/volatility10_lstm_stacking_predictions.png
outputs/volatility10_lstm_attention.csv
```

## Sonuç Yorumu

Deneyler şunu gösteriyor:

- Bu veri setinde Brent günlük getiri tahmini neredeyse gürültüye (noise) yakın.
- Daha karmaşık modeller günlük getirilerde performansı otomatik olarak artırmıyor.
- OVX ve jeopolitik risk göstergeleri, fiyat yönü için değil volatilite için daha kullanılabilir bilgi taşıyor.
- En savunulabilir final model, 10 günlük forward Brent volatilitesi için XGBoost ve Attention BiLSTM'in basit hibrit ortalaması.

Bu çerçeveleme, doğrudan Brent fiyat tahmini iddiasından daha güçlü; akademik/raporlama amaçları için daha uygun.
