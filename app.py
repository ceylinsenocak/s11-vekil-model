"""
2.4 GHz slotlu patch anten icin vekil model uygulamasi (Streamlit).

Calistirma (VS Code terminali, bu klasorde):
    streamlit run app.py

Dosya yapisi (yukaridan asagiya okunur, her bolum bir adimdir):
    BOLUM 1  Veriyi yukle
    BOLUM 2  Satirlari geometriye cevir (55.053 satir -> 53 geometri) + S11 egri matrisi
    BOLUM 3  Modeli egit (Gaussian Process: f_res, S11_min ve tum S11 egrisi)
    BOLUM 4  Tahmin fonksiyonu (+ guvenilirlik icin en yakin komsu)
    BOLUM 5  Aday arama ve "siradaki simulasyon" onerisi
    BOLUM 6  Arayuz (kenar cubukta olculer, ana sayfada tahmin ve aday onerisi)
"""
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

warnings.filterwarnings("ignore")  # sklearn uyarilari terminali kalabaliklastirmasin

st.set_page_config(page_title="Anten S11 Vekil Model", page_icon="📡", layout="wide")

# Veri setinde degisen 7 geometri parametresi (digerleri sabit) ve ekranda gorunen adlari
OZELLIKLER = ["w_sub", "l_line", "w_slot", "l_slot", "w_patch", "l_sub", "l_patch"]
ADLAR = {
    "w_sub": "Alt taş genişliği", "l_line": "Besleme hattı uzunluğu",
    "w_slot": "Slot genişliği", "l_slot": "Slot boyu",
    "w_patch": "Patch genişliği", "l_sub": "Alt taş uzunluğu", "l_patch": "Patch boyu",
}
HEDEF_FREKANS = 2.4   # GHz
HEDEF_S11 = -10.0     # dB: bunun altı "iyi uyum" kabul edilir


# ----------------------------------------------------------------------------
# BOLUM 1: Veriyi yukle
# ----------------------------------------------------------------------------
@st.cache_data
def veri_yukle(yol="DataSet.csv"):
    df = pd.read_csv(yol, sep=";")  # dosyada ayirici ';' dir
    df.columns = ["w_sub", "w_line", "l_line", "h_sub", "h_patch", "w_slot",
                  "l_slot", "w_patch", "l_sub", "l_patch", "f", "s11"]
    return df


# ----------------------------------------------------------------------------
# BOLUM 2: Satirlardan geometri tablosu cikar
# Her satir bir (anten, frekans) cifti. Ayni anten yuzlerce satirda tekrar eder.
# Bu yuzden antenleri gruplayip her biri icin iki sayi cikariyoruz:
#   f_res  : S11'in en dusuk oldugu frekans (rezonans frekansi)
#   s11_min: o noktadaki S11 degeri (cukurun derinligi)
# ----------------------------------------------------------------------------
@st.cache_data
def geometri_tablosu(df):
    def hedefler(x):
        i = x.s11.idxmin()
        return pd.Series({"f_res": x.loc[i, "f"], "s11_min": x.s11.min()})
    geo = df.groupby(OZELLIKLER).apply(hedefler).reset_index()
    return geo


# Ortak frekans izgarasi: her antenin S11 egrisini ayni 201 noktaya getiririz (53 x 201 matris),
# boylece egrilerin tamamini bir model ile tahmin edebiliriz.
FREKANS_IZGARASI = np.linspace(1.8, 2.8, 201)


@st.cache_data
def egri_matrisi(df, geo):
    satirlar = []
    for _, r in geo.iterrows():
        m = (df[OZELLIKLER] == r[OZELLIKLER]).all(axis=1)
        d = df[m].sort_values("f")
        satirlar.append(np.interp(FREKANS_IZGARASI, d["f"].values, d["s11"].values))
    return np.array(satirlar)


# ----------------------------------------------------------------------------
# BOLUM 3: Modeli egit
# Girdiler (olculer) ayni olcege getirilir (StandardScaler). Uc Gaussian Process:
#   1) f_res          (dogrudan)
#   2) s11_min        (log uzayinda: ln(-S11); cukurun derinligi carpik dagildigi icin)
#   3) tum S11 egrisi (egriler once PCA ile 8 sayiya indirilir, her sayi icin bir GP)
# Cekirdek: Matern(2.5) = RBF'ten daha "puruzlu" fonksiyonlara izin verir; deneylerde
# hem f_res hem S11_min hatasini dusurdu (rapordaki Tablo 1).
# ----------------------------------------------------------------------------
@st.cache_resource
def model_egit(geo, egriler):
    olcekleyici = StandardScaler().fit(geo[OZELLIKLER].values)
    X = olcekleyici.transform(geo[OZELLIKLER].values)

    def gp_kur(tekrar=3):
        cekirdek = ConstantKernel(1.0) * Matern(np.ones(len(OZELLIKLER)), nu=2.5) + WhiteKernel(0.1)
        return GaussianProcessRegressor(cekirdek, normalize_y=True,
                                        n_restarts_optimizer=tekrar, random_state=0)

    model_f = gp_kur().fit(X, geo["f_res"].values)
    model_s = gp_kur().fit(X, np.log(-geo["s11_min"].values))
    ort_egri = egriler.mean(axis=0)
    pca = PCA(8).fit(egriler - ort_egri)
    Z = pca.transform(egriler - ort_egri)
    egri_gp = [gp_kur(2).fit(X, Z[:, j]) for j in range(8)]
    return olcekleyici, X, model_f, model_s, (pca, ort_egri, egri_gp)


# ----------------------------------------------------------------------------
# BOLUM 4: Tahmin
# Donus: tahmin +- belirsizlik, tahmin edilen S11 egrisi ve verideki en yakin antene uzaklik.
# Uzaklik buyudukce (veriden uzaklastikca) tahmine guven azalir.
# ----------------------------------------------------------------------------
def s11_tahmin(model_s, x):
    """Log uzayindaki tahmini dB'ye cevirir. Belirsizlik yaklasik (delta yontemi)."""
    mu, sd = model_s.predict(x, return_std=True)
    return -np.exp(mu), np.exp(mu) * sd


def egri_tahmin(egri_modeli, x):
    pca, ort_egri, egri_gp = egri_modeli
    z = np.array([[g.predict(x)[0] for g in egri_gp]])
    return pca.inverse_transform(z)[0] + ort_egri


def tahmin_et(olculer, geo, olcekleyici, X, model_f, model_s, egri_modeli):
    x = olcekleyici.transform(np.array(olculer, dtype=float).reshape(1, -1))
    f, f_std = model_f.predict(x, return_std=True)
    s, s_std = s11_tahmin(model_s, x)
    mesafeler = np.linalg.norm(X - x, axis=1)
    en_yakin = int(np.argmin(mesafeler))
    return dict(f=f[0], f_std=f_std[0], s=s[0], s_std=s_std[0],
                egri=egri_tahmin(egri_modeli, x),
                en_yakin=en_yakin, uzaklik=float(mesafeler[en_yakin]))


def tipik_komsu_uzakligi(X):
    """Verideki her antenin en yakin komsusuna uzakligin ortancasi (guvenilirlik olcutu)."""
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


# ----------------------------------------------------------------------------
# BOLUM 5: Aday arama
# Veriden rastgele bir antenin olculerini al, kucuk rastgele degisiklik yap,
# modele sor, hedefe yakinligina gore puanla. Belirsizligi yuksek adaylari ele.
# ----------------------------------------------------------------------------
def aday_uret(geo, adet, sapma, seed=None):
    """Veriden rastgele antenler al, olculerini aralik genisliginin 'sapma' kadari oynat."""
    rng = np.random.default_rng(seed)
    ham = geo[OZELLIKLER].values
    alt, ust = ham.min(axis=0), ham.max(axis=0)
    taban = ham[rng.integers(0, len(ham), adet)]
    return np.clip(taban + rng.normal(0, sapma, taban.shape) * (ust - alt), alt, ust)


def aday_ara(geo, olcekleyici, X, model_f, model_s, adet=3000, sapma=0.05, seed=None):
    aday = aday_uret(geo, adet, sapma, seed)
    Xa = olcekleyici.transform(aday)
    f, f_std = model_f.predict(Xa, return_std=True)
    s, s_std = s11_tahmin(model_s, Xa)
    tablo = pd.DataFrame(aday, columns=OZELLIKLER)
    tablo["f_res (MHz)"] = f * 1000
    tablo["f_std (MHz)"] = f_std * 1000
    tablo["S11_min (dB)"] = s
    tablo["S11_std (dB)"] = s_std
    # puan: ne kadar dusukse o kadar iyi
    tablo["puan"] = (np.abs(f - HEDEF_FREKANS) * 1000
                     + 5 * np.clip(s - HEDEF_S11, 0, None)
                     + tablo["f_std (MHz)"])
    tablo = tablo[tablo["f_std (MHz)"] < 15]   # belirsizligi yuksek adaylari ele
    return tablo.sort_values("puan").head(8).reset_index(drop=True)


# "Siradaki simulasyonu oner" (aktif ogrenme fikri):
# Model nerede en az emin? Orayi simulatorde (CST/HFSS) calistirirsak veri setine en cok bilgiyi
# biz ekleriz. Adaylar yine mevcut antenlerin daha buyuk sapmalariyla uretilir (gercekci kalsin);
# puan = iki belirsizligin (frekans, S11) kendi ortancalarina bolunmus toplami.
def siradaki_simulasyon(geo, olcekleyici, X, model_f, model_s, adet=3000, sapma=0.15, seed=None):
    aday = aday_uret(geo, adet, sapma, seed)
    Xa = olcekleyici.transform(aday)
    f, f_std = model_f.predict(Xa, return_std=True)
    s, s_std = s11_tahmin(model_s, Xa)
    belirsizlik = f_std / np.median(f_std) + s_std / np.median(s_std)
    mesafe = np.array([np.linalg.norm(X - x, axis=1).min() for x in Xa])
    tablo = pd.DataFrame(aday, columns=OZELLIKLER)
    tablo["f_std (MHz)"] = f_std * 1000
    tablo["S11_std (dB)"] = s_std
    tablo["en yakın antene uzaklık"] = mesafe
    tablo["belirsizlik puanı"] = belirsizlik
    return tablo.sort_values("belirsizlik puanı", ascending=False).head(5).reset_index(drop=True)


# ----------------------------------------------------------------------------
# BOLUM 6: Arayuz
# ----------------------------------------------------------------------------
df = veri_yukle()
geo = geometri_tablosu(df)
egriler = egri_matrisi(df, geo)
olcekleyici, X, model_f, model_s, egri_modeli = model_egit(geo, egriler)
komsu_tipik = tipik_komsu_uzakligi(X)
alt, ust = geo[OZELLIKLER].min(), geo[OZELLIKLER].max()

# Varsayilan olarak hedefe en yakin gercek anteni ac
puan_gercek = (geo.f_res - HEDEF_FREKANS).abs() * 1000 + 5 * (geo.s11_min - HEDEF_S11).clip(lower=0)
varsayilan = geo.loc[puan_gercek.idxmin(), OZELLIKLER]
for k in OZELLIKLER:
    st.session_state.setdefault(f"olcu_{k}", float(varsayilan[k]))


def adayi_yukle(satir):
    for k in OZELLIKLER:
        st.session_state[f"olcu_{k}"] = round(float(satir[k]), 1)


st.title("📡 2.4 GHz patch anten S11 tahmini")
st.caption("Soldan anten ölçülerini ayarla; rezonans frekansı, S11 ve S11 eğrisi anında tahmin edilir.")

with st.sidebar:
    st.header("Anten ölçüleri (mm)")
    olculer = []
    for k in OZELLIKLER:
        olculer.append(st.slider(ADLAR[k], float(alt[k]), float(ust[k]), step=0.1, key=f"olcu_{k}"))
    st.caption("Aralıklar, veri setinde gözlenen aralıklardır.")
    st.button("↺ Başlangıç ölçülerine dön", on_click=adayi_yukle, args=(varsayilan,))


# ---- Tahmin ----------------------------------------------------------------
r = tahmin_et(olculer, geo, olcekleyici, X, model_f, model_s, egri_modeli)
a, b = st.columns(2)
a.metric("Rezonans frekansı", f"{r['f']*1000:.0f} MHz", f"± {r['f_std']*1000:.0f} MHz", delta_color="off")
b.metric("Minimum S11", f"{r['s']:.1f} dB", f"± {r['s_std']:.1f} dB", delta_color="off")

frekans_tamam = abs(r["f"] - HEDEF_FREKANS) * 1000 <= 20
s11_tamam = r["s"] <= HEDEF_S11
if r["uzaklik"] <= komsu_tipik:
    guven = ("success", "Güvenilirlik: yüksek")
elif r["uzaklik"] <= 2 * komsu_tipik:
    guven = ("warning", "Güvenilirlik: orta")
else:
    guven = ("error", "Güvenilirlik: düşük (ölçüler veri setinden uzak)")
d1, d2, d3 = st.columns(3)
with d1:
    (st.success if frekans_tamam else st.warning)(
        "Frekans 2.4 GHz'de (±20 MHz)" if frekans_tamam
        else f"Frekans hedeften {abs(r['f']-HEDEF_FREKANS)*1000:.0f} MHz uzak")
with d2:
    (st.success if s11_tamam else st.warning)("S11 ≤ −10 dB" if s11_tamam else "S11 −10 dB üstünde")
with d3:
    getattr(st, guven[0])(guven[1])

# Grafik: modelin tahmin ettigi S11 egrisi + referans olarak en yakin bilinen antenin gercek (CST) egrisi
egri_ref = egriler[r["en_yakin"]]
alt_sinir_egri = min(-30, r["egri"].min() - 2, egri_ref.min() - 2)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(FREKANS_IZGARASI, egri_ref, color="#9aa3b2", lw=1.2, label="En yakın bilinen anten (CST, referans)")
ax.plot(FREKANS_IZGARASI, r["egri"], color="#2e5597", lw=2, label="Tahmin edilen S11 eğrisi")
ax.fill_between(FREKANS_IZGARASI, r["egri"], HEDEF_S11, where=r["egri"] < HEDEF_S11,
                color="#2e5597", alpha=0.15, label="−10 dB altı bölge")
ax.axvline(HEDEF_FREKANS, color="green", ls="--", lw=1, label="2.4 GHz")
ax.axhline(HEDEF_S11, color="orange", ls="--", lw=1, label="−10 dB")
ax.scatter([r["f"]], [r["s"]], color="red", zorder=5, label="Çukur tahmini")
ax.set(xlabel="Frekans (GHz)", ylabel="S11 (dB)", ylim=(alt_sinir_egri, 1), xlim=(2.1, 2.7))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---- Aday oner -------------------------------------------------------------
st.divider()
st.subheader("Hedefe uygun tasarım öner")
st.write("2.4 GHz'e yakın rezonans ve S11 ≤ −10 dB hedefleyen aday ölçüleri arar.")
if st.button("Aday ara"):
    st.session_state["adaylar"] = aday_ara(geo, olcekleyici, X, model_f, model_s)
if "adaylar" in st.session_state:
    t = st.session_state["adaylar"]
    st.dataframe(t.drop(columns="puan").round(2))
    sec = st.selectbox("Bir adayı yukarıda incelemek için seç", range(1, len(t) + 1))
    st.button("Seçilen adayı yükle", on_click=adayi_yukle, args=(t.iloc[sec - 1],))

with st.expander("Not: tahminler ne kadar güvenilir?"):
    st.write("Model, 53 gerçek anten simülasyonundan öğrenir. Rezonans frekansı tahmini iyidir "
             "(ortalama hata ≈ 6 MHz); S11 derinliği daha zordur (ortalama hata ≈ 5 dB). "
             "Veri setinden uzak ölçülerde güvenilirlik düşer. Öneriler simülatörle doğrulanmamıştır.")
