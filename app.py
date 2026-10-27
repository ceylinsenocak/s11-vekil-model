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
    BOLUM 6  Arayuz (kenar cubuk + 6 sekme)
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
# hem f_res hem S11_min hatasini dusurdu (Sekme 5).
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


st.title("2.4 GHz slotlu patch anten: vekil model")
st.caption("Anten ölçülerinden rezonans frekansı ve S11 çukur derinliğini tahmin eder. "
           "Yapay Zekâya Giriş vize projesi prototipi.")

with st.sidebar:
    st.header("Anten ölçüleri (mm)")
    olculer = []
    for k in OZELLIKLER:
        olculer.append(st.slider(ADLAR[k], float(alt[k]), float(ust[k]), step=0.1, key=f"olcu_{k}"))
    st.caption("Aralıklar, veri setinde gözlenen aralıklardır.")

sekme_veri, sekme_tahmin, sekme_aday, sekme_sonraki, sekme_model, sekme_nasil = st.tabs(
    ["1. Veri", "2. Tahmin", "3. Aday öner", "4. Sıradaki simülasyon",
     "5. Model ve sonuçlar", "6. Nasıl çalışıyor"])

# ---- Sekme 1: Veri ---------------------------------------------------------
with sekme_veri:
    st.subheader("Veri seti: 55.053 satır ama kaç anten?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Satır sayısı", f"{len(df):,}".replace(",", "."))
    c2.metric("Gerçek örnek (farklı anten)", len(geo))
    c3.metric("Değişen parametre", len(OZELLIKLER))
    st.write("Her satır bir **(anten, frekans)** çiftidir. Aynı anten 1.8–2.8 GHz arasında "
             "yaklaşık bin frekans noktasında simüle edilmiştir. Bu yüzden asıl örnek sayısı 53'tür.")
    st.markdown("**Ham veriden ilk satırlar**")
    st.dataframe(df.head(8))
    st.markdown("**Her anten için çıkardığımız iki hedef (53 satır)**")
    st.dataframe(geo.round(3), height=250)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i in np.random.default_rng(0).choice(len(geo), 6, replace=False):
        m = (df[OZELLIKLER] == geo.loc[i, OZELLIKLER]).all(axis=1)
        ax.plot(df.loc[m, "f"], df.loc[m, "s11"], lw=1)
    ax.axhline(HEDEF_S11, color="k", ls="--", lw=0.8)
    ax.set(xlabel="Frekans (GHz)", ylabel="S11 (dB)", title="Altı örnek antenin S11 eğrisi")
    st.pyplot(fig)

# ---- Sekme 2: Tahmin --------------------------------------------------------
with sekme_tahmin:
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
        guven = ("error", "Güvenilirlik: düşük")
    (st.success if frekans_tamam else st.warning)(
        "Frekans hedefte (±20 MHz)" if frekans_tamam
        else f"Frekans hedeften {abs(r['f']-HEDEF_FREKANS)*1000:.0f} MHz uzak")
    (st.success if s11_tamam else st.warning)("S11 ≤ −10 dB" if s11_tamam else "S11 −10 dB üstünde")
    getattr(st, guven[0])(guven[1])
    st.caption(f"Veri setindeki en yakın antene uzaklık: {r['uzaklik']:.2f} "
               f"(tipik komşu uzaklığı {komsu_tipik:.2f}). Uzaklık arttıkça tahmine güven azalır.")

    # Grafik: modelin tahmin ettigi S11 egrisi + referans olarak en yakin bilinen antenin gercek (CST) egrisi
    egri_ref = egriler[r["en_yakin"]]
    alt_sinir_egri = min(-30, r["egri"].min() - 2, egri_ref.min() - 2)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(FREKANS_IZGARASI, egri_ref, color="#9aa3b2", lw=1.2, label="En yakın bilinen anten (CST sonucu, referans)")
    ax.plot(FREKANS_IZGARASI, r["egri"], color="#2e5597", lw=2, label="Modelin tahmin ettiği S11 eğrisi")
    ax.fill_between(FREKANS_IZGARASI, r["egri"], HEDEF_S11, where=r["egri"] < HEDEF_S11,
                    color="#2e5597", alpha=0.15, label="Tahmini −10 dB altı bölge")
    ax.axvline(HEDEF_FREKANS, color="green", ls="--", lw=1, label="2.4 GHz")
    ax.axhline(HEDEF_S11, color="orange", ls="--", lw=1, label="−10 dB")
    ax.scatter([r["f"]], [r["s"]], color="red", zorder=5, label="Doğrudan modelin çukur tahmini")
    ax.set(xlabel="Frekans (GHz)", ylabel="S11 (dB)", ylim=(alt_sinir_egri, 1), xlim=(2.1, 2.7))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Mavi eğri ayrı bir model (PCA + Gaussian Process) ile tahmin edilir; kırmızı nokta ise "
               "f_res ve S11_min için eğitilen doğrudan modellerin çıktısıdır. İkisi birbirinden birkaç MHz "
               "sapabilir (eğri modelinde çukur yeri hatası ≈ 9 MHz). Gri çizgi tahmin değil, referanstır.")

# ---- Sekme 3: Aday öner -----------------------------------------------------
with sekme_aday:
    st.subheader("Hedefe uygun aday geometriler")
    st.write("Hedef: rezonans frekansı 2.4 GHz'e yakın ve S11 ≤ −10 dB. Adaylar, veri setindeki antenlerin "
             "küçük sapmalarıyla üretilir; belirsizliği yüksek olanlar elenir.")
    if st.button("Aday ara"):
        st.session_state["adaylar"] = aday_ara(geo, olcekleyici, X, model_f, model_s)
    if "adaylar" in st.session_state:
        t = st.session_state["adaylar"]
        st.dataframe(t.drop(columns="puan").round(2))
        sec = st.selectbox("Bir adayı Tahmin sekmesinde incelemek için seç", range(1, len(t) + 1))
        st.button("Seçilen adayı yükle", on_click=adayi_yukle, args=(t.iloc[sec - 1],))
    st.caption("Dikkat: S11 tahminleri belirsizdir (± yaklaşık 10 dB). Bu yüzden adaylar esas olarak "
               "rezonans frekansına göre sıralanır. Öneriler simülatörle doğrulanmamıştır.")
# ---- Sekme 4: Siradaki simulasyon ------------------------------------------
with sekme_sonraki:
    st.subheader("Sıradaki simülasyonu hangi geometride çalıştırmalı?")
    st.write("Elimizde yalnızca 53 gerçek örnek var. Yeni bir CST/HFSS simülasyonu pahalıdır, bu yüzden "
             "rastgele değil, **modelin en az emin olduğu** geometriyi seçmek mantıklıdır: o örnek modele "
             "en çok bilgiyi katar. Bu, aktif öğrenmenin ilk adımıdır (ML/DL geçiş haritasındaki sonraki aşama).")
    if st.button("Önerileri hesapla"):
        st.session_state["sonraki"] = siradaki_simulasyon(geo, olcekleyici, X, model_f, model_s, seed=1)
    if "sonraki" in st.session_state:
        t = st.session_state["sonraki"]
        st.dataframe(t.round(2))
        sec = st.selectbox("Bir geometriyi Tahmin sekmesinde incelemek için seç",
                           range(1, len(t) + 1), key="sec_sonraki")
        st.button("Seçileni yükle", on_click=adayi_yukle, args=(t.iloc[sec - 1],), key="yukle_sonraki")
    st.caption("Bu yöntem basit bir belirsizlik örneklemesidir (en büyük belirsizlik = en yüksek puan). "
               "Önerilen geometriler simülatörde çalıştırılmadı; sonraki adım budur.")

# ---- Sekme 5: Model ve sonuclar --------------------------------------------
with sekme_model:
    st.subheader("Model")
    st.write("- **Gaussian Process** (Matern 5/2 çekirdek + gürültü). Üç ayrı model: f_res, S11_min ve tüm S11 eğrisi.\n"
             "- S11_min, çukur derinliği çarpık dağıldığı için **log uzayında** (ln(−S11)) öğrenilir.\n"
             "- Girdi: 7 değişen geometri parametresi. Sabit olanlar dışarıda bırakıldı.\n"
             "- Tahminin yanındaki ± değeri GP'nin kendi belirsizliğidir.")

    st.subheader("1) Model karşılaştırması")
    st.caption("Anten bazlı doğrulama (aynı antenin satırları hem eğitimde hem testte olmaz): "
               "5 kat × 20 tekrar, n = 53 anten, ortalama ± std. Tüm modeller aynı bölmelerle sınandı.")
    st.table(pd.DataFrame({
        "Model": ["baz (ortalama)", "Ridge", "Random Forest", "GP (RBF, ilk sürüm)", "GP (son sürüm)"],
        "f_res MAE (MHz)": ["16.0", "10.6 ± 0.6", "7.8 ± 0.7", "7.2 ± 1.7", "6.2 ± 0.6"],
        "S11_min MAE (dB)": ["8.25", "8.74 ± 0.52", "5.91 ± 0.35", "5.64 ± 0.37", "5.06 ± 0.44"]}))
    st.write("Son sürüm GP, baz modele göre rezonans frekansı hatasını ortalama **%61** (en kötü bölmede %55; "
             "hedef %40: karşılandı), S11 çukur derinliği hatasını ortalama **%39** (en kötü bölmede %27; "
             "hedef %40: çok yakın ama altında) azaltır.")
    st.info("Dürüst not: S11_min için RMSE (büyük hatalara daha duyarlı ölçüt) log + Matern ile 7.98 → 8.40 dB "
            "oldu (RF: 8.32). Yani tipik hata (MAE) düştü, ama derin çukurlarda büyük sapmalar sürüyor; "
            "log dönüşümü geri çevrilirken büyük değerleri büyütüyor. Sıralama ölçüte göre biraz değişir.")

    st.subheader("2) S11_min için denenen iyileştirmeler")
    st.caption("5 kat × 10 tekrar, tohum 42. Seçim bu bölmelerde yapıldı; sonuç tablodaki taze bölmelerle "
               "(tohum 123) doğrulandı, böylece seçim yanlılığı kontrol edildi.")
    st.table(pd.DataFrame({
        "Deneme": ["V0 RBF (ilk sürüm)", "V1 + oran özellikleri", "V2 log dönüşümü", "V3 Matern",
                   "V4 log + Matern (seçilen)", "V5 oran + log"],
        "S11_min MAE (dB)": ["5.97", "5.74", "5.56", "5.73", "5.01", "5.56"]}))
    st.write("Eğri modeli (PCA + GP) denemesi: S11 çukur hatası 4.6 dB ile benzer, fakat rezonans frekansı hatası "
             "9.1 MHz ile doğrudan modelden (6.2 MHz) kötü. Bu yüzden sayılar doğrudan modellerden, eğri ise "
             "görsel destek olarak kullanılır.")

    st.subheader("3) Belirsizlik gerçekten güvenilir mi?")
    st.table(pd.DataFrame({
        "Ölçüt": ["%68 aralığı gerçeği kapsama oranı", "%95 aralığı gerçeği kapsama oranı",
                  "Belirsizlik ile gerçek hata korelasyonu"],
        "f_res": ["%68", "%85", "0.30"],
        "S11_min": ["%67", "%83", "0.33 (RBF ile 0.03)"]}))
    st.write("%68 aralığı iyi kalibre, %95 aralığı biraz fazla iddialı (%83–85 kapsıyor, ideali %95). "
             "Yani modelin ± değeri küçük örnek yüzünden hafif iyimser; yorumlarken bunu hesaba katın. "
             "Log + Matern, belirsizliği hatayla ilişkili hale getirdi (0.03 → 0.33).")

    st.subheader("4) Veriden uzak antenlerde ne olur?")
    st.caption("Her antenin en yakın komşusuna uzaklığına göre en uzak 10 anten teste ayrıldı, geri kalanıyla eğitildi. "
               "Karşılaştırma: rastgele ayrılan 10 anten (20 tekrar).")
    st.table(pd.DataFrame({
        "Test grubu": ["Rastgele 10 anten", "En uzak 10 anten"],
        "f_res MAE (MHz)": ["6.9", "18.8"],
        "Modelin bildirdiği f_res belirsizliği (MHz)": ["9.9", "24.3"],
        "S11_min MAE (dB)": ["4.9", "6.2"],
        "Modelin bildirdiği S11 belirsizliği (dB)": ["11.0", "17.0"]}))
    st.write("Veriden uzaklaşınca hata gerçekten artar (f_res için yaklaşık 2.7 kat), ama model bunu "
             "kendi belirsizliğini büyüterek bildirir. Güvenilirlik rozeti bu yüzden vardır.")

    st.subheader("5) Veri miktarının etkisi")
    st.write("Eğitimde kullanılan anten sayısı 10 → 20 → 30 → 40 olurken GP'nin f_res hatası "
             "10.7 → 9.8 → 6.7 → 5.0 MHz düştü (baz model ≈ 13.5–15.4 MHz; ilk sürüm RBF çekirdekle ölçüldü). "
             "Daha fazla simülasyon eklemek bu yüzden anlamlıdır.")

    st.subheader("Sınırlamalar")
    st.write("- Yalnızca 53 gerçek örnek; parametre uzayı seyrek ve aralıklar dar.\n"
             "- S11 çukur derinliği hâlâ zor tahmin edilir (hedef %40, ulaşılan %39); büyük hatalar (RMSE) azalmadı.\n"
             "- Veri tek bir anten yapısına ve CST simülasyonlarına aittir; fiziksel ölçüm yoktur.\n"
             "- Aday ve sıradaki simülasyon önerileri simülatörle doğrulanmamıştır.\n"
             "- %95 belirsizlik aralıkları hafif iyimserdir (gerçek kapsama ≈ %84).")

# ---- Sekme 6: Nasil calisiyor ----------------------------------------------
with sekme_nasil:
    st.subheader("PEAS: yapay zekâ sistem modeli")
    st.table(pd.DataFrame({
        "Öğe": ["Performans ölçütü", "Ortam", "Eyleyiciler (çıktılar)", "Algılayıcılar (girdiler)"],
        "Bu projede": [
            "f_res MAE (MHz), S11_min MAE (dB), baz modele göre iyileşme, belirsizlik kapsama oranı",
            "7 geometri parametresi; 53 CST simülasyonu (1.8–2.8 GHz); açık Zenodo verisi",
            "Tahmin ± belirsizlik, S11 eğrisi, güvenilirlik rozeti, aday listesi, sıradaki simülasyon önerisi",
            "Kenar çubuktaki anten ölçüleri (kullanıcı), DataSet.csv"]}))
    st.caption("Sistem türü: veriden öğrenen, belirsizliğini bildiren, karar destek amaçlı bir vekil model. "
               "Anteni kendisi tasarlamaz veya üretmez; öneri sunar.")

    st.subheader("Akış şeması")
    st.graphviz_chart("""
    digraph G {
      rankdir=TB; nodesep=0.3;
      node [shape=box, style="rounded,filled", fillcolor="#F5F3EE", color="#14213D", fontname="Helvetica"];
      a [label="DataSet.csv\\n55.053 satır"];
      b [label="Antene göre grupla\\n53 geometri"];
      c [label="Hedefleri çıkar\\nf_res, S11_min, S11 eğrisi"];
      d [label="3 Gaussian Process eğit\\n(f_res, ln(-S11_min), eğri)"];
      e [label="Kullanıcı ölçüleri girer"];
      f [label="Tahmin ± belirsizlik\\n+ S11 eğrisi"];
      g [shape=diamond, fillcolor="#FFE9D6", label="En yakın antene\\nuzaklık tipik\\nuzaklıktan küçük mü?"];
      h [label="Güvenilirlik: yüksek / orta"];
      i [label="Güvenilirlik: düşük\\n(uyarı göster)"];
      j [label="Aday ara veya\\nsıradaki simülasyonu öner"];
      a -> b -> c -> d -> e -> f -> g;
      g -> h [label="evet"]; g -> i [label="hayır"];
      h -> j; i -> j;
    }
    """)

    st.subheader("Sözde kod")
    st.code("""EĞİTİM
  veri  <- DataSet.csv oku
  antenler <- veriyi 7 geometri parametresine göre grupla       # 55.053 satır -> 53 anten
  her anten için: f_res <- S11'in en düşük olduğu frekans
                  s11_min <- o noktadaki S11
                  eğri <- S11'i 201 noktalık ortak frekans ızgarasına yerleştir
  ölçekle(geometri parametreleri)
  GP_f   <- GP.eğit(geometri, f_res)
  GP_s   <- GP.eğit(geometri, ln(-s11_min))
  GP_eğri <- PCA(eğri, 8 bileşen) + her bileşen için GP

TAHMİN (ölçüler girilince)
  f, f_std <- GP_f.tahmin(ölçüler)
  s, s_std <- -exp(GP_s.tahmin(ölçüler))
  eğri     <- GP_eğri.tahmin(ölçüler)
  d <- en yakın eğitim antenine uzaklık
  eğer d <= tipik_komşu_uzaklığı:   güven <- "yüksek"
  değilse eğer d <= 2 * tipik:      güven <- "orta"
  değilse:                          güven <- "düşük"

ADAY ARAMA
  3000 kez: rastgele bir antenin ölçülerini küçük oynat
  puan <- |f - 2.4| * 1000 + 5 * max(0, s + 10) + f_std
  f_std >= 15 MHz olanları ele; en düşük puanlı 8 adayı göster

SIRADAKİ SİMÜLASYON
  3000 kez: ölçüleri daha büyük oynat
  puan <- f_std / medyan(f_std) + s_std / medyan(s_std)
  en yüksek puanlı 5 geometriyi öner (model en az burada emin)""", language="text")

    st.subheader("Tekrarlanabilirlik")
    st.write("Tüm modeller sabit rastgelelik tohumuyla (random_state) eğitilir; testler `pytest` ile çalışır "
             "(`tests/test_app.py`). Deneyler `deney_defteri.ipynb` dosyasında bağımsız hücreler olarak durur.")

    st.subheader("Makine öğrenmesi → derin öğrenme geçişi")
    st.write("1. **Bugün:** Gaussian Process ve Random Forest ile vekil model (53 anten).\n"
             "2. **Sonraki adım:** Aktif öğrenme: 4. sekmedeki öneriyi simülatörde çalıştır, veriye ekle, modeli yenile.\n"
             "3. **İleri:** Yüzlerce simülasyon toplanınca tüm S11 eğrisini tahmin eden sinir ağı ve çok doğruluklu yöntemler.")
