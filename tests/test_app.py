"""Uygulamanin temel davranisini sinar. Calistirma (proje klasorunde): pytest -q"""
import os
import pytest
from streamlit.testing.v1 import AppTest

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def uygulama():
    os.chdir(KOK)  # DataSet.csv burada
    at = AppTest.from_file(os.path.join(KOK, "app.py"), default_timeout=600)
    at.run()
    return at


def test_uygulama_hatasiz_aciliyor(uygulama):
    assert not uygulama.exception


def test_veri_sayilari(uygulama):
    degerler = {m.label: m.value for m in uygulama.metric}
    assert degerler["Gerçek örnek (farklı anten)"] == "53"
    assert degerler["Değişen parametre"] == "7"


def test_tahmin_makul_aralikta(uygulama):
    f = next(m for m in uygulama.metric if m.label == "Rezonans frekansı")
    s = next(m for m in uygulama.metric if m.label == "Minimum S11")
    f_mhz = float(f.value.split()[0])
    s_db = float(s.value.split()[0])
    assert 2300 <= f_mhz <= 2500          # veri setinde f_res 2.38-2.48 GHz
    assert -70 <= s_db <= 0               # S11 dB cinsinden negatif


def test_aday_arama_belirsizlik_filtresi(uygulama):
    dugme = next(b for b in uygulama.button if b.label == "Aday ara")
    dugme.click().run()
    assert not uygulama.exception
    tablo = uygulama.session_state["adaylar"]
    assert 0 < len(tablo) <= 8
    assert (tablo["f_std (MHz)"] < 15).all()         # belirsiz adaylar elenmis olmali
    assert tablo["puan"].is_monotonic_increasing      # en iyi aday en uste


def test_siradaki_simulasyon_en_belirsizi_ustte_veriyor(uygulama):
    dugme = next(b for b in uygulama.button if b.label == "Önerileri hesapla")
    dugme.click().run()
    assert not uygulama.exception
    tablo = uygulama.session_state["sonraki"]
    assert len(tablo) == 5
    assert tablo["belirsizlik puanı"].is_monotonic_decreasing


def test_ayni_tohum_ayni_sonuc():
    """Tekrarlanabilirlik: ayni tohumla uretilen aday listesi ayni olmali."""
    import importlib.util, sys
    os.chdir(KOK)
    at1 = AppTest.from_file(os.path.join(KOK, "app.py"), default_timeout=600).run()
    at2 = AppTest.from_file(os.path.join(KOK, "app.py"), default_timeout=600).run()
    m1 = [m.value for m in at1.metric]
    m2 = [m.value for m in at2.metric]
    assert m1 == m2
