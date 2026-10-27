2.4 GHz SLOTLU PATCH ANTEN VEKIL MODELI - CALISTIRMA REHBERI (VS Code)

KLASOR ICERIGI
  app.py              : Arayuzlu uygulama (Streamlit). Sunumda canli gosterilecek.
  deney_defteri.ipynb : Deneyler (model karsilastirma, tekrarli dogrulama, veri azaltma).
  DataSet.csv         : Zenodo acik veri seti (55.053 satir).
  requirements.txt    : Uygulamanin kutuphaneleri (internete yayinlarken de bu kullanilir).
  requirements-dev.txt: Ek olarak test ve deney defteri icin gerekenler.
  tests/test_app.py   : Otomatik testler (pytest).

1) KLASORU AC
  VS Code > File > Open Folder > bu klasoru sec.
  Sol alttan Python yorumlayicisi (Python 3.10+) secili olsun.
  (Extensions: "Python" ve "Jupyter" eklentileri kurulu olmali.)

2) KUTUPHANELERI KUR (bir kez)
  VS Code'da Terminal > New Terminal, sonra:
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements-dev.txt
  (PowerShell "scripts disabled" hatasi verirse once sunu calistir:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

3) UYGULAMAYI CALISTIR
    streamlit run app.py
  Tarayici otomatik acilir (acilmazsa terminaldeki http://localhost:8501 adresine git).
  Durdurmak icin terminalde Ctrl+C.

4) DENEY DEFTERINI CALISTIR
  deney_defteri.ipynb dosyasini VS Code'da ac, sag ustten Kernel olarak .venv'i sec,
  hucreleri yukaridan asagiya sirayla calistir (Run All).
  Her hucre bagimsizdir; sonuclar artifacts/ klasorune yazilir.
  Tekrarli dogrulama hucresi (5b) birkac dakika surer.

5) TESTLERI CALISTIR (isteğe bagli, ~10 saniye)
    pytest -q
  6 test: uygulama aciliyor mu, veri sayilari (53 anten), tahmin makul aralikta mi,
  aday listesi belirsizlik filtresinden geciyor mu, siradaki simulasyon puani siralamasi,
  ayni tohumla ayni sonuc.

UYGULAMA SEKMELERI
  1 Veri | 2 Tahmin (S11 egrisi + guvenilirlik) | 3 Aday oner | 4 Siradaki simulasyon
  5 Model ve sonuclar (karsilastirma, denemeler, kalibrasyon, uzak geometri testi)
  6 Nasil calisiyor (PEAS, akis semasi, sozde kod, ML/DL gecis)

INTERNETTE YAYINLAMA (Streamlit Community Cloud, ucretsiz, ~5 dakika)
  a) github.com'da yeni bir PUBLIC depo ac (ornegin s11-vekil-model).
  b) Bu klasordeki dosyalari depoya yukle: app.py, DataSet.csv, requirements.txt (+ isteğe bagli
     README.txt, tests/, deney_defteri.ipynb). .venv klasorunu YUKLEME.
  c) share.streamlit.io adresine GitHub hesabinla gir > "Create app" > depoyu sec,
     "Main file path" = app.py > Deploy.
  d) Birkac dakika sonra https://....streamlit.app adresi olusur. Bu adresi proje formundaki
     "(eklenecek)" yerine ve raporun kapagina yaz. Acilista uygulama 20-30 sn modeli egitir.
  Not: Ucretsiz uygulama bir sure kullanilmazsa uyur; sunumdan once adresi bir kez ac.

SIK SORUNLAR
  - "ModuleNotFoundError": sanal ortam aktif degil veya pip install yapilmadi.
  - "streamlit tanınmıyor": once .venv\Scripts\activate komutunu calistir.
  - DataSet.csv bulunamadi: terminal, app.py ile ayni klasorde olmali.
