🎵 Müzik Notaları Asistanı (RAG Chatbot)

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, RAG (Retrieval Augmented Generation) mimarisine dayanan, müzik teorisi üzerine soruları yanıtlayan bir yapay zeka asistanıdır. Web arayüzü Streamlit kullanılarak hazırlanmıştır.

🚀 1. Projenin Amacı

Amaç, belirli bir müzik teorisi PDF'i üzerinden özelleştirilmiş bilgi sağlayabilen bir sohbet robotu geliştirmektir. Kullanıcılar, doğal dil kullanarak temel müzik kavramları (nota, ritim, armoni vb.) hakkında hızlı, doğru ve belgelendirilmiş cevaplar almayı hedefler.

📚 2. Veri Seti Hakkında Bilgi

Adı: Temel Müzik Eğitimi.pdf

İçerik: Proje, temel seviyede müzik teorisi konularını (gamlar, notasyon, zaman ölçüleri, akorlar) içeren hazır bir PDF eğitim dokümanı kullanılarak eğitilmiştir.

Hazırlanış Metodolojisi: Hazır bir doküman kullanıldığı için toplama veya hazırlık süreci yürütülmemiştir. Doküman, RAG zincirine beslenmeden önce LangChain'in PyPDFLoader ve RecursiveCharacterTextSplitter araçlarıyla parçalanmıştır.

🛠️ 3. Kullanılan Yöntemler ve Çözüm Mimarisi

Bu proje, Retrieval Augmented Generation (RAG) mimarisini kullanmaktadır.

RAG Mimarisi Adımları:

Veri Yükleme: Temel Müzik Eğitimi.pdf dosyası yüklenir.

Parçalama (Chunking): RecursiveCharacterTextSplitter kullanılarak dokümanlar 1000 karakterlik parçalara ayrılır (chunk_overlap=200).

Gömme (Embedding): Parçalar, Google'ın models/text-embedding-004 modeli kullanılarak sayısal vektörlere dönüştürülür.

Vektör Veritabanı: Vektörler, ChromaDB'ye kaydedilir (yerel olarak ./chroma_db dizininde).

Geri Alma (Retrieval): Kullanıcı bir soru sorduğunda, retriever en ilgili 3 doküman parçasını (search_kwargs={"k": 3}) veritabanından çeker.

Cevap Üretme: Çekilen bu bağlam (context) ve kullanıcı sorusu, gemini-2.5-flash modeline gönderilerek nihai Türkçe cevap oluşturulur.

Teknolojiler:

RAG Framework: LangChain

Gömme Modeli: Google models/text-embedding-004

Üretken Model: Google gemini-2.5-flash

Vektör Veritabanı: ChromaDB

Web Arayüzü: Streamlit

🎯 4. Elde Edilen Sonuçlar

Chatbot, yalnızca eğitildiği PDF içeriğiyle sınırlı kalarak müzik teorisi sorularını Türkçe olarak yanıtlamaktadır. Yanıtlar, Gemini'ın üretken gücüyle harmanlanmakta ve cevapla birlikte kaynak (sayfa numarası ve dosya adı) bilgisi sunularak güvenilirlik artırılmıştır. Geliştirme aşamasında yaşanan API model uyumsuzluk sorunları (404 ve 400 hataları) başarıyla çözülerek sistemin stabil çalışması sağlanmıştır.

⚙️ 5. Çalışma Kılavuzu

Projenin yerel makinede çalıştırılabilmesi için gerekli adımlar aşağıdadır:

1. Ön Koşullar

Python 3.8+

Git (GitHub'a yükleme yapmak için)

Google Gemini API Key veya Service Account dosyası (service-account.json).

2. Ortam Kurulumu

Sanal Ortam Oluşturma ve Aktivasyon:

python -m venv venv_muzik
# Windows
.\venv_muzik\Scripts\activate
# Linux/macOS
source venv_muzik/bin/activate


Gerekli Kütüphaneleri Kurma:

pip install -r requirements.txt


(Bu dosyayı, AŞAMA 1'de anlattığımız pip freeze > requirements.txt komutuyla oluşturmuş olmanız gerekmektedir.)

Yapılandırma: service-account.json dosyasını ana proje klasörüne yerleştirin.

3. Uygulamayı Başlatma

Projenin ana klasöründe aşağıdaki komutu çalıştırarak Streamlit uygulamasını başlatın:

streamlit run chatbot.py


🌐 6. Web Arayüzü & Product Kılavuzu

Deploy Linki

[Lütfen buraya projenizin dağıtıldığı (Streamlit Cloud vb.) linki yapıştırın.]

Çalışma Akışı ve Test Senaryoları

Uygulama açıldığında, önce "Müzik teorisi belgeleri yükleniyor..." spinner'ı görünür. Bu aşamada Vektör Veritabanı kontrol edilir/oluşturulur.

Veritabanı hazır olduğunda, sohbet kutusu aktif hale gelir.

Kullanıcı, alt kısımdaki giriş kutusuna müzik teorisiyle ilgili sorularını yazar.

Cevap, sohbet penceresinde, kaynak olarak kullanılan sayfa bilgisiyle birlikte görüntülenir.

Test Edilebilecek Temel Kabiliyet:

Doğru bilgi çıkarımı: "Do majör gamının notaları nedir?" gibi doğrudan PDF'te yer alan bir sorunun doğru yanıtlanması.

Hata yönetimi: PDF'te olmayan bir soru sorulduğunda ("Mars'ta hava durumu nasıl?"), botun "Elimdeki müzik teorisi bilgileri bu soruyu tam olarak yanıtlamak için yetersizdir." yanıtını vermesi beklenir.