# Laporan Proyek Machine Learning: Movie & Serial Recommendation System - Aisy Al Fawwaz - Universitas Airlangga
---
### Domain Project

> **Domain yang dipilih adalah berkaitan dengan industri perfilman dengan judul proyek "Movie & Serial Recommendation System"**.

* Latar belakang
Teknologi pada era modern saat ini memungkinkan sesorang untuk dapat melakukan aktivitas seperti mengakses internet,media sosial, belanja online, menonton film dan berbagai layanan yang tersedia melalui gadget. Media hiburan seperti streaming film termasuk menjadi hal yang banyak diminati dan penting dalam kehidupan karena dapat menghilangkan stress pada seseorang [1]. Industri penyedia layanan streming film yang memiliki banyak pengguna memanfaatkan data tersebut untuk membuat sebuah sistemrekomendasi yang dapat memberikan kemudahan kepada pengguna
dalam mencari sesuatu sesuai dengan karakteristik pengguna tersebut. Penelitian terkait sistem rekomendasi juga telah berlangsung selama hampir setengah abad hingga saat ini. Beberapa perusahaan besar juga telah menggunakan sistem rekomendasi pada sistem mereka untuk memberikan layanan yang terbaik seperti Amazon, Netflix dan lain sebagainya [2]. 

Secara umum, sistem rekomendasi memiliki dua kategori model yang dapat sering digunakan, yaitu *Collaborative Filtering,* dan *Content Based Filtering*[3]. *Collaborative filtering* digunakan untuk mengidentifikasi kesamaan antar pengguna dan memberikan rekomendasi item yang sesuai. Sistem ini merekomendasikan item yang disukai oleh kategori pengguna yangserupa. *Collaborative filtering* dibagi menjadi dua kategori yaitu *memory based* yang digunakan untuk menilai berdasarkan kesamaan antara pengguna atau item dan *model based filtering* untuk memprediksi peringkat pengguna dari item yang tidak diberi peringkat. Content based memberikan rekomendasi kepada penggunaberdasarkan pada profil preferensi pengguna dan hubungan antar deskripsi item. Hal itu dilakukan dengan memilih item yang palingmirip dengan item target menggunakan komputasi kesamaan(computing similarity based) berdasarkan fitur terkait menggunakan berbagai fungsi matematika. pada project kali ini akan digunakan model *content based filtering* untuk memberikan rekomendasi film & serial yang cocok bagi pengguna berdasarkan kemiripan genre antar film, sehingga pelanggan dapat dengan mudah menyesuaikan selera film mereka.
___
# Business Understanding
---
#### Problem Statements
berdasarkan latar belakang diatas, berikut ini rumusan masalah yang dapat diselesaikan pada proyek ini:

- Bagaimana memberikan rekomendasi judul serial & film berdasarkan kemiripan genre pada setiap judul movie & serial dari judul yang telah pelanggan masukan
 
#### Goals
- Membuat pelanggan lebih mudah menemukan judul serial & film yang cocok melalui bantuan sistem rekomendasi judul movie & serial berdasarkan genre yang dibuat.

#### Solution Statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
* Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
    * Mengecek masalah data yang kosong dengan melakukan pengecekan terlebih dahulu.
    * Menghitung besar/panjang data pada dataset terlebih dahulu kemudian mencoba untuk menguraikan jenis-jenis fitur pada kolom genre
    * mengurutkan data movieId dan menghapus data yg sama
    * Membuang judul film yg sama dengan library pandas dengan fungsi  ``` drop_duplicats()``` 
* Model ML yang digunakan pada projek ini adalah Content Based Filtering. Content-based filtering adalah pemfilteran berbasis konten di mana sistem ini memberikan rekomendasi untuk menebak apa yang disukai pengguna berdasarkan aktivitas pengguna tersebut. Content-based filtering bergantung pada penetapan atribut ke objek database sehingga algoritma mengetahui sesuatu tentang setiap objek. Atribut ini terutama bergantung pada produk, layanan, atau konten yang Anda rekomendasikan. ada beberapa kelebihan dan kekurangan dari model ini.
- Kelebihan :
    1. Tidak bergantung kepada user lain dalam memberikan rekomendasi yang ada
- Kekurangan :
    1. Hanya dapat digunakan untuk fitur yang sesuai, seperti gambar, film dan musik
    2. Barang / jasa yang di spesialisasi berlebihan, contoh pada content-based jarang sekali ditemukan rekomendasi yang kegunaannya berbeda padahal beberapa orang mungkin memiliki kepentingan yang berbeda. Misalkan seseorang yang baru saja membeli mobil sepertinya kurang pas apabila di berikan rekomendsasi barang yang serupa, namun akan cocok apabila barang yang direkomendasikan merupakan aksesoris dari mobil tersebut seperti (ban, lampu mobil, GPS dan akeseoris lain pelengkap mobil tersebut).
    3. Tidak mampu menentukan profil dari user baru.

# Data Understanding
---
![kaggle](https://postimg.cc/WDqZg70p)
Data pada project ini didapatkan dari kaggle, data disintesis dari IMDB adalah situs web populer untuk menilai peringkat film dan sibena. berisi sekitar 7000 film dan sinema paling populer di IMDB dengan adanya data Ideal untuk Analisis Data Eksplorasi.
Informasi dataset dapat dilihat pada tabel dibawah ini :
Jenis | Keterangan
--- | ---
Sumber | [Kaggle Dataset : Imdb Most popular Films and series](https://www.kaggle.com/datasets/mazenramadan/imdb-most-popular-films-and-series)
Usability | 8.24
Lisensi | CC0: Public Domain
Kategori | Recomendations movie & Serial Dataset
Jenis dan Ukuran Berkas | Zip (171 kB)

Pada data yang diunduh yakni imdb.csv berisi 6178 baris × 14 kolom. Kolom-kolom tersebut terdiri dari 13 buah kolom bertipe objek dan 1 buah kolom bertipe numerik (tipe data int64), Untuk penjelasan mengenai variabel-variable pada dapat dilihat sebagai berikut:
- **Name** merupakan parameter yang menyimpan judul film/serial
- **Date** merupakan parameter bernilai waktu perilisan film/serial
- **Rate**  merupakan parameter berisi rata-rata nilai rating menurut pengguna
- **Votes** merupakan parameter yg menyimpan jumlah pengguna yang melakukan voting
- **Genre** merupakan parameter yg menyimpan genre tiap film/serial
- **Duration** merupakan parameter bernilai durasi tiap film/serial
- **Type** merupakan parameter apakah jenis film atau serial
- **Certificate**
- **Episodes** merupakan parameter yg berisi nilai episode tiap serial
- **Nudity, Violence, Profanity, Alcohol, Frightening** merupakan parameter yg menyatakan berapa banyak unsur tersebut terkandung dalam film/serial

Berikut beberapa tahapan Data Understanding diantaranya sebagai berikut:
- Memuat Dataset ke dalam sebuah Dataframe menggunakan pandas
- ``` df.info()``` digunakan untuk mengecek tipe kolom pada dataset
- ```df.isnull().sum()``` digunakan untuk mengecek apakah ada kolom yg bernilai kosong
- ```df.describe()``` digunakan utk mendapatkan info statistik dari dataset
- ``` len(nama_variable.unique()) ```menghitung panjang data tiap variabel
- Menampilkan genre yg paling sering muncul 


# Data Preparation
---
Berikut adalah tahapan-tahapan dalam melakukan Persiapan data:
1. Menghitung jumlah data pada genre
2. Membersihkan data dengan melakukan drop pada kolom judul yang terduplikat
3. Melakukan *data transform* dengan cara menyusun ulang  penomoran index data 

Teknik yang digunakan pada langkah pengolahan data adalah memvektorisasi fungsi CountVectorizer dari library scikit-learn. CountVectorizer digunakan untuk mengubah teks yang diberikan menjadi vektor berdasarkan frekuensi (hitungan) setiap kata dalam teks.
CountVectorizer membuat matriks di mana setiap kata unik diwakili oleh kolom matriks dan setiap sampel teks dalam dokumen adalah deretan matriks. Nilai setiap sel tidak lain adalah jumlah kata dalam sampel teks yang diberikan.

Pada proses vektorisasi ini, digunakan metode sebagai berikut. 
1. ```fit``` metode berfungsi untuk melakukan perhitungan idf pada data
2. ```get_feature_names_out()``` berfungsi untuk melakukan mapping array dari fitur index integer ke fitur nama
3. ```fit_transform()``` berfungsi untuk mempelajari kosa kata dan Inverse Document Frequency (IDF) dengan memberikan nilai return berupa *document-term matrix*
4. ```todense()``` berfungsi untuk mengubah vektor tf-idf dalam bentuk matriks

# Modeling
---
Setelah dilakukan pra-pemrosesan pada dataset, langkah selanjutnya adalah *modeling* terhadap data. Pada tahap ini Model machine learning yang digunakan pada sistem rekomendasi ini adalah model _content-based filtering_ dengan mengukur  _Cosine Similarity_.. 
Model _content-based filtering_ ini bekerja dengan Pendekatan ini digunakan untuk metode yang akan mengambil informasi yang berguna dari item yang telah diekstraksi. Informasi ini harus dipastikan merupakan informasi yang baik dan dapat dipastikan akan menjadi relevan terhadap pengguna. Proses ektraksi terhadap item yang digunakan akan memperbesar kemungkinan munculnya item baru yang belum pernah terlihat sebelumnya. Pada dasarnya metode ini sangat bergantung pada perilaku pengguna. Asumsi utama di bawah pendekatan berbasis konten adalah bahwa item atau dokumen dapat diidentifikasi oleh serangkaian fitur yang diekstraksi langsung dari konten mereka sesuai dengan derajat kemiripan tiap fitur

_Cosine Similarity_ adalah teknik pengukuran kesamaan yang bekerja dengan mengukur kesamaan dua vektor dan menentukan apakah kedua vektor menunjuk ke arah yang sama dengan menghitung sudut kosinus antara kedua vektor. Semakin kecil sudut cosinus, semakin besar _cosine similarity_.
Fungsi *cosine similarity* berfungsi dengan perhitungan yang sering digunakan untuk menghitung kesamaan antar objek. Secara umum, fungsi kesamaan adalah fungsi yang mengambil dua objek dalam bilangan real (0 dan 1) dan mengembalikan nilai kesamaan antara dua objek dalam bilangan real. Kesamaan kosinus adalah salah satu metode pengukuran kesamaan yang paling populer. Metode ini digunakan untuk menghitung kosinus sudut antara dua vektor dan biasanya digunakan untuk mengukur kesamaan antara dua teks/dokumen. Fungsi kesamaan kosinus antara titik A dan titik B direpresentasikan sebagai berikut:
Keterangan:
```
𝑠𝑖𝑚(𝐴, 𝐵) = nilai similaritas dari item A dan item B
𝑛(𝐴) = banyaknya fitur konten item A 
𝑛(𝐵) = banyaknya fitur konten item B 
𝑛(𝐴 ∩ 𝐵)  = banyaknya fitur konten yang terdapat pada item A dan juga terdapat pada item B
```
Jika kedua objek memiliki nilai similaritas 0, maka kedua objek dikatakan tidak identik dan apabila nilai similiaritas semakin mendekati 1 maka objek dapat dikatakan dekat kemiripannya

Langkah-langkah untuk proses ini adalah sebagai berikut.
1. Telusuri folder dari nama film yang ditentukan sebelumnya
2. Ambil nilai kemiripan dari semua film
3. Urutkan film berdasarkan kemiripan
4. Mengambil 5 judul berdasarkan kesamaan antara 1-6 karena urutan 0 memberikan indeks yang sama dengan judul film yang dimasukkan
5. Judul film diambil sesuai  dengan nilai kemiripan
6. Mengembalikan 5 judul film yang direkomendasikan dari hasil yang diurutkan berdasarkan kemiripan dan menampilkan genre dari 5 film yang direkomendasikan

Berikut _top_-5 _recommemdation_ berdasarkan genre dari judul film "*King Kong*"
judul | genre
---|---
Kingkong | Action, Adventure, Drama
![img](https://postimg.cc/Q9vmVcf5)
Dengan hasil yang diberikan di atas berdasarkan judul film "King Kong" dengan genre Action, Adventur, Drama maka didapatkan 5 rekomendasi judul film dengan genre yang memiliki kemiripan.

# Evaluation
---
Pada proyek ini, Metric yang digunakan pada sistem rekomendasi judul film dan serial berdasarkan genre adalah *precision*. *Precision* adalah metrik yang membandingkan rasio prediksi benar atau positif dengan keseluruhan hasil yang diprediksi positif sesuai dengan persamaan sebagai berikut :

$$\ Precission=TP/(TP+FP)$$
~~~
keterangan:
TP = True Positif (prediksi positif dan hal tersebut benar)
FP = False Positif (prediksi positif dan hal tersebut salah)
~~~
Presisi dipilih karena metrik ini dapat membandingkan rasio prediksi yang benar atau positif dengan hasil prediksi yang positif. Dalam hal ini, itu adalah rasio elemen yang menyarankan genre yang mirip atau mirip dengan genre film input.

~~~
# menghitung banyaknya data genre pada hasil rekomendasi 
value = pd.DataFrame(recomendation['genre'].value_counts().reset_index().values, columns = ['genre', 'count'])
value.head()
~~~
Output:
~~~

        genre   	                        count
0	    Action, Adventure, Drama            5
~~~
Dari output tersebut dihitung accuracy precision nya adalah
```
TP = 5 #jumlah prediksi benar untuk genre yang mirip atau serupa
FP = 0 #jumlah prediksi salah untuk genre yang mirip atau serupa

Precision = TP/(TP+FP)
print("{0:.0%}".format(Precision))
```
Dipilih nya nilai True Positif 5 karna ia merupakan nilai atau jumlah yg diduga memiliki kemiripan/identik dengan genre yg dipilih yaitu 5. hasil rekomendasi yg dihasilkan model menunjukan kemiripan dengan genre film yg dinput yaitu Action, Adventure, Drama
sedangkan utk nilai False Positif tidak teridentifikasi pada hasil output dari genre yg diinput maka nilai nya 0 
Output:
```
100%
```
Kesimpulan dari output yang dihasilkan bahwa prediksi rekomendasi yang diberikan memiliki nilai presisi 100%  sesuai genre yang mirip atau serupa dengan genre dari judul film atau serial yang diinput.
## Daftar Pustaka
---
[1]Jennings, G. (2019, September 2). 5 very real benefits of watching movies. Retrieved November 24, 2020
[2]Kaushik, A., Gupta, S., & Bhatia, M. (2018). A Movie Recommendation System using Neural Networks. International Journal of Advance Research, Ideas and Innovations in Technology, 425-430.
[3]Zhang, S., Yao, L., Sun, A., & Tay, Y. (2018). Deep Learning based Recommender System: A Survey and New Perspectives. ACM Computing Surveys, 1-35. 
---Ini adalah bagian akhir laporan---