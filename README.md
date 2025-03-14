**Analisis dan Dokumentasi Hasil Eksperimen SkipGram dengan NewsAPI**

### **1. Pendahuluan**
Eksperimen ini bertujuan untuk melatih model SkipGram menggunakan data dari NewsAPI dengan variasi **window size** (1, 2, 3) dan **dimensi embedding** (20, 50, 100). Model diharapkan dapat menemukan hubungan antar kata dalam konteks berita dan mengevaluasi kedekatan kata berdasarkan embedding yang dihasilkan.

### **2. Proses Eksperimen**
- Data diperoleh melalui API dari NewsAPI dengan query *technology*.
- Data diproses dengan tokenisasi dan normalisasi.
- Model SkipGram dilatih dengan variasi **window size** dan **dimensi embedding**.
- Evaluasi dilakukan dengan melihat **loss** selama pelatihan dan **kedekatan kata** berdasarkan embedding yang dihasilkan.

### **3. Penjelasan Hasil dan Temuan dari parameter experiment**
#### **3.1 Perbandingan Loss Selama Pelatihan**
| Window Size | Embedding Dimensi | Epoch Sebelum Inf | Loss Awal | Loss Akhir Sebelum Inf |
|------------|------------------|-------------------|-----------|------------------------|
| 1          | 20               | 16                | 11.45     | 105.54                 |
| 1          | 50               | 15                | 16.66     | 129.31                 |
| 1          | 100              | 12                | 24.25     | 156.91                 |
| 2          | 20               | 10                | 12.02     | 135.74                 |
| 2          | 50               | 7                 | 18.68     | 98.21                  |
| 2          | 100              | 7                 | 28.68     | 149.84                 |
| 3          | 20               | 4                 | 12.39     | 56.93                  |
| 3          | 50               | 6                 | 21.14     | 138.53                 |
| 3          | 100              | 4                 | 33.08     | 117.22                 |

> *Catatan: Model mengalami **inf loss** setelah epoch tertentu

#### **3.2 Kedekatan Kata Berdasarkan Window Size dan Embedding Size**

##### **Window Size = 1**
| Embedding Dimensi | Kata        | Kata-Kata Serupa yang Ditemukan |
|------------------|------------|--------------------------------|
| 20               | samsungs   | with, werent, volvo, towards, one |
|                  | digital    | march, new, works, series, fired |
|                  | key        | via, and, xd, to, gpus |
|                  | technology | world, launching, planning, one, with |
|                  | now        | polestar, games, company, xd, crisis |
| 50               | samsungs   | pure, games, out, a, crisis |
|                  | digital    | pure, planning, a, new, out |
|                  | key        | polestar, a, nvidias, games, internet |
|                  | technology | recycle, march, planning, digital, trying |
|                  | now        | developing, in, directs, launching, lgs |
| 100              | samsungs   | speed, games, looks, evs, towards |
|                  | digital    | technology, out, just, hope, werent |
|                  | key        | launching, internet, directs, trying, lgs |
|                  | technology | digital, out, just, hope, werent |
|                  | now        | like, to, a, th, and |

##### **Window Size = 2**
| Embedding Dimensi | Kata        | Kata-Kata Serupa yang Ditemukan |
|------------------|------------|--------------------------------|
| 20               | samsungs   | via, nasa, just, key, light |
|                  | digital    | crisis, physx, gpus, planning, werent |
|                  | key        | light, speed, works, pure, now |
|                  | technology | its, like, way, out, internet |
|                  | now        | pure, works, speed, light, way |
| 50               | samsungs   | ryzen, th, march, polyester, one |
|                  | digital    | key, works, now, deliver, developing |
|                  | key        | works, now, deliver, developing, digital |
|                  | technology | nasa, the, physx, just, play |
|                  | now        | works, key, deliver, developing, digital |
| 100              | samsungs   | one, latest, crisis, xd, a |
|                  | digital    | recycle, physx, and, nvidias, way |
|                  | key        | way, polestar, and, trying, recycle |
|                  | technology | google, volvo, march, internet, nvidias |
|                  | now        | works, to, via, with, play |

##### **Window Size = 3**
| Embedding Dimensi | Kata        | Kata-Kata Serupa yang Ditemukan |
|------------------|------------|--------------------------------|
| 20               | samsungs   | key, launching, and, werent, planning |
|                  | digital    | physx, now, works, polestar, evs |
|                  | key        | samsungs, launching, werent, planning, and |
|                  | technology | and, samsungs, launching, key, deliver |
|                  | now        | polestar, physx, digital, works, evs |
| 50               | samsungs   | nvidias, it, and, th, looks |
|                  | digital    | key, like, new, pure, on |
|                  | key        | digital, like, new, pure, on |
|                  | technology | digital, key, like, new, pure |
|                  | now        | works, with, polestar, deliver, evs |
| 100              | samsungs   | digital, now, works, technology, hope |
|                  | digital    | samsungs, now, works, technology, hope |
|                  | key        | volvo, with, speed, hope, light |
|                  | technology | works, now, hope, speed, with |
|                  | now        | works, digital, technology, samsungs, hope |

- Model berhasil mengidentifikasi kata-kata yang memiliki konteks serupa.
- Namun, beberapa pasangan kata tampak tidak logis, kemungkinan besar karena **pelatihan yang tidak stabil akibat error numerik**.

### **4. Kesimpulan dan Perbaikan**
#### **4.1 Kesimpulan**
- Embedding tetap dapat menghasilkan kedekatan kata tetapi agak kurang relevan

Perbaikan yang Dapat Dilakukan
- **Evaluasi dataset** untuk memastikan keberagaman kata cukup besar agar model tidak mengalami bias terhadap kata tertentu.
- Melakukan Stopword Removal dan Steamming agar kata-kata yang tidak perlu bisa dieleminasi dan fokus ke bentuk kata dasar

