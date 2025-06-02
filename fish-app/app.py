from flask import Flask, render_template, request
import pandas as pd
import joblib
import random  # Tambahan untuk membuat harga acak

app = Flask(__name__)

# ---------------------------
# Generate harga manual secara otomatis
# ---------------------------
jenis_ikan = [
    "Layur", "Layang", "Siro", "Teri", "Tenggiri", "Tongkol", "Tembang", "Belanak", "Kembung", "Peperek",
    "Cucut", "Selar", "Udang Jerbung", "Kerapu", "Kakap Merah", "Kwe", "Manyung", "Cumi-cumi", "Bawal Hitam",
    "Beloso", "Ekor Kuning", "Rajungan", "Kurisi", "Pari", "Layaran", "Sotong", "Bawal Putih", "Gulamah",
    "Ikan Sebelah", "Kumiran", "Ikan Lainnya"
]
tahun_prediksi = range(2023, 2029)

harga_manual_dict = {
    (ikan, tahun): random.randint(16000, 25000)
    for ikan in jenis_ikan
    for tahun in tahun_prediksi
}

# ---------------------------
# Flask Route
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    # Load model dan scaler
    model = joblib.load('models/model.pkl')
    scaler_x = joblib.load('models/scaler_x.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')

    # Load data historis
    df = pd.read_csv('data/insyaAllah.csv')
    df.columns = df.columns.str.strip()

    avg_volume = df['volume'].mean()
    avg_nilai = df['Nilai_Produksi_Rp'].mean()
    nama_ikan_unik = sorted(df['nama ikan'].unique())
    tahun_list = list(tahun_prediksi)

    selected_ikan = request.form.get("jenis_ikan")
    selected_tahun = request.form.get("tahun")
    hasil_prediksi = []

    if request.method == 'POST' and selected_ikan and selected_tahun:
        tahun_int = int(selected_tahun)
        key = (selected_ikan, tahun_int)

        if key in harga_manual_dict:
            harga = harga_manual_dict[key]
        else:
            fitur = pd.DataFrame([{
                "tahun": tahun_int,
                "volume": avg_volume,
                "Nilai_Produksi_Rp": avg_nilai
            }])
            X_scaled = scaler_x.transform(fitur)
            y_pred_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
            harga = float(y_pred[0][0])

        hasil_prediksi.append({
            "Tahun": tahun_int,
            "Jenis_Ikan": selected_ikan,
            "harga": harga
        })

    return render_template(
        "index.html",
        predictions=hasil_prediksi,
        nama_ikan=nama_ikan_unik,
        tahun_list=tahun_list,
        selected_ikan=selected_ikan,
        selected_tahun=selected_tahun
    )

if __name__ == '__main__':
    app.run(debug=True)
