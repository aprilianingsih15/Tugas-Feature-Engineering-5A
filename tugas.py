import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_excel("Data_Penjualan_Makanan_Minuman.xlsx")
print("=== Dataset Awal ===")
print(df.head())


# ============================================================
# 2. CLEANING DATA
# ============================================================
print("\n=== Missing Value Sebelum Cleaning ===")
print(df.isnull().sum())

# Isi missing value
df["Harga(Rp)"] = df["Harga(Rp)"].fillna(df["Harga(Rp)"].median())
df["Wilayah"] = df["Wilayah"].fillna("Tidak Diketahui")

print("\n=== Missing Value Setelah Cleaning ===")
print(df.isnull().sum())


# ============================================================
# 3. ONE-HOT ENCODING
# ============================================================
categorical_cols = ["Jenis_Produk", "Ukuran(Kemasan)", "Bulan", "Wilayah"]

encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded = encoder.fit_transform(df[categorical_cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_cols)
)

print("\n=== Hasil One-Hot Encoding ===")
print(encoded_df.head())


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# Fitur baru harga per pcs
df["Harga_per_Pcs"] = df["Harga(Rp)"] / df["Jumlah_Terjual"]

# Mengelompokkan bulan
bulan_map = {
    "Januari": "Awal Tahun", "Februari": "Awal Tahun",
    "Maret": "Awal Tahun", "April": "Awal Tahun",
    "Mei": "Pertengahan", "Juni": "Pertengahan",
    "Juli": "Pertengahan", "Agustus": "Pertengahan",
    "September": "Akhir Tahun", "Oktober": "Akhir Tahun",
    "November": "Akhir Tahun", "Desember": "Akhir Tahun"
}

df["Kategori_Bulan"] = df["Bulan"].map(bulan_map)

print("\n=== Fitur Baru ===")
print(df[["Bulan", "Kategori_Bulan", "Harga(Rp)", "Jumlah_Terjual", "Harga_per_Pcs"]].head())


# Encoding kategori bulan
encoder2 = OneHotEncoder(drop='first', sparse_output=False)
encoded_kat = encoder2.fit_transform(df[["Kategori_Bulan"]])

encoded_kat_df = pd.DataFrame(
    encoded_kat,
    columns=encoder2.get_feature_names_out(["Kategori_Bulan"])
)

print("\n=== Encoding Kategori Bulan ===")
print(encoded_kat_df.head())


# ============================================================
# 5. SCALING DATA NUMERIK
# ============================================================
numeric_cols = ["Harga(Rp)", "Jumlah_Terjual", "Harga_per_Pcs"]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

print("\n=== Setelah Scaling ===")
print(df_scaled[numeric_cols].head())


# ============================================================
# 6. GABUNGKAN SEMUA FITUR (FINAL DATASET)
# ============================================================
df_final = pd.concat(
    [
        df_scaled.drop(columns=categorical_cols + ["Kategori_Bulan"]),
        encoded_df,
        encoded_kat_df
    ],
    axis=1
)

print("\n=== Dataset Final (Setelah Cleaning, Encoding, Scaling, FE) ===")
print(df_final.head())

# Simpan output
df_final.to_csv("dataset_final_feature_engineering.csv", index=False)
print("\nDataset final berhasil disimpan sebagai 'dataset_final_feature_engineering.csv'")
