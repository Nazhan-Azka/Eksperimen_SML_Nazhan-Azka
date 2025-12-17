# automate_Nazhan-Azka.py

import pandas as pd
from pathlib import Path
from typing import Iterable, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari path relatif."""
    return pd.read_csv(file_path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Mengisi nilai kosong dengan median kolom numerik."""
    return df.fillna(df.median(numeric_only=True))


def encode_categorical(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Label encoding kolom kategorikal agar siap diproses model."""
    label_encoder = LabelEncoder()
    for col in columns:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col])
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Membuang outlier berbasis IQR pada kolom numerik."""
    df_cleaned = df.copy()
    for column in columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
    return df_cleaned


def normalize_data(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """StandardScaler untuk kolom numerik."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def split_features_labels(df: pd.DataFrame, label_column: str):
    """Pisahkan fitur dan label; buang kolom ID/Platform yang tidak dipakai model."""
    drop_cols = [col for col in ["User_ID", "Social_Media_Platform"] if col in df.columns]
    X = df.drop(columns=[label_column, *drop_cols])
    y = df[label_column]
    return X, y


def save_preprocessed_data(df: pd.DataFrame, output_path: str) -> None:
    """Simpan dataset siap latih ke lokasi output."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def preprocess_data(
    file_path: str,
    label_column: str = "Happiness_Index(1-10)",
    categorical_columns: Optional[Iterable[str]] = None,
    output_path: Optional[str] = None,
):
    """Pipeline preprocessing end-to-end: load → isi NaN → encode → buang outlier → normalisasi → split + simpan."""
    if categorical_columns is None:
        categorical_columns = ["Gender"]

    # 1. Memuat dataset
    df = load_data(file_path)

    # 2. Menangani nilai kosong
    df = handle_missing_values(df)

    # 3. Mengonversi kolom kategorikal menjadi numerik
    df = encode_categorical(df, categorical_columns)

    # 4. Menghapus outlier menggunakan IQR
    numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
    df = remove_outliers_iqr(df, numerical_columns)

    # 5. Normalisasi data numerik
    df = normalize_data(df, numerical_columns)

    # 6. Simpan dataset siap latih bila diminta
    if output_path:
        save_preprocessed_data(df, output_path)

    # 7. Memisahkan fitur dan label
    X, y = split_features_labels(df, label_column)

    return X, y


if __name__ == "__main__":
    # Jalankan otomatisasi preprocessing dari raw → siap latih
    raw_path = "../namadataset_raw/Mental_Health_and_Social_Media_Balance.csv"
    output_path = "namadataset_preprocessing/Mental_Health_and_Social_Media_Balance_No_Outlier.csv"

    X, y = preprocess_data(raw_path, output_path=output_path)
    print("Fitur dan label telah diproses dan siap digunakan untuk pelatihan.")
