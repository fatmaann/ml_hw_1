import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Car Price Prediction", layout="wide")

@st.cache_resource
def load_artifacts():
    with open("car_price_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

@st.cache_data
def load_train_data():
    url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
    df = pd.read_csv(url)

    df["mileage"] = df["mileage"].str.replace(" kmpl", "").str.replace(" km/kg", "")
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
    df["engine"] = df["engine"].str.replace(" CC", "")
    df["engine"] = pd.to_numeric(df["engine"], errors="coerce")
    df["max_power"] = df["max_power"].str.replace(" bhp", "")
    df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")

    df = df.fillna(df.median(numeric_only=True))

    return df

try:
    artifacts = load_artifacts()
    model = artifacts["model"]
    ohe = artifacts["ohe"]
    scaler = artifacts["scaler"]
    num_cols = artifacts["num_cols"]
    cat_cols = artifacts["cat_cols"]
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("Файл car_price_model.pkl не найден! Запустите ноутбук для создания модели.")

df_raw = load_train_data()

st.title("Предсказание стоимости автомобилей")

st.markdown("""
Приложение для анализа данных и предсказания цен на автомобили:
- **EDA** — исследовательский анализ данных
- **Предсказание** — получение прогноза цены по CSV или ручному вводу
- **Веса модели** — визуализация коэффициентов Ridge-регрессии
""")


def full_preprocess(df: pd.DataFrame):
    df = df.copy()

    drop_cols = ["name", "torque", "selling_price"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    expected_cols = set(num_cols + cat_cols)
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")

    if df["mileage"].dtype == "object":
        df["mileage"] = df["mileage"].str.extract(r"(\d+\.?\d*)").astype(float)
    if df["engine"].dtype == "object":
        df["engine"] = df["engine"].str.extract(r"(\d+\.?\d*)").astype(float)
    if df["max_power"].dtype == "object":
        df["max_power"] = df["max_power"].str.extract(r"(\d+\.?\d*)").astype(float)

    df["seats"] = df["seats"].astype(int)

    X_num = df[num_cols].astype(float)

    X_cat = df[cat_cols].copy()
    for i, col in enumerate(cat_cols):
        trained_type = type(ohe.categories_[i][0])
        if np.issubdtype(trained_type, np.integer):
            X_cat[col] = X_cat[col].astype(int)
        else:
            X_cat[col] = X_cat[col].astype(str)

    X_cat_ohe = ohe.transform(X_cat)
    ohe_feature_names = ohe.get_feature_names_out(cat_cols)
    X_cat_ohe = pd.DataFrame(X_cat_ohe, index=df.index, columns=ohe_feature_names)

    X_final = pd.concat([X_num.reset_index(drop=True), X_cat_ohe.reset_index(drop=True)], axis=1)

    X_scaled = scaler.transform(X_final)

    return X_scaled, X_final.columns


def predict_df(df_features: pd.DataFrame) -> pd.Series:
    X_scaled, _ = full_preprocess(df_features)
    y_pred = model.predict(X_scaled)
    return pd.Series(y_pred, index=df_features.index)


def get_coefficients():
    ohe_feature_names = ohe.get_feature_names_out(cat_cols)
    feature_names = list(num_cols) + list(ohe_feature_names)
    coefs = model.coef_
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs)
    }).sort_values("abs_coef", ascending=False)
    return coef_df


tab_eda, tab_predict, tab_weights = st.tabs(["EDA", "Предсказание", "Веса модели"])

with tab_eda:
    st.header("Исследовательский анализ данных")

    st.subheader("Первые строки датасета")
    st.dataframe(df_raw.head(10))

    st.subheader("Основные статистики")
    st.dataframe(df_raw.describe())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Распределение цены (selling_price)**")
        fig, ax = plt.subplots()
        sns.histplot(df_raw["selling_price"], bins=30, ax=ax)
        ax.set_xlabel("Цена")
        st.pyplot(fig)

        st.markdown("**Цена по типу топлива**")
        fig, ax = plt.subplots()
        sns.boxplot(data=df_raw, x="fuel", y="selling_price", ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("**Распределение года выпуска**")
        fig, ax = plt.subplots()
        sns.histplot(df_raw["year"], bins=20, ax=ax)
        ax.set_xlabel("Год")
        st.pyplot(fig)

        st.markdown("**Цена vs Год (по типу КПП)**")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_raw, x="year", y="selling_price", hue="transmission", alpha=0.4, ax=ax)
        st.pyplot(fig)

    st.subheader("Корреляционная матрица")
    num_cols_eda = df_raw.select_dtypes(include=["int64", "float64"]).columns
    corr = df_raw[num_cols_eda].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

with tab_predict:
    st.header("Предсказание цены автомобиля")

    if not model_loaded:
        st.warning("Модель не загружена. Предсказания недоступны.")
    else:
        mode = st.radio("Выберите режим ввода:", ("Загрузить CSV", "Ручной ввод"))

        if mode == "Загрузить CSV":
            st.write("CSV должен содержать столбцы:", ", ".join(num_cols + cat_cols))

            uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")

            if uploaded_file is not None:
                try:
                    df_new = pd.read_csv(uploaded_file)
                    st.write("Загруженные данные:")
                    st.dataframe(df_new.head())

                    if df_new.isna().any().any():
                        st.warning("Обнаружены пропуски — строки с NaN будут удалены.")
                        df_new = df_new.dropna()

                    required_cols = set(num_cols + cat_cols)
                    missing = required_cols - set(df_new.columns)

                    if missing:
                        st.error(f"В CSV не хватает колонок: {missing}")
                    elif st.button("Сделать предсказания"):
                        preds = predict_df(df_new)
                        result = df_new.copy()
                        result["predicted_price"] = preds
                        st.success("Предсказания готовы!")
                        st.dataframe(result)

                        csv_out = result.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Скачать результаты CSV",
                            data=csv_out,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Ошибка: {e}")

        else:
            st.subheader("Ручной ввод признаков")

            col1, col2 = st.columns(2)

            with col1:
                year = st.number_input("Год выпуска", min_value=1983, max_value=2024, value=2015)
                km_driven = st.number_input("Пробег (км)", min_value=1, max_value=500000, value=60000)
                mileage = st.number_input("Расход (kmpl)", min_value=0.0, max_value=50.0, value=18.0)

            with col2:
                engine = st.number_input("Объем двигателя (CC)", min_value=600, max_value=4000, value=1200)
                max_power = st.number_input("Мощность (bhp)", min_value=0.0, max_value=500.0, value=80.0)
                seats = st.number_input("Количество мест", min_value=2, max_value=14, value=5)

            fuel = st.selectbox("Тип топлива", ["Petrol", "Diesel", "CNG", "LPG"])
            seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
            transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
            owner = st.selectbox("Владелец", [
                "First Owner", "Second Owner", "Third Owner",
                "Fourth & Above Owner", "Test Drive Car"
            ])

            if st.button("Предсказать цену"):
                df_one = pd.DataFrame([{
                    "year": year,
                    "km_driven": km_driven,
                    "mileage": mileage,
                    "engine": engine,
                    "max_power": max_power,
                    "seats": seats,
                    "fuel": fuel,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner
                }])

                pred = predict_df(df_one)[0]
                st.success(f"Предсказанная цена: **{pred:,.0f}** рупий")

with tab_weights:
    st.header("Веса модели Ridge Regression")

    if not model_loaded:
        st.warning("Модель не загружена.")
    else:
        coef_df = get_coefficients()

        st.subheader("Таблица коэффициентов")
        st.dataframe(coef_df)

        st.subheader("Визуализация важности признаков")
        top_n = st.slider("Количество признаков:", min_value=5, max_value=len(coef_df), value=15)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=coef_df.head(top_n), x="abs_coef", y="feature", hue="feature", ax=ax, palette="viridis", legend=False)
        ax.set_xlabel("|Коэффициент|")
        ax.set_ylabel("Признак")
        ax.set_title("Топ признаков по важности")
        st.pyplot(fig)

        st.markdown("""
        **Интерпретация:**
        - Чем больше абсолютное значение коэффициента, тем сильнее признак влияет на цену
        - Положительный коэффициент — увеличение признака повышает цену
        - Отрицательный коэффициент — увеличение признака снижает цену
        """)
