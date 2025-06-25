import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import IsolationForest
import random
import tensorflow as tf
from scipy.stats import ttest_rel
from scipy.stats import zscore
from prophet import Prophet
from curl_cffi import requests
from pyESN import ESN
from scipy import stats
import json
import matplotlib.pyplot as plt
import platform
import os

# yfinance 429 우회 세션 적용
session = requests.Session(impersonate="chrome")

# 한글 폰트 설정 (Windows 기준)
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'

# 저장된 폴드 결과 파일 이름
TSS_RESULT_FILE = "gru_lstm_tss_results1.npz"

# best_gru_config.json 파일이 존재하지 않을 경우 기본값 설정
try:
    with open("best_gru_config.json", "r") as f:
        best_gru_config = json.load(f)
    with open("best_lstm_config.json", "r") as f:
        best_lstm_config = json.load(f)
except FileNotFoundError:
    print("GA 파일을 찾을 수 없습니다. 기본 하이퍼파라미터를 사용합니다.")
    best_gru_config = {'gru_units1': 128, 'gru_units2': 64, 'dropout': 0.2, 'batch_size': 32}
    best_lstm_config = {'lstm_units1': 128, 'lstm_units2': 64, 'dropout': 0.2, 'batch_size': 32}

# 1. 데이터 로딩 및 기술적 지표 계산
df = yf.download("GOOGL", period="3000d", auto_adjust=True, session=session)

print(df)
# df.columns가 MultiIndex인 경우 처리
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
# 'Close' 컬럼이 Series가 아니라 DataFrame으로 로드되는 경우를 방지
if isinstance(df['Close'], pd.DataFrame):
    df['Close'] = df['Close'].squeeze()

# 이동평균
df['MA7'] = df['Close'].rolling(window=7).mean()
df['MA14'] = df['Close'].rolling(window=14).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()


# RSI (Relative Strength Index)
def calculate_rsi(data, period=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


df['RSI'] = calculate_rsi(df['Close'])


# MACD (Moving Average Convergence Divergence)
def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    short_ema = data.ewm(span=short_period, adjust=False).mean()
    long_ema = data.ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


df['MACD_Line'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])

# NaN 값 처리 (이상치 제거 전에 수행하는 것이 안전)
df.ffill(inplace=True)
df.bfill(inplace=True)
df.dropna(inplace=True)

# 이상치 제거 (추가된 부분)
Z_THRESHOLD = 3
GAP_THRESHOLD = 0.1
MA_WINDOW = 5
VOLUME_MEAN_THRESHOLD = 10000

if not df.empty:
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    if not df["log_return"].dropna().empty:
        z_scores = zscore(df["log_return"].dropna())
        z_score_series = pd.Series(z_scores, index=df["log_return"].dropna().index)
        df["z_log_return"] = z_score_series.reindex(df.index)
    else:
        df["z_log_return"] = np.nan
    df["ma_5"] = df["Close"].rolling(window=MA_WINDOW).mean()
    df["gap_ratio"] = (df["Close"] - df["ma_5"]) / df["ma_5"]
    df["true_event"] = (df["z_log_return"].abs() > Z_THRESHOLD) & (df["gap_ratio"].abs() > GAP_THRESHOLD)
    df["maybe_bad_data"] = (
            (df["Volume"] < 1) |
            (df["Close"] < 1) |
            ((df["Close"] == df["Open"]) & (df["Close"] == df["High"]) & (df["Close"] == df["Low"])) |
            (df["High"] < df["Low"]) |
            ((df["Close"].pct_change().abs() > 0.7) & (df["Volume"].rolling(3).mean() < 10000))
    )
    df["likely_real_event"] = (
            (df["Volume"].rolling(3).mean() > df["Volume"].rolling(30).mean() * 1.5) |
            ((df["z_log_return"].abs() > Z_THRESHOLD) & (df["gap_ratio"].abs() > GAP_THRESHOLD) & (
                    df["Volume"] > VOLUME_MEAN_THRESHOLD))
    )
    df["clean_target"] = df["true_event"] & df["maybe_bad_data"] & ~df["likely_real_event"]
    df["Close_clean"] = df["Close"]
    df.loc[df["clean_target"], "Close_clean"] = np.nan
    df["Close_clean"] = df["Close_clean"].fillna(
        df["Close_clean"].rolling(window=MA_WINDOW, min_periods=1, center=True).mean()
    )
    df["Anomaly"] = df["clean_target"].astype(int)
    df["Close"] = df["Close_clean"]
    df.drop(columns=["Close_clean"], inplace=True)
else:
    print("경고: 데이터프레임이 비어있어 이상치 제거를 건너뜁니다.")

# 결측 제거 (이상치 제거 후 발생할 수 있는 NaN 최종 제거)
df.ffill(inplace=True)
df.bfill(inplace=True)
df.dropna(inplace=True)

# 2. GC/DC 분석
df['GC'] = np.where((df['MA7'] > df['MA50']) & (df['MA7'].shift(1) <= df['MA50'].shift(1)), 1, 0)
df['DC'] = np.where((df['MA7'] < df['MA50']) & (df['MA7'].shift(1) >= df['MA50'].shift(1)), -1, 0)
df['GC/DC Signal'] = df['GC'] + df['DC']


# 3. 시계열 데이터 생성 함수(다중 Dense 출력)
def create_sequences(data, target_idx, seq_len, future_len):
    x, y = [], []
    if len(data) <= seq_len + future_len:
        print(f"경고: 데이터 길이가 시퀀스 생성에 충분하지 않습니다. 최소 {seq_len + future_len} 필요, 현재 {len(data)}.")
        return np.array([]), np.array([])
    for i in range(seq_len, len(data) - future_len):
        x.append(data[i - seq_len:i])
        y.append(data[i:i + future_len, target_idx])
    return np.array(x), np.array(y)


# ESN용 시계열 생성 함수 (원본 그대로 사용)
def create_ESN_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])  # ESN은 기본적으로 단일 스텝 예측을 목표로 하므로 y는 단일 값
    return np.array(x), np.array(y)


# 4. 정규화 ('Anomaly' 컬럼 포함)
features = ['Close', 'MA7', 'MA14', 'MA50', 'RSI', 'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'Anomaly']
num_features = len(features)
full_scaler = MinMaxScaler()
scaled_full = full_scaler.fit_transform(df[features])
close_scaler = MinMaxScaler()
scaled_close = close_scaler.fit_transform(df[['Close']])


# 5. 모델 정의
def create_gru_model(seq_len, num_features, units1, units2, dropout_rate, future_days):
    model = Sequential([
        GRU(units1, return_sequences=True, input_shape=(seq_len, num_features)),
        Dropout(dropout_rate),
        GRU(units2),
        Dropout(dropout_rate),
        Dense(future_days)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_lstm_model(seq_len, num_features, units1, units2, dropout_rate, future_days):
    model = Sequential([
        LSTM(units1, return_sequences=True, input_shape=(seq_len, num_features)),
        Dropout(dropout_rate),
        LSTM(units2),
        Dropout(dropout_rate),
        Dense(future_days)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ✅ ESN 실행 함수 (단기 예측 기준) - 파라미터 이름을 기존 main 코드에 맞게 조정
def run_esn(scaled_data_full, scaled_data_close, seq_len, split_ratio, scaler_close_obj):
    # ESN 입력 시퀀스 생성: X_esn은 다변량(전체 특성), y_esn은 단변량(종가)
    # y_esn의 target은 scaled_data_close에서 가져옴
    X_esn_raw, _ = create_ESN_sequences(scaled_data_full, seq_len)
    _, y_esn_raw = create_ESN_sequences(scaled_data_close, seq_len)  # ESN의 y는 다음 날 1개만 예측

    # ESN 입력에 맞게 X_esn_raw를 2D로 reshape (samples, seq_len * num_features)
    # scaled_data_full.shape[1]은 원래 num_features
    X_esn = X_esn_raw.reshape(X_esn_raw.shape[0], seq_len * scaled_data_full.shape[1])
    # y_esn_raw는 (samples, 1) 형태이므로 1D로 flatten
    y_esn = y_esn_raw.flatten()  # (samples,) 형태로 만듦

    # train/validation split
    split_index = int(len(X_esn) * split_ratio)
    X_esn_train, X_esn_val = X_esn[:split_index], X_esn[split_index:]
    y_esn_train, y_esn_val = y_esn[:split_index], y_esn[split_index:]

    # ESN 모델 초기화 및 학습
    # n_inputs는 X_esn.shape[1] (seq_len * num_features)
    # n_outputs는 1 (종가 단일 예측)
    esn = ESN(
        n_inputs=X_esn.shape[1],
        n_outputs=1,
        n_reservoir=200,
        spectral_radius=0.95,
        sparsity=0.2,
        noise=0.001,
        random_state=42
    )
    esn.fit(X_esn_train, y_esn_train)

    # 예측 수행
    pred_val_esn_scaled = esn.predict(X_esn_val)
    # 예측값을 [0, 1] 범위로 클리핑 (정규화된 값이므로)
    pred_val_esn_scaled = np.clip(pred_val_esn_scaled, 0, 1)

    # 예측값 역정규화
    # pred_val_esn_scaled는 (n_samples,) 형태이므로 reshape(-1, 1)
    pred_val_esn_close = scaler_close_obj.inverse_transform(pred_val_esn_scaled.reshape(-1, 1))

    # ESN은 단일 스텝 예측이므로, y_esn_val에 해당하는 실제값을 역정규화하여 반환
    real_val_esn_close = scaler_close_obj.inverse_transform(y_esn_val.reshape(-1, 1))

    return pred_val_esn_close, real_val_esn_close, X_esn_val  # X_esn_val도 반환하여 나중에 전체 예측에 사용


# 6. 데이터 분할 (Time Series Split 적용)
seq_len, future_days = 60, 5
target_idx = features.index('Close')

if df.empty:
    print("경고: 시퀀스를 생성할 데이터가 없어 모델 학습 및 예측을 건너뜁니다.")
    gru_predicted_series_avg = pd.Series()
    lstm_predicted_series_avg = pd.Series()
    prophet_predicted_series = pd.Series()
    pred_val_esn_close_for_plot = pd.Series()  # ESN 추가
    gru_true_prices_for_plot = pd.Series()
    gru_predicted_prices_for_plot = pd.Series()
    lstm_predicted_prices_for_plot = pd.Series()
else:
    X, y = create_sequences(scaled_full, target_idx, seq_len, future_days)

    # ESN을 위한 추가 변수
    # ESN은 TimeSeriesSplit 내부에서 돌리는 대신, 전체 데이터셋에 대해 한번 학습/예측 후,
    # 검증셋에 해당하는 부분을 잘라서 비교합니다.
    # 기존 run_esn_and_plot 함수가 split을 내부적으로 수행하므로,
    # 여기서는 전체 scaled_full을 전달하고 내부에서 split_ratio를 사용합니다.
    # split_ratio는 TimeSeriesSplit의 마지막 폴드 비율에 맞춰 조정합니다.
    # 여기서는 단순화를 위해 전체 데이터셋의 훈련/검증 분할을 한번만 수행합니다.

    # ESN의 학습/검증 비율 설정 (예: 훈련 80%, 검증 20%)
    # GRU/LSTM의 마지막 폴드의 test_idx 길이를 기준으로 ESN의 split_ratio를 결정할 수 있습니다.
    # 여기서는 간단히 전체 데이터의 80%를 훈련, 20%를 검증으로 설정하겠습니다.
    esn_split_ratio = 0.8
    pred_val_esn_close, real_val_esn_close, X_esn_val_original_shape = run_esn(
        scaled_full, scaled_close, seq_len, esn_split_ratio, close_scaler
    )
    if X.size == 0 or y.size == 0:
        print("경고: 생성된 시퀀스 데이터가 비어있어 모델 학습 및 예측을 건너뜁니다.")
        gru_predicted_series_avg = pd.Series()
        lstm_predicted_series_avg = pd.Series()
        prophet_predicted_series = pd.Series()
        pred_val_esn_close_for_plot = pd.Series()  # ESN 추가
        gru_true_prices_for_plot = pd.Series()
        gru_predicted_prices_for_plot = pd.Series()
        lstm_predicted_prices_for_plot = pd.Series()
    else:
        if os.path.exists(TSS_RESULT_FILE):
            print(f"[✔] 기존 TimeSeriesSplit 결과 파일이 존재함 → 불러옵니다: {TSS_RESULT_FILE}")
            data = np.load(TSS_RESULT_FILE)
            gru_preds_all = data['gru_preds_all']
            gru_trues_all = data['gru_trues_all']
            lstm_preds_all = data['lstm_preds_all']
            lstm_trues_all = data['lstm_trues_all']
        else:
            print(f"[❗] 결과 파일이 없음 → GRU/LSTM TimeSeriesSplit 학습 수행 중...")
            n_splits = 5
            tscv = TimeSeriesSplit(n_splits=n_splits)

            print("\n--- GRU & LSTM 모델 TSS 기반 학습 시작 ---")
            gru_preds_all = []
            gru_trues_all = []
            lstm_preds_all = []
            lstm_trues_all = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                print(f"\n--- Fold {fold + 1} ---")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # === GRU 학습 ===
                gru_model = create_gru_model(seq_len, num_features,
                                             best_gru_config['gru_units1'],
                                             best_gru_config['gru_units2'],
                                             best_gru_config['dropout'],
                                             future_days)
                gru_callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)]
                gru_model.fit(X_train, y_train, epochs=50, batch_size=best_gru_config['batch_size'],
                              validation_data=(X_test, y_test), callbacks=gru_callbacks, verbose=1)
                gru_pred = gru_model.predict(X_test)
                gru_preds_all.extend(gru_pred)
                gru_trues_all.extend(y_test)

                # === LSTM 학습 ===
                #lstm_model = create_lstm_model(seq_len, num_features, 128, 64, 0.2, future_days)
                lstm_model = create_lstm_model(seq_len, num_features,
                                               best_lstm_config['lstm_units1'],
                                               best_lstm_config['lstm_units2'],
                                               best_lstm_config['dropout'],
                                               future_days)
                lstm_callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)]
                lstm_model.fit(X_train, y_train, epochs=50, batch_size=32,
                               validation_data=(X_test, y_test), callbacks=lstm_callbacks, verbose=1)
                lstm_pred = lstm_model.predict(X_test)
                lstm_preds_all.extend(lstm_pred)
                lstm_trues_all.extend(y_test)

                # 학습 결과 저장
            np.savez(TSS_RESULT_FILE,
                    gru_preds_all=np.array(gru_preds_all),
                    gru_trues_all=np.array(gru_trues_all),
                    lstm_preds_all=np.array(lstm_preds_all),
                    lstm_trues_all=np.array(lstm_trues_all))
            print(f"[💾] K-Fold 예측 결과 저장 완료 → {TSS_RESULT_FILE}")

        # --- 결과 정리 및 역정규화 (TSS 모든 폴드의 예측값 취합) ---
        gru_predicted_common_tss = close_scaler.inverse_transform(np.array(gru_preds_all))
        actual_common_gru_tss = close_scaler.inverse_transform(np.array(gru_trues_all))

        lstm_predicted_common_tss = close_scaler.inverse_transform(np.array(lstm_preds_all))
        actual_common_lstm_tss = close_scaler.inverse_transform(np.array(lstm_trues_all))

        # --- Prophet 모델 추가 ---
        print("\n--- Prophet 모델 학습 시작 ---")
        prophet_df = df.reset_index()[['Date', 'Close', 'RSI', 'GC/DC Signal']].copy()
        prophet_df.columns = ['ds', 'y', 'rsi', 'signal']
        prophet_df['rsi'] = prophet_df['rsi'].astype(np.float64)
        prophet_df['signal'] = prophet_df['signal'].astype(np.float64)
        prophet_df['y'] = prophet_df['y'].astype(np.float64)
        prophet_df.dropna(subset=['ds', 'y', 'rsi', 'signal'], inplace=True)

        m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        m.add_regressor('rsi')
        m.add_regressor('signal')
        m.fit(prophet_df)

        future_prophet_df = pd.DataFrame({'ds': df.index})
        future_prophet_df = pd.merge(future_prophet_df, prophet_df[['ds', 'rsi', 'signal']], on='ds', how='left')
        future_prophet_df['rsi'].fillna(method='ffill', inplace=True)
        future_prophet_df['signal'].fillna(method='ffill', inplace=True)
        future_prophet_df['rsi'].fillna(method='bfill', inplace=True)
        future_prophet_df['signal'].fillna(method='bfill', inplace=True)

        prophet_forecast = m.predict(future_prophet_df)
        prophet_predicted_series = prophet_forecast.set_index('ds')['yhat']

        # --- GRU, LSTM 최종 모델 재학습 (전체 데이터 사용) ---
        print("\n--- GRU 최종 모델 (전체 데이터) 학습 시작 ---")
        gru_model_final = create_gru_model(seq_len, num_features,
                                           best_gru_config['gru_units1'],
                                           best_gru_config['gru_units2'],
                                           best_gru_config['dropout'],
                                           future_days)

        gru_model_final.fit(X, y, epochs=50, batch_size=best_gru_config['batch_size'], validation_split=0.1, shuffle=False, verbose=1)


        print("\n--- LSTM 최종 모델 (전체 데이터) 학습 시작 ---")
        lstm_model_final = create_lstm_model(seq_len, num_features, best_lstm_config['lstm_units1'],
                                             best_lstm_config['lstm_units2'], best_lstm_config['dropout'], future_days)

        lstm_model_final.fit(X, y, epochs=50, batch_size= best_lstm_config['batch_size'], validation_split=0.1, shuffle=False, verbose=1)

        # 9. 최근 100일 예측 비교 (GRU, LSTM, Prophet, ESN)
        latest_date = df.index[-1]
        start_date_plot = latest_date - timedelta(days=100)

        actual_daily_plot = df['Close'].loc[start_date_plot:].copy()

        gru_pred_for_plot = gru_model_final.predict(X)
        lstm_pred_for_plot = lstm_model_final.predict(X)
        dates_for_predictions = df.index[seq_len: seq_len + len(X)]

        gru_predicted_series_for_plot = pd.Series(close_scaler.inverse_transform(gru_pred_for_plot)[:, 0],
                                                  index=dates_for_predictions)
        lstm_predicted_series_for_plot = pd.Series(close_scaler.inverse_transform(lstm_pred_for_plot)[:, 0],
                                                   index=dates_for_predictions)

        # ESN의 예측 결과 준비: run_esn 함수에서 반환된 validation set 예측값
        # ESN 예측값은 이미 역정규화되어 있고, 실제값도 역정규화되어 있음
        # ESN 예측은 단일 스텝이므로, 해당 예측값의 인덱스를 맞추는 것이 중요합니다.
        # run_esn에서 반환된 X_esn_val_original_shape는 (samples, seq_len * num_features) 형태
        # 이 데이터의 원래 인덱스를 찾아서 ESN 예측값 Series를 만들어야 합니다.

        # df 인덱스와 X_esn_val_original_shape (scaled_full 기반) 인덱스 매핑
        # ESN의 검증셋 시작 인덱스
        esn_val_start_idx_in_original_df = int(len(scaled_full) * esn_split_ratio)
        esn_val_end_idx_in_original_df = len(scaled_full) - 1  # ESN의 y_esn은 마지막 값까지 예측

        # ESN 시퀀스 생성 함수는 len(data) - seq_length 까지 시퀀스를 생성합니다.
        # X_esn은 (len(data) - seq_length) 개의 시퀀스를 가집니다.
        # y_esn은 X_esn의 각 시퀀스 다음 날의 값입니다.
        # 따라서 ESN 예측값의 날짜는 df.index[seq_len + split_index : seq_len + len(X_esn)] 입니다.

        # ESN 예측값에 해당하는 날짜 인덱스 추출
        # X_esn의 전체 길이는 len(df) - seq_len
        # esn_val_split_start_idx는 X_esn 기준으로 split_index
        esn_full_len = len(df) - seq_len
        esn_val_split_start_idx = int(esn_full_len * esn_split_ratio)

        # ESN 예측값의 날짜는 실제 df 인덱스에서 seq_len을 더한 값에서 시작
        esn_pred_dates = df.index[seq_len + esn_val_split_start_idx: seq_len + esn_full_len]

        # ESN 예측값을 Series로 변환
        esn_predicted_series_for_plot = pd.Series(pred_val_esn_close.flatten(), index=esn_pred_dates)
        esn_true_series_for_plot = pd.Series(real_val_esn_close.flatten(), index=esn_pred_dates)

        gru_predicted_recent_plot = gru_predicted_series_for_plot.loc[start_date_plot:].groupby(level=0).mean()
        lstm_predicted_recent_plot = lstm_predicted_series_for_plot.loc[start_date_plot:].groupby(level=0).mean()
        prophet_predicted_recent_plot = prophet_predicted_series.loc[start_date_plot:].groupby(level=0).mean()
        esn_predicted_recent_plot = esn_predicted_series_for_plot.loc[start_date_plot:].groupby(level=0).mean()

        common_dates_plot_all = actual_daily_plot.index.intersection(gru_predicted_recent_plot.index).intersection(
            lstm_predicted_recent_plot.index).intersection(prophet_predicted_recent_plot.index).intersection(
            esn_predicted_recent_plot.index)

        actual_common_plot = actual_daily_plot.loc[common_dates_plot_all]
        gru_predicted_common_plot_for_eval = gru_predicted_recent_plot.loc[common_dates_plot_all]
        lstm_predicted_common_plot_for_eval = lstm_predicted_recent_plot.loc[common_dates_plot_all]
        prophet_predicted_common_plot_for_eval = prophet_predicted_recent_plot.loc[common_dates_plot_all]
        esn_predicted_common_plot_for_eval = esn_predicted_recent_plot.loc[common_dates_plot_all]

        # 평가 지표 계산 (최근 100일 예측)
        gru_mse = mean_squared_error(actual_common_plot, gru_predicted_common_plot_for_eval)
        gru_rmse = np.sqrt(gru_mse)
        gru_mae = mean_absolute_error(actual_common_plot, gru_predicted_common_plot_for_eval)
        gru_r2 = r2_score(actual_common_plot, gru_predicted_common_plot_for_eval)

        lstm_mse = mean_squared_error(actual_common_plot, lstm_predicted_common_plot_for_eval)
        lstm_rmse = np.sqrt(lstm_mse)
        lstm_mae = mean_absolute_error(actual_common_plot, lstm_predicted_common_plot_for_eval)
        lstm_r2 = r2_score(actual_common_plot, lstm_predicted_common_plot_for_eval)

        prophet_mse = mean_squared_error(actual_common_plot, prophet_predicted_common_plot_for_eval)
        prophet_rmse = np.sqrt(prophet_mse)
        prophet_mae = mean_absolute_error(actual_common_plot, prophet_predicted_common_plot_for_eval)
        prophet_r2 = r2_score(actual_common_plot, prophet_predicted_common_plot_for_eval)

        esn_mse = mean_squared_error(actual_common_plot, esn_predicted_common_plot_for_eval)
        esn_rmse = np.sqrt(esn_mse)
        esn_mae = mean_absolute_error(actual_common_plot, esn_predicted_common_plot_for_eval)
        esn_r2 = r2_score(actual_common_plot, esn_predicted_common_plot_for_eval)

        print(f"\n--- 모델 예측 성능 (최근 100일) ---")
        print(f"GRU (GA Tuned) - MSE: {gru_mse:.4f}, RMSE: {gru_rmse:.4f}, MAE: {gru_mae:.4f}, R2: {gru_r2:.4f}")
        print(f"LSTM - MSE: {lstm_mse:.4f}, RMSE: {lstm_rmse:.4f}, MAE: {lstm_mae:.4f}, R2: {lstm_r2:.4f}")
        print(
            f"Prophet - MSE: {prophet_mse:.4f}, RMSE: {prophet_rmse:.4f}, MAE: {prophet_mae:.4f}, R2: {prophet_r2:.4f}")
        print(f"ESN - MSE: {esn_mse:.4f}, RMSE: {esn_rmse:.4f}, MAE: {esn_mae:.4f}, R2: {esn_r2:.4f}")

        # 시각화 (GRU & LSTM & Prophet & ESN 모델 비교 - 최근 100일)
        plt.figure(figsize=(14, 8))
        plt.plot(actual_common_plot.index, actual_common_plot.values, label="📈 실제 종가", color='black',
                 linewidth=2)
        plt.plot(gru_predicted_common_plot_for_eval.index, gru_predicted_common_plot_for_eval.values,
                 label=f"🧠 GRU 예측 (GA Tuned)", color='blue', linestyle='--', linewidth=2)
        plt.plot(lstm_predicted_common_plot_for_eval.index, lstm_predicted_common_plot_for_eval.values,
                 label="🔮 LSTM 예측", color='red', linestyle='--', linewidth=2)
        plt.plot(prophet_predicted_common_plot_for_eval.index, prophet_predicted_common_plot_for_eval.values,
                 label="📊 Prophet 예측", color='green', linestyle='--', linewidth=2)
        plt.plot(esn_predicted_common_plot_for_eval.index, esn_predicted_common_plot_for_eval.values,
                 label="🔄 ESN 예측", color='purple', linestyle='--', linewidth=2)
        plt.title("GOOGL - 최근 100일 예측 비교: GRU (GA Tuned) vs LSTM vs Prophet vs ESN", fontsize=14)
        plt.xlabel("날짜")
        plt.ylabel("가격 (USD)")
        plt.legend()
        plt.grid(True)
        plt.text(0.01, 0.90, f'GRU (GA) MSE: {gru_mse:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.87, f'GRU (GA) RMSE: {gru_rmse:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.84, f'GRU (GA) MAE: {gru_mae:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.81, f'GRU (GA) R2: {gru_r2:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.76, f'LSTM MSE: {lstm_mse:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.73, f'LSTM RMSE: {lstm_rmse:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.70, f'LSTM MAE: {lstm_mae:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.67, f'LSTM R2: {lstm_r2:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.62, f'Prophet MSE: {prophet_mse:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.59, f'Prophet RMSE: {prophet_rmse:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.56, f'Prophet MAE: {prophet_mae:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.53, f'Prophet R2: {prophet_r2:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.48, f'ESN MSE: {esn_mse:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.45, f'ESN RMSE: {esn_rmse:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.42, f'ESN MAE: {esn_mae:.4f}', transform=plt.gca().transAxes)
        plt.text(0.01, 0.39, f'ESN R2: {esn_r2:.4f}', transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.axis('off')

        col_labels = ["MSE", "RMSE", "MAE", "R²"]
        row_labels = ["GRU (GA)", "LSTM", "Prophet", "ESN"]
        cell_text = [
            [gru_mse, gru_rmse, gru_mae, gru_r2],
            [lstm_mse, lstm_rmse, lstm_mae, lstm_r2],
            [prophet_mse, prophet_rmse, prophet_mae, prophet_r2],
            [esn_mse, esn_rmse, esn_mae, esn_r2],
        ]

        table = ax.table(cellText=cell_text,
                         rowLabels=row_labels,
                         colLabels=col_labels,
                         cellLoc='center',
                         loc='center')

        table.scale(1.2, 1.8)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.auto_set_column_width(col=list(range(len(col_labels))))

        plt.title("최근 100일 예측 성능 비교", fontsize=13, weight='bold', pad=12)
        plt.tight_layout()
        plt.show()

        # 예측 오차 계산 (최근 100일 기준, 이미 교차된 날짜 기준)
        if not actual_common_plot.empty:
            gru_errors = np.abs(actual_common_plot.values - gru_predicted_common_plot_for_eval.values)
            lstm_errors = np.abs(actual_common_plot.values - lstm_predicted_common_plot_for_eval.values)
            prophet_errors = np.abs(actual_common_plot.values - prophet_predicted_common_plot_for_eval.values)
            esn_errors = np.abs(actual_common_plot.values - esn_predicted_common_plot_for_eval.values)

            # GRU vs LSTM
            t_stat_gru_lstm, p_value_gru_lstm = ttest_rel(gru_errors, lstm_errors)
            print(f"\n📊 통계 검정 결과 (GRU vs LSTM 예측 오차):")
            print(f"t-statistic: {t_stat_gru_lstm:.4f}")
            print(f"p-value: {p_value_gru_lstm:.4f}")
            if p_value_gru_lstm < 0.05:
                print("✅ 귀무가설 기각: GRU와 LSTM 간 예측 성능 차이는 통계적으로 유의합니다.")
            else:
                print("⚠️ 귀무가설 채택: GRU와 LSTM 간 예측 성능 차이는 유의하지 않습니다.")

            # GRU vs Prophet
            t_stat_gru_prophet, p_value_gru_prophet = ttest_rel(gru_errors, prophet_errors)
            print(f"\n📊 통계 검정 결과 (GRU vs Prophet 예측 오차):")
            print(f"t-statistic: {t_stat_gru_prophet:.4f}")
            print(f"p-value: {p_value_gru_prophet:.4f}")
            if p_value_gru_prophet < 0.05:
                print("✅ 귀무가설 기각: GRU와 Prophet 간 예측 성능 차이는 통계적으로 유의합니다.")
            else:
                print("⚠️ 귀무가설 채택: GRU와 Prophet 간 예측 성능 차이는 유의하지 않습니다.")

            # GRU vs ESN
            t_stat_gru_esn, p_value_gru_esn = ttest_rel(gru_errors, esn_errors)
            print(f"\n📊 통계 검정 결과 (GRU vs ESN 예측 오차):")
            print(f"t-statistic: {t_stat_gru_esn:.4f}")
            print(f"p-value: {p_value_gru_esn:.4f}")
            if p_value_gru_esn < 0.05:
                print("✅ 귀무가설 기각: GRU와 ESN 간 예측 성능 차이는 통계적으로 유의합니다.")
            else:
                print("⚠️ 귀무가설 채택: GRU와 ESN 간 예측 성능 차이는 유의하지 않습니다.")

            # LSTM vs Prophet
            t_stat_lstm_prophet, p_value_lstm_prophet = ttest_rel(lstm_errors, prophet_errors)
            print(f"\n📊 통계 검정 결과 (LSTM vs Prophet 예측 오차):")
            print(f"t-statistic: {t_stat_lstm_prophet:.4f}")
            print(f"p-value: {p_value_lstm_prophet:.4f}")
            if p_value_lstm_prophet < 0.05:
                print("✅ 귀무가설 기각: LSTM과 Prophet 간 예측 성능 차이는 통계적으로 유의합니다.")
            else:
                print("⚠️ 귀무가설 채택: LSTM과 Prophet 간 예측 성능 차이는 유의하지 않습니다.")

            # LSTM vs ESN
            t_stat_lstm_esn, p_value_lstm_esn = ttest_rel(lstm_errors, esn_errors)
            print(f"\n📊 통계 검정 결과 (LSTM vs ESN 예측 오차):")
            print(f"t-statistic: {t_stat_lstm_esn:.4f}")
            print(f"p-value: {p_value_lstm_esn:.4f}")
            if p_value_lstm_esn < 0.05:
                print("✅ 귀무가설 기각: LSTM과 ESN 간 예측 성능 차이는 통계적으로 유의합니다.")
            else:
                print("⚠️ 귀무가설 채택: LSTM과 ESN 간 예측 성능 차이는 유의하지 않습니다.")

            # Prophet vs ESN
            t_stat_prophet_esn, p_value_prophet_esn = ttest_rel(prophet_errors, esn_errors)
            print(f"\n📊 통계 검정 결과 (Prophet vs ESN 예측 오차):")
            print(f"t-statistic: {t_stat_prophet_esn:.4f}")
            print(f"p-value: {p_value_prophet_esn:.4f}")
            if p_value_prophet_esn < 0.05:
                print("✅ 귀무가설 기각: Prophet과 ESN 간 예측 성능 차이는 통계적으로 유의합니다.")
            else:
                print("⚠️ 귀무가설 채택: Prophet과 ESN 간 예측 성능 차이는 유의하지 않습니다.")
        else:
            print("경고: 예측 오차 통계 검정을 위한 데이터가 부족합니다.")

        # 10. 전체 예측 시각화 (GRU & LSTM & Prophet & ESN 모델 비교)
        gru_pred_full_all_data = gru_model_final.predict(X)
        gru_predicted_full_inv = close_scaler.inverse_transform(gru_pred_full_all_data)[:, -1]

        lstm_pred_full_all_data = lstm_model_final.predict(X)
        lstm_predicted_full_inv = close_scaler.inverse_transform(lstm_pred_full_all_data)[:, -1]

        actual_full_inv = close_scaler.inverse_transform(y)[:, -1]

        dates_full_plot = df.index[seq_len: seq_len + len(X)]

        prophet_predicted_full_plot = prophet_predicted_series.loc[dates_full_plot].values

        # ESN 전체 예측 (X_esn을 전체 데이터에 대해 예측)
        # ESN은 단일 스텝 예측이므로, 마지막 'y_esn'을 전체 예측으로 사용합니다.
        # run_esn 함수에서 실제 'X_esn_val'에 대한 예측을 반환하므로,
        # 전체 데이터에 대한 ESN 예측을 위해서는 `run_esn` 내부에서 `esn.predict(X_esn)` (전체 X_esn)을 사용해야 합니다.
        # 여기서는 간단히 `pred_val_esn_close` (검증셋 예측값)을 전체 예측의 일부로 시각화합니다.
        # 하지만, 전체 데이터에 대한 예측을 하려면 ESN 모델을 전체 데이터에 대해 다시 학습시키고 예측해야 합니다.
        # 기존 `run_esn` 함수는 훈련/검증 분할을 하기 때문에, 전체 예측을 위해서는 ESN 모델을 다시 만들고 `esn.fit(X_esn, y_esn)` 후 `esn.predict(X_esn)`을 해야 합니다.
        # 편의상, 여기서는 `run_esn`에서 나온 `pred_val_esn_close`와 `esn_true_series_for_plot`을 사용합니다.
        # 즉, 전체 기간이 아닌, ESN의 검증셋 기간에 대한 예측값만 시각화됩니다.

        # 전체 기간에 대한 ESN 예측을 얻기 위한 추가 코드 (옵션)
        # ESN 모델을 전체 데이터에 다시 학습 (전체 기간 예측을 위함)
        print("\n--- ESN 최종 모델 (전체 데이터) 학습 시작 ---")
        # X_esn_full_data_raw는 모든 특성 (scaled_full)을 사용
        X_esn_full_data_raw_for_input, _ = create_ESN_sequences(scaled_full, seq_len)
        X_esn_full_data = X_esn_full_data_raw_for_input.reshape(
            X_esn_full_data_raw_for_input.shape[0], seq_len * scaled_full.shape[1]
        )

        # y_esn_full_data_raw는 종가 (scaled_close)만 사용
        _, y_esn_full_data_raw_for_target = create_ESN_sequences(scaled_close, seq_len)
        # y_esn_full_data는 1차원이어야 합니다.
        y_esn_full_data = y_esn_full_data_raw_for_target.flatten()  # (num_sequences,)

        esn_full_model = ESN(
            n_inputs=X_esn_full_data.shape[1],
            n_outputs=1,  # 종가 1개 예측
            n_reservoir=200,
            spectral_radius=0.95,
            sparsity=0.2,
            noise=0.001,
            random_state=42
        )
        esn_full_model.fit(X_esn_full_data, y_esn_full_data)
        esn_predicted_full_scaled = esn_full_model.predict(X_esn_full_data)
        esn_predicted_full_scaled = np.clip(esn_predicted_full_scaled, 0, 1)
        esn_predicted_full_inv = close_scaler.inverse_transform(esn_predicted_full_scaled.reshape(-1, 1)).flatten()

        # ESN 전체 예측의 날짜 인덱스
        esn_dates_full_plot = df.index[seq_len: seq_len + len(esn_predicted_full_inv)]

        plt.figure(figsize=(14, 6))
        plt.plot(dates_full_plot, actual_full_inv, label="📈 실제 종가", color='black')
        plt.plot(dates_full_plot, gru_predicted_full_inv, label=f"🧠 GRU 예측 (GA Tuned)", color='blue',
                 linestyle='--')
        plt.plot(dates_full_plot, lstm_predicted_full_inv, label="🔮 LSTM 예측", color='red', linestyle='--')
        plt.plot(dates_full_plot, prophet_predicted_full_plot, label="📊 Prophet 예측", color='green', linestyle='--')
        plt.plot(esn_dates_full_plot, esn_predicted_full_inv, label="🔄 ESN 예측", color='purple', linestyle='--')
        plt.title("GOOGL - 전체 기간 예측 비교 (~3000일): GRU / LSTM / Prophet / ESN")
        plt.xlabel("날짜")
        plt.ylabel("가격 (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 11. GC/DC 시점 시각화
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df['Close'], label="Close", color='black')
        plt.scatter(df.index[df['GC'] == 1], df['Close'][df['GC'] == 1], marker='^', color='green',
                    label="Golden Cross")
        plt.scatter(df.index[df['DC'] == -1], df['Close'][df['DC'] == -1], marker='v', color='red', label="Dead Cross")
        plt.title("GOOGL - Golden/Dead Cross Points")
        plt.xlabel("날짜");
        plt.ylabel("가격")
        plt.legend();
        plt.grid(True);
        plt.tight_layout()
        plt.show()

        # 12. GC/DC Signal 시각화
        plt.figure(figsize=(14, 4))
        plt.plot(df.index, df['GC/DC Signal'], color='purple')
        plt.title("GC/DC Signal Time Series (1 = GC, -1 = DC)")
        plt.xlabel("날짜");
        plt.ylabel("신호")
        plt.yticks([-1, 0, 1], ['Dead Cross', 'None', 'Golden Cross'])
        plt.grid(True);
        plt.tight_layout()
        plt.show()

        # 13. 이동평균선 시각화
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df['Close'], label='Close', color='black')
        plt.plot(df.index, df['MA7'], label='MA7', color='blue', linestyle='--')
        plt.plot(df.index, df['MA14'], label='MA14', color='orange', linestyle='--')
        plt.plot(df.index, df['MA50'], label='MA50', color='green', linestyle='--')
        plt.title("GOOGL - 이동평균선과 가격 추이")
        plt.xlabel("날짜");
        plt.ylabel("가격")
        plt.legend();
        plt.grid(True);
        plt.tight_layout()
        plt.show()


        # --- MDD, Sharpe Ratio 계산 함수들 (ESN 독립적) ---
        def calculate_mdd(equity_curve):
            if equity_curve.empty or len(equity_curve) < 2:
                return 0.0
            peak = equity_curve.cummax()
            drawdown = (equity_curve - peak) / peak
            return drawdown.min() * -1


        def calculate_sharpe_ratio(returns_series, risk_free_rate=0.02 / 252):
            if returns_series.empty or returns_series.std() == 0:
                return 0.0
            daily_returns = returns_series.dropna()
            if daily_returns.std() == 0:
                return 0.0
            excess_returns = daily_returns - risk_free_rate
            return excess_returns.mean() / daily_returns.std() * np.sqrt(252)


        # 14. GC/DC 전략 수익률 분석 및 성능지표 평가 (원본 GC/DC)
        df_signals = df[df['GC/DC Signal'] != 0].copy()

        strategy_daily_returns_original_gc_dc = pd.Series(0.0, index=df.index)
        position = 0

        for i in range(1, len(df)):
            date = df.index[i]
            prev_date = df.index[i - 1]

            current_close_price = df['Close'].iloc[i]
            prev_close_price = df['Close'].iloc[i - 1]

            gc_signal = df['GC'].iloc[i]
            dc_signal = df['DC'].iloc[i]

            if position == 1:
                strategy_daily_returns_original_gc_dc.loc[date] = (
                                                                          current_close_price - prev_close_price) / prev_close_price
            else:
                strategy_daily_returns_original_gc_dc.loc[date] = 0.0

            if gc_signal == 1 and position == 0:
                position = 1
            elif dc_signal == -1 and position == 1:
                position = 0

        cumulative_return_original_gc_dc = (1 + strategy_daily_returns_original_gc_dc).cumprod()
        if cumulative_return_original_gc_dc.empty:
            cumulative_return_original_gc_dc = pd.Series([1.0], index=[df.index[0]])

        sharpe_original_gc_dc = calculate_sharpe_ratio(strategy_daily_returns_original_gc_dc)
        mdd_original_gc_dc = calculate_mdd(cumulative_return_original_gc_dc)

        print(f"\n GC/DC 전략 평가 (MA7/MA50):")
        print(f"  누적 수익률: {cumulative_return_original_gc_dc.iloc[-1] * 100:.2f}%")
        print(f"  샤프지수 (연율화): {sharpe_original_gc_dc:.4f}")
        print(f"  최대 낙폭 (MDD): {mdd_original_gc_dc * 100:.2f}%")

        # --- Buy and Hold 전략 평가 ---
        buy_and_hold_daily_returns = df['Close'].pct_change().dropna()
        buy_and_hold_cum_return = (1 + buy_and_hold_daily_returns).cumprod()
        if buy_and_hold_cum_return.empty:
            buy_and_hold_cum_return = pd.Series([1.0], index=[df.index[0]])

        buy_and_hold_sharpe = calculate_sharpe_ratio(buy_and_hold_daily_returns)
        buy_and_hold_mdd = calculate_mdd(buy_and_hold_cum_return)

        print(f"\n Buy and Hold 전략 평가:")
        print(f"  누적 수익률: {buy_and_hold_cum_return.iloc[-1] * 100:.2f}%")
        print(f"  샤프지수 (연율화): {buy_and_hold_sharpe:.4f}")
        print(f"  최대 낙폭 (MDD): {buy_and_hold_mdd * 100:.2f}%")

        # --- 누적 수익률 비교 그래프 (원본 GC/DC vs Buy & Hold) ---
        plt.figure(figsize=(14, 6))
        plt.plot(cumulative_return_original_gc_dc.index, cumulative_return_original_gc_dc.values,
                 label="GC/DC 전략 누적 수익률", color='green')
        plt.plot(buy_and_hold_cum_return.index, buy_and_hold_cum_return.values, label="매수 후 보유 전략 누적 수익률",
                 color='blue', linestyle='--')
        plt.title("누적 수익률 비교 (GC/DC vs 매수 후 보유)", fontsize=14)
        plt.xlabel("날짜")
        plt.ylabel("누적 수익률 (배율)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- 최근 100일 예측 정확도 비교 (구간별 MSE 추출) ---
        if gru_predicted_common_tss.size > 0 and lstm_predicted_common_tss.size > 0:
            min_len_tss = min(len(actual_common_gru_tss), len(gru_predicted_common_tss), len(lstm_predicted_common_tss))

            chunk_size = min_len_tss // 5
            gru_mse_arr, lstm_mse_arr = [], []

            for i in range(5):
                start = i * chunk_size
                end = (i + 1) * chunk_size if (i + 1) * chunk_size <= min_len_tss else min_len_tss

                if start < end:
                    mse_gru = mean_squared_error(actual_common_gru_tss[start:end, 0],
                                                 gru_predicted_common_tss[start:end, 0])
                    mse_lstm = mean_squared_error(actual_common_lstm_tss[start:end, 0],
                                                  lstm_predicted_common_tss[start:end, 0])
                    gru_mse_arr.append(mse_gru)
                    lstm_mse_arr.append(mse_lstm)

            if gru_mse_arr and lstm_mse_arr:
                gru_mse_arr = np.array(gru_mse_arr)
                lstm_mse_arr = np.array(lstm_mse_arr)

                t_stat_mse, p_value_mse = stats.ttest_ind(gru_mse_arr, lstm_mse_arr)
                print(f"\n📊 t-검정 통계량 (MSE 구간 - TSS 결과): {t_stat_mse:.4f}, p-value: {p_value_mse:.5f}")
                if p_value_mse < 0.05:
                    print("✅ 대립가설 채택: GRU와 LSTM 간 MSE 성능 차이가 유의미함.")
                else:
                    print("✅ 귀무가설 채택: GRU와 LSTM 간 MSE 성능 차이가 통계적으로 유의미하지 않음.")
            else:
                print("경고: MSE 구간 분석을 위한 데이터가 부족합니다 (TSS 결과).")
        else:
            print("경고: MSE 구간 분석을 위한 데이터가 충분하지 않습니다 (TSS 결과).")

        # 최근 구간 평가 (Prophet 포함)
        if 'Date' not in df.columns:
            df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df_eval = df.copy()
        df.set_index("Date", inplace=True)

        pred_results = {
            "GRU": gru_predicted_series_for_plot,
            "LSTM": lstm_predicted_series_for_plot,
            "Prophet": prophet_predicted_series,
            "ESN": esn_predicted_series_for_plot  # ESN 추가
        }

        eval_periods = {
            "1Y": pd.DateOffset(years=1),
            "6M": pd.DateOffset(months=6),
            "3M": pd.DateOffset(months=3),
            "1M": pd.DateOffset(months=1)
        }


        def compute_return_and_mdd(series: pd.Series) -> tuple:
            if series.empty or len(series) < 2:
                return 0.0, 0.0
            if series.iloc[0] == 0:
                return 0.0, 0.0
            cum_return = series / series.iloc[0]
            mdd = (cum_return / cum_return.cummax() - 1).min()
            total_return = cum_return.iloc[-1] - 1
            return total_return, mdd


        eval_results = []
        for label, offset in eval_periods.items():
            start_date_eval = df_eval["Date"].max() - offset
            test_dates_in_period = df.index[df.index >= start_date_eval]

            for model_name, pred_series in pred_results.items():
                pred_filtered = pred_series.reindex(test_dates_in_period).dropna()
                actual_filtered = df["Close"].reindex(pred_filtered.index).dropna()
                common_idx = actual_filtered.index.intersection(pred_filtered.index)
                actual_vals = actual_filtered.loc[common_idx]
                pred_vals = pred_filtered.loc[common_idx]

                if len(common_idx) < 2 or actual_vals.empty or pred_vals.empty:
                    print(f"경고: {label} 기간, {model_name} 모델에 대한 평가 데이터가 부족합니다.")
                    continue

                mse = mean_squared_error(actual_vals, pred_vals)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual_vals, pred_vals)
                r2 = r2_score(actual_vals, pred_vals)
                ret, mdd = compute_return_and_mdd(pred_vals)

                eval_results.append({
                    "Period": label,
                    "Model": model_name,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2,
                    "Return": ret,
                    "MDD": mdd
                })

if eval_results:
    eval_df = pd.DataFrame(eval_results)
    import seaborn as sns

    sns.set(style="whitegrid")
    # 한글 폰트 설정 (Windows 기준)
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    metrics = ["R2", "Return", "MDD"]
    for metric in metrics:
        if not eval_df.empty:
            plt.figure(figsize=(10, 5))
            sns.barplot(data=eval_df, x="Period", y=metric, hue="Model")
            plt.title(f"{metric} (기간 및 모델별)", fontsize=14)
            plt.ylabel(metric)
            plt.xlabel("평가 기간")
            plt.legend(title="모델")
            plt.tight_layout()
            plt.show()
        else:
            print(f"경고: {metric} 시각화를 위한 평가 데이터가 없습니다.")
else:
    print("경고: 모든 평가 기간 및 모델에 대한 결과가 부족하여 성능 시각화를 건너뜁니다.")
