import json
import random
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- 시계열 샘플 생성 함수 ---
def create_sequences(data, target_idx, seq_len, future_len):
    X, y = [], []
    for i in range(seq_len, len(data) - future_len):
        X.append(data[i - seq_len:i])
        y.append(data[i:i + future_len, target_idx])
    return np.array(X), np.array(y)

# --- 모델 생성 함수 ---
def create_gru_model(seq_len, num_features, units1, units2, dropout_rate, future_days):
    model = Sequential([
        GRU(units1, return_sequences=True, input_shape=(seq_len, num_features)),
        Dropout(dropout_rate),
        GRU(units2),
        Dropout(dropout_rate),
        Dense(future_days)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 하이퍼파라미터 탐색을 위한 유전 알고리즘 ---
def train_evaluate_gru(config, X_train, y_train, X_val, y_val, seq_len, num_features, future_days):
    model = create_gru_model(seq_len, num_features, config['gru_units1'], config['gru_units2'], config['dropout'], future_days)
    callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
    history = model.fit(X_train, y_train, epochs=10, batch_size=config['batch_size'], verbose=0,
                        validation_data=(X_val, y_val), callbacks=callbacks)
    loss = model.evaluate(X_val, y_val, verbose=0)
    return loss

def generate_random_individual(gru_units_range, dropout_range, batch_size_options):
    return {
        'gru_units1': random.choice(gru_units_range),
        'gru_units2': random.choice(gru_units_range),
        'dropout': random.uniform(dropout_range[0], dropout_range[1]),
        'batch_size': random.choice(batch_size_options)
    }

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(individual, gru_units_range, dropout_range, batch_size_options, mutation_rate=0.1):
    mutated_individual = individual.copy()
    if random.random() < mutation_rate:
        mutated_individual['gru_units1'] = random.choice(gru_units_range)
    if random.random() < mutation_rate:
        mutated_individual['gru_units2'] = random.choice(gru_units_range)
    if random.random() < mutation_rate:
        mutated_individual['dropout'] = random.uniform(dropout_range[0], dropout_range[1])
    if random.random() < mutation_rate:
        mutated_individual['batch_size'] = random.choice(batch_size_options)
    return mutated_individual

def genetic_algorithm_search(X_train, y_train, X_val, y_val, seq_len, num_features, future_days,
                            population_size=10, generations=3,
                            gru_units_range=[32, 64, 128], dropout_range=[0.1, 0.3], batch_size_options=[8, 16, 32]):
    population = [generate_random_individual(gru_units_range, dropout_range, batch_size_options)
                  for _ in range(population_size)]

    best_loss = float('inf')
    best_config = None

    for generation in range(generations):
        print(f"\n--- Generation {generation + 1} ---")
        evaluated_population = []
        for individual in population:
            loss = train_evaluate_gru(individual, X_train, y_train, X_val, y_val, seq_len, num_features, future_days)
            evaluated_population.append((individual, loss))
            print(f"Individual: {individual}, Loss: {loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                best_config = individual

        evaluated_population.sort(key=lambda item: item[1])
        best_individuals = [item[0] for item in evaluated_population[:population_size // 2]]

        new_population = best_individuals.copy()
        while len(new_population) < population_size:
            parent1 = random.choice(best_individuals)
            parent2 = random.choice(best_individuals)
            child = crossover(parent1, parent2)
            child = mutate(child, gru_units_range, dropout_range, batch_size_options)
            new_population.append(child)

        population = new_population

    print(f"\n--- Genetic Algorithm Search Finished ---")
    print(f"Best Configuration: {best_config}")
    print(f"Best Validation Loss: {best_loss:.4f}")
    return best_config

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- 데이터 준비 및 실행 (한 번만 실행) ---
if __name__ == "__main__":
    # 데이터 다운로드
    df = yf.download("GOOGL", period="3000d", auto_adjust=True)
    df.dropna(inplace=True)
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA14'] = df['Close'].rolling(14).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = compute_RSI(df['Close'])
    df.dropna(inplace=True)

    features = ['Close', 'MA7', 'MA14', 'MA50', 'RSI']
    target_idx = features.index('Close')
    seq_len, future_days = 60, 5
    num_features = len(features)

    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(df[features])
    X, y = create_sequences(scaled_full, target_idx, seq_len, future_days)

    tscv = TimeSeriesSplit(n_splits=5)
    train_index, val_index = next(tscv.split(X))
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    best_config = genetic_algorithm_search(X_train, y_train, X_val, y_val,
                                           seq_len, num_features, future_days)

    # 저장
    with open("best_gru_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print("✅ 저장 완료: best_gru_config.json")
