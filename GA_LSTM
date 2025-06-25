import json
import random
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


#####################
def create_lstm_model(seq_len, num_features, units1, units2, dropout_rate, future_days):
    model = Sequential([
        LSTM(units1, return_sequences=True, input_shape=(seq_len, num_features)),
        Dropout(dropout_rate),
        LSTM(units2),
        Dropout(dropout_rate),
        Dense(future_days)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_evaluate_lstm(config, X_train, y_train, X_val, y_val, seq_len, num_features, future_days):
    model = create_lstm_model(seq_len, num_features,
                              config['lstm_units1'], config['lstm_units2'],
                              config['dropout'], future_days)
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
    model.fit(X_train, y_train, epochs=10,
              batch_size=config['batch_size'],
              validation_data=(X_val, y_val),
              verbose=0, callbacks=callbacks)
    loss = model.evaluate(X_val, y_val, verbose=1)
    return loss


####################
def generate_random_individual(units_range, dropout_range, batch_size_options):
    return {
        'lstm_units1': random.choice(units_range),
        'lstm_units2': random.choice(units_range),
        'dropout': random.uniform(*dropout_range),
        'batch_size': random.choice(batch_size_options)
    }


def crossover(parent1, parent2):
    return {
        key: parent1[key] if random.random() < 0.5 else parent2[key]
        for key in parent1
    }


def mutate(individual, units_range, dropout_range, batch_size_options, mutation_rate=0.1):
    mutant = individual.copy()
    if random.random() < mutation_rate:
        mutant['lstm_units1'] = random.choice(units_range)
    if random.random() < mutation_rate:
        mutant['lstm_units2'] = random.choice(units_range)
    if random.random() < mutation_rate:
        mutant['dropout'] = random.uniform(*dropout_range)
    if random.random() < mutation_rate:
        mutant['batch_size'] = random.choice(batch_size_options)
    return mutant


def genetic_algorithm_search(X_train, y_train, X_val, y_val, seq_len, num_features, future_days,
                             population_size=10, generations=3,
                             units_range=[32, 64, 128], dropout_range=(0.1, 0.3), batch_size_options=[8, 16, 32]):
    population = [generate_random_individual(units_range, dropout_range, batch_size_options)
                  for _ in range(population_size)]
    best_loss = float('inf')
    best_config = None


def genetic_algorithm_search(X_train, y_train, X_val, y_val, seq_len, num_features, future_days,
                             population_size=10, generations=3,
                             units_range=[32, 64, 128],
                             dropout_range=(0.1, 0.3),
                             batch_size_options=[8, 16, 32]):
    population = [generate_random_individual(units_range, dropout_range, batch_size_options)
                  for _ in range(population_size)]

    best_loss = float('inf')
    best_config = None

    for generation in range(generations):
        print(f"\n--- Generation {generation + 1} ---")
        evaluated = []
        for individual in population:
            loss = train_evaluate_lstm(individual, X_train, y_train, X_val, y_val,
                                       seq_len, num_features, future_days)
            evaluated.append((individual, loss))
            print(f"Individual: {individual}, Loss: {loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                best_config = individual

        # Selection
        evaluated.sort(key=lambda x: x[1])
        survivors = [x[0] for x in evaluated[:population_size // 2]]

        # Crossover + Mutation
        population = survivors.copy()
        while len(population) < population_size:
            p1, p2 = random.sample(survivors, 2)
            child = crossover(p1, p2)
            child = mutate(child, units_range, dropout_range, batch_size_options)
            population.append(child)

    print(f"\n--- Genetic Algorithm Search Finished ---")
    print(f"Best Configuration: {best_config}")
    print(f"Best Validation Loss: {best_loss:.4f}")
    return best_config

    ######################
    # 데이터 다운로드 및 전처리


df = yf.download("GOOGL", period="3000d", auto_adjust=True)
df.dropna(inplace=True)
df['MA7'] = df['Close'].rolling(7).mean()
df['MA14'] = df['Close'].rolling(14).mean()
df['MA50'] = df['Close'].rolling(50).mean()


def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


df['RSI'] = compute_RSI(df['Close'])
df.dropna(inplace=True)

features = ['Close', 'MA7', 'MA14', 'MA50', 'RSI']
target_idx = features.index('Close')
seq_len, future_days = 60, 5
num_features = len(features)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])


def create_sequences(data, target_idx, seq_len, future_len):
    X, y = [], []
    for i in range(seq_len, len(data) - future_len):
        X.append(data[i - seq_len:i])
        y.append(data[i:i + future_len, target_idx])
    return np.array(X), np.array(y)


X, y = create_sequences(scaled, target_idx, seq_len, future_days)

tscv = TimeSeriesSplit(n_splits=5)
train_idx, val_idx = next(tscv.split(X))
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# GA 실행
best_lstm_config = genetic_algorithm_search(X_train, y_train, X_val, y_val,
                                            seq_len, num_features, future_days)

# 결과 저장
with open("best_lstm_config.json", "w") as f:
    json.dump(best_lstm_config, f, indent=2)

print(" LSTM 최적 하이퍼파라미터 저장 완료: best_lstm_config.json")
