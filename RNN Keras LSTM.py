import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback, LearningRateScheduler
from termcolor import colored

# Vocabulary
all_chars = '0123456789+'
num_features = len(all_chars)
char_to_index = {c: i for i, c in enumerate(all_chars)}
index_to_char = {i: c for i, c in enumerate(all_chars)}

# Generate synthetic data
def generate_data():
    first = np.random.randint(0, 100)
    second = np.random.randint(0, 100)
    example = str(first) + '+' + str(second)
    label = str(first + second)
    return example, label

# Vectorize examples
def vectorize_example(example, label, max_time_steps):
    x = np.zeros((max_time_steps, num_features))
    y = np.zeros((max_time_steps, num_features))
    
    diff_x = max_time_steps - len(example)
    diff_y = max_time_steps - len(label)
    
    for i, c in enumerate(example):
        x[i + diff_x, char_to_index[c]] = 1
    for i in range(diff_x):
        x[i, char_to_index['0']] = 1
    for i, c in enumerate(label):
        y[i + diff_y, char_to_index[c]] = 1
    for i in range(diff_y):
        y[i, char_to_index['0']] = 1
    return x, y

# Devectorize examples
def devectorize_example(example):
    result = [index_to_char[np.argmax(vec)] for vec in example]
    return ''.join(result)

# Create dataset
def create_dataset(num_examples=10000, max_time_steps=5):
    x = np.zeros((num_examples, max_time_steps, num_features))
    y = np.zeros((num_examples, max_time_steps, num_features))
    for i in range(num_examples):
        e, l = generate_data()
        e_v, l_v = vectorize_example(e, l, max_time_steps)
        x[i] = e_v
        y[i] = l_v
    return x, y

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)

# Build the model
hidden_units = 128
max_time_steps = 5

model = Sequential([
    Embedding(input_dim=num_features, output_dim=hidden_units, input_length=max_time_steps),
    LSTM(hidden_units, return_sequences=False),
    RepeatVector(max_time_steps),
    LSTM(hidden_units, return_sequences=True),
    TimeDistributed(Dense(num_features, activation='softmax'))
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Generate dataset
x, y = create_dataset()
x_test, y_test = create_dataset(num_examples=1000)  # Test dataset

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model
history = model.fit(
    x, y,
    epochs=100,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stopping, lr_callback],
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict and print results
preds = model.predict(x_test)
for i, pred in enumerate(preds[:10]):  # Print first 10 predictions
    y_true = devectorize_example(y_test[i])
    y_pred = devectorize_example(pred)
    col = 'green' if y_true == y_pred else 'red'
    out = f"Input: {devectorize_example(x_test[i])} | True: {y_true} | Pred: {y_pred}"
    print(colored(out, col))