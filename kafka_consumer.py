import dill
from kafka import KafkaConsumer
import json

# build a model
with open('ml_analysis_fraud.dill', 'rb') as f:
    model = dill.load(f)

# inisialisasi Kafka Consumer
consumer = KafkaConsumer(
    'transaction-stream',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.load(x.decode('utf-8'))
)

print("Listening for messages...")
for message in consumer:
    df_axis = message.value
    print(f"received transaction: {df_axis}")

    # ekstraksi fitur dari transaksi
    features = df_axis['fraud']
    prediction = model.predict([features])

    if prediction[0] == 1:
        print(f"fraud transactions based on amt data while getting fraud: {df_axis['amt']}")
    else:
        print(f"transaction is not fraud: {df_axis['amt']}")