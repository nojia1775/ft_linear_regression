import matplotlib.pyplot as plt
import pandas as panda

df = panda.read_csv("data.csv")
df['km'] = panda.to_numeric(df['km'], errors='coerce')
df['price'] = panda.to_numeric(df['price'], errors='coerce')
df = df.sort_values(by='km')
df = df.sort_values(by='price')
train = panda.read_csv("train.csv", header=None, names=['weight', 'bias'])
train['weight'] = panda.to_numeric(train['weight', errors='coerce'])
train['bias'] = panda.to_numeric(train['bias', errors='coerce'])
train = train.sort_values(by='weight')
train = train.sort_values(by='bias')
plt.scatter(df['km'], df['price'], color='blue')
plt.title("Price by mileage")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.grid(True)
plt.savefig("graph.png")