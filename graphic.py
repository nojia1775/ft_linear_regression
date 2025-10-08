import matplotlib.pyplot as plt
import pandas as pd

# Lire les données
df1 = pd.read_csv("data.csv", header=None, names=['km', 'price'])
df2 = pd.read_csv("test.csv", header=None, names=['km', 'price'])

# Tracer la première droite (ou points)
df1.sort_values('price')
plt.plot(df1['km'], df1['price'], color='blue', label='Droite 1')


# Tracer la deuxième droite (ou points)
# plt.plot(df2['km'], df2['price'], color='red', label='Droite 2')

# Ajouter titre et labels
plt.title("Deux droites sur le même graphique")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.legend()   # Affiche la légende
plt.grid(True)

plt.show()
