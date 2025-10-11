import matplotlib.pyplot as plt
import pandas as panda

def get_data(file, x, y):
	try:
		df = panda.read_csv(file)
	except FileNotFoundError:
		print(f"Error: {file} not found")
		exit(1)
	df[x] = panda.to_numeric(df[x], errors='coerce')
	df[y] = panda.to_numeric(df[y], errors='coerce')
	df = df.sort_values(by=x)
	df = df.sort_values(by=y)
	return df

def function(x, weight, bias, max_x, max_y):
	x_norm = x / max_x
	y_norm = weight * x_norm + bias
	return y_norm * max_y

def draw_dataset(dataset):
	plt.scatter(dataset['km'], dataset['price'], color='blue')
	plt.title("Price by mileage")
	plt.xlabel("Mileage")
	plt.ylabel("Price")
	plt.xlim(0, 250000)
	plt.ylim(0, 9000)
	plt.grid(True)

dataset = get_data("data.csv", 'km', 'price')
ai = get_data("ai.csv", 'weight', 'bias')

i = 0
nbr_file = 1
while i < len(ai):
	plt.clf()
	draw_dataset(dataset)
	plt.plot(dataset['km'], function(dataset['km'], ai.loc[int(i), 'weight'], ai.iloc[int(i), 1], dataset['km'].max(), dataset['price'].max()), color='red')
	plt.savefig("graph" + str(nbr_file) + ".png")
	nbr_file += 1
	i += (len(ai) - 1) / 10