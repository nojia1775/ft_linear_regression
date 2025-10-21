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
	df = df.dropna(subset=[x, y])
	df = df.sort_values(by=x)
	return df

def function(x, weight, bias, max_x, max_y):
	x_norm = x / max_x
	y_norm = weight * x_norm + bias
	return y_norm * max_y

def draw_dataset(ax, dataset):
	ax.scatter(dataset['km'], dataset['price'], color='blue')
	ax.set_title("Price by mileage")
	ax.set_xlabel("Mileage")
	ax.set_ylabel("Price")
	ax.set_xlim(0, 250000)
	ax.set_ylim(0, 9000)
	ax.grid(True)

def draw_ai(ax, dataset):
	ax.plot(dataset['epoch'], dataset['r2'], color='green')
	ax.set_title("R squared")
	ax.set_xlabel("Epochs")
	ax.set_ylabel("R squared")
	ax.set_xlim(0, dataset['epoch'].size)
	ax.set_ylim(0, 1)
	ax.grid(True)

dataset_price = get_data("data.csv", 'km', 'price')
dataset_ai = get_data("ai.csv", 'epoch', 'r2')

nbr_file = 1
num_epochs = len(dataset_ai)

for i in range(0, num_epochs, max(1, num_epochs // 10)):
	fig, axes = plt.subplots(1, 2, figsize=(10, 4))

	draw_dataset(axes[0], dataset_price)
	weight = dataset_ai.loc[i, 'weight']
	bias = dataset_ai.loc[i, 'bias']
	y_pred = function(dataset_price['km'], weight, bias, dataset_price['km'].max(), dataset_price['price'].max())
	axes[0].plot(dataset_price['km'], y_pred, color='red')

	draw_ai(axes[1], dataset_ai.iloc[:i+1])

	plt.tight_layout()
	plt.savefig(f"graph_{nbr_file}.png")
	plt.close(fig)
	nbr_file += 1
