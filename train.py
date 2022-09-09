import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Mean squared error
def MSE(y, y_hat):
	return np.mean(np.power(y - y_hat, 2))

# Used to normalize data
def minmax(lst):
	return (lst - np.min(lst)) / (np.max(lst) - np.min(lst))

def main():
	DATA_FILE_PATH	= "./data.csv"
	DEFAULT_THETAS	= [0, 0]
	MODEL_FILE_PATH	= "./model"
	TRAINING_STEPS	= 250
	LEARNING_RATE	= .1

	# Read data
	data = pd.read_csv(DATA_FILE_PATH).to_numpy(dtype=np.int64)

	# Format data
	x, y = data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1)

	# Normalize data
	min_x, max_x = x.min(), x.max()
	x = minmax(x)

	# Initialize thetas and loss history
	thetas = np.array(DEFAULT_THETAS, dtype=np.float64).reshape(-1, 1)
	loss_hist = []

	initial_y_hat = np.dot(np.c_[np.ones(x.size), x], thetas)

	for _ in range(TRAINING_STEPS):

		# Adds a columns with ones
		x_modif = np.c_[np.ones(x.size), x]

		# Calculates estimate prices for every piece of data
		y_hat = np.dot(x_modif, thetas)

		# Add loss to history
		loss_hist.append(MSE(y, y_hat))

		# Update thetas using given formula
		thetas = np.array([
			thetas[0] - LEARNING_RATE * np.mean(
				np.dot(
					x_modif, thetas
				) - y
			),
			thetas[1] - LEARNING_RATE * np.mean(
				(np.dot(
					x_modif, thetas
				) - y) * x
			),
		], dtype=np.float64)

	# Save thetas in a file
	df = pd.DataFrame(np.c_[thetas.T, min_x, max_x])
	print(f'Initial loss:		{MSE(y, initial_y_hat):.2f}\nLoss after training:	{MSE(y, y_hat):.2f}\nThetas:\n{thetas.T}')
	df.to_csv(MODEL_FILE_PATH, index=False, header=False)

	# Show data on graph
	_, ax = plt.subplots(1, 2)
	ax[0].scatter(x, y, label='True values')
	ax[0].plot(x, y_hat, label='Estimation', color='orange')
	ax[0].set(xlabel='Mileage', ylabel='Cost')
	ax[0].legend()

	ax[1].plot(np.arange(TRAINING_STEPS), loss_hist, label='MSE')
	ax[1].set(xlabel='Training steps', ylabel='Loss')
	ax[1].legend()

	plt.show()


if __name__ == "__main__":
	main()
