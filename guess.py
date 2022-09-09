import numpy as np
import pandas as pd

# Useless  because we will prefer matrix multiplication to estimate prices
def estimatePrice(mileage, thetas) -> int:
	return thetas[0] + thetas[1] * mileage

def main():
	MODEL_FILE_PATH	= "./model"
	DEFAULT_THETAS	= [0, 0]

	# Read thetas from file or use default value
	thetas = DEFAULT_THETAS
	try:
		data = pd.read_csv(MODEL_FILE_PATH, header=None).to_numpy()[0]
		thetas = data[:2]
		min_x, max_x = data[2], data[3] 
		print(f'Found thetas in model file, using {thetas}')
	except Exception:
		print(f'Model file not found, using default values for thetas ({DEFAULT_THETAS})')

	# Get mileage from user input
	mileage = int(input("What mileage would you like to use ? (number between 0 and 1 000 000)\n"))

	# Handle errors with user input
	if mileage < 0 or mileage > 1000000: raise ValueError("mileage should be between 0 and 1 000 000")

	print(f'Estimated price: {estimatePrice((mileage - min_x) / (max_x - min_x), thetas):.0f}')


if __name__ == "__main__":
	main()
