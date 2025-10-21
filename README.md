# 🧠 ft_linear_regression

A simple project to implement **Linear Regression** from scratch in **C++**, built entirely around two **homemade libraries**:
1️⃣ a **Linear Algebra Library**
2️⃣ a **Neural Network Library**

These custom libraries were written from the ground up to deepen understanding of how mathematical operations and learning mechanisms work internally without relying on large external dependencies.

---

## 🧩 Overview

This project demonstrates how to perform **linear regression**, one of the most fundamental machine learning algorithms, using only self-made code and libraries.

It aims to:

* Reinforce the mathematical understanding of regression
* Explore low-level operations through a handcrafted **Linear Algebra Library**
* Bridge into neural architectures using a **custom Neural Network Library**

The goal is **educational clarity** and **reusability** of the libraries in other AI or physics simulations.

---
## 🚗 Goal of the Project



The purpose of this project is to predict the price of a car based on its mileage using linear regression.




Given a dataset that includes pairs of values:

x: mileage (in kilometers)
y: car price (in euros or another unit)



the program learns a linear model of the form:




[
y = θ₀ + θ₁ * x
]




where:

θ₀ is the intercept (price when mileage = 0)
θ₁ is the slope (how much the price decreases with each kilometer)



After training, the model can estimate a car’s price simply by inputting its mileage — a classic use case of linear regression 🧮.

---

## ⚙️ Features

✨ Implemented fully in modern C++
📊 Reads and trains from CSV data
🧮 Custom matrix and vector operations
🧠 Basic neural network integration for experiments
📈 Visualization script in Python (`graphic.py`)
🧱 Minimal dependencies — almost entirely self-contained

---

## 🖥️ Usage

1. Prepare your dataset (`data.csv`)
2. Build the project with `make`
3. Run the executable (`./train --dataset <datafile> [--epoch <epoch> --learning_rate <learning_rate> --loss_function <loss_function> --layer_activation <layer_activation> --output_activation <output_activation> --batch <batch>]`)
4. Visualize the results using `graphic.py`

Example:

```bash
make
./train --dataset data.csv
./compute <mileage> network.json
python3 graphic.py
```

---

## 📜 License

This project is open-source under the **MIT License**.

```
MIT License

Copyright (c) 2025 ...

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction...
```

