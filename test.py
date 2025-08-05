import matplotlib.pyplot as plt

# Données
x = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
y = [
    0.002227043185651742, 0.0004454005938132842, 0.00022268822660026963,
    0.0001484576320741592, 0.00011134307100847673, 8.907523601833778e-05,
    7.422992308359772e-05, 6.362544722556447e-05, 5.5670265174556643e-05,
    4.94808270678516e-05, 4.452895504875737e-05
]

# Création du graphique
plt.figure()
plt.plot(x, y)
plt.plot(x, list(map(lambda c: 1/c * 2.2e-3, x)))

plt.xlabel('n_steps_sim')
plt.ylabel('diff COM start and COM end')
plt.xticks(x)
plt.tight_layout()
plt.show()
