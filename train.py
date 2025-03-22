import numpy as np
import matplotlib.pyplot as plt
import joblib # usado para salvar o treinamento e carregar em outro arquivo

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB, BernoulliNB


# Modelos do Naive Bayes:
f = [MultinomialNB, GaussianNB, ComplementNB, BernoulliNB]

# Carrega os Dados:
matriz = np.genfromtxt("./spambase_data.csv", delimiter=",", skip_header=1)
X, y = matriz[:, :-1], matriz[:, -1]

# Treina e salva cada Modelo
k = 0
for i in f:
    for j in range(1, 10):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=(j/10), random_state=0)
        gnb = i()
        gnb.fit(X_train, y_train)
        
        # Salvar o modelo
        joblib.dump(gnb, f"models/{i.__name__}_{j}.pkl")
        print(f"Modelo {i}-{j} treinado e salvo")
        
    k += 1