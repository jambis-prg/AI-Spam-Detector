import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB, BernoulliNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay

from collections import defaultdict

from sklearn.base import BaseEstimator, ClassifierMixin

class PCAPredictWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, pca):
        super().__init__()
        self.model = model
        self.pca = pca
        self.n_features_in_ = pca.n_components_  # Define o número de features de entrada
        # Marca o estimador como ajustado (já que model e pca são pré-treinados)
        self.is_fitted_ = True

    def fit(self, X, y=None):
        # Método fit dummy para compatibilidade com a API do scikit-learn.
        # Não é necessário ajustar, pois model e pca já estão treinados
        return self

    def predict(self, X_reduced):
        # Faz a previsão no espaço original após inverter a transformação PCA.
        X_original = self.pca.inverse_transform(X_reduced)
        return self.model.predict(X_original)
    
    # Adicione se seu modelo base tiver predict_proba (opcional)
    def predict_proba(self, X_reduced):
        X_original = self.pca.inverse_transform(X_reduced)
        return self.model.predict_proba(X_original)

# Como Usar:
# modelo = joblib.load("nome do modelo.pkl")
# novo_email = np.array([[0, 0.2, 0.1, 0, 0.5, 0, 0.3]])
# previsão = modelo.predict(novo_email)

f = [MultinomialNB, GaussianNB, ComplementNB, BernoulliNB]
matriz = np.genfromtxt("./spambase_data.csv", delimiter=",", skip_header=1)
X, y = matriz[:, :-1], matriz[:, -1]

# Reduzir de 58 para 2 dimensões com PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Dicionário para armazenar os resultados
d = defaultdict(list)

labels_confusion_matrix = [ 1, 0 ]
labels_confusion_matrix_display = [ "spam", "not spam"]

k = 0
for i in f:
    for j in range(1, 10):
        _, X_test, _, y_test = train_test_split(X, y, test_size=(j/10), random_state=0)
        model = joblib.load(f"models/{i.__name__}_{j}.pkl")
        
        y_pred = model.predict(X_test)
        acertos = np.sum(y_test == y_pred)  # Número de acertos
        total = y_test.size  # Total de elementos

        if j == 9:
            fig, ax = plt.subplots()

            wrapper = PCAPredictWrapper(model, pca)
            wrapper.fit(X_reduced, y_test)
            
            DecisionBoundaryDisplay.from_estimator(
                estimator = wrapper,  # Passa o wrapper como estimador
                X = X_reduced,        # Usa os dados reduzidos para visualização
                response_method = "predict",
                cmap = plt.cm.Paired, 
                ax = ax, 
                alpha = 0.3
            )

            # 4. Plotar os dados no gráfico reduzido
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
            plt.xlabel("Componente Principal 1")
            plt.ylabel("Componente Principal 2")
            plt.title("Fronteira de Decisão - Naïve Bayes (PCA apenas para visualização)")
            plt.savefig(f"images/decision_boundaries/dcb_{i.__name__}.jpeg")
            plt.close(fig)

        cm = confusion_matrix(y_test, y_pred, labels = labels_confusion_matrix)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels_confusion_matrix_display)

        fig, ax = plt.subplots(figsize=(6, 6))  # Opcional: definir o tamanho da imagem
        disp.plot(ax=ax)
        plt.savefig(f"images/confusion_matrixes/cm_{i.__name__}_{j}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Calcula a porcentagem de acertos
        porcentagem_acertos = round((acertos / total) * 100, 2)
        d[k].append(porcentagem_acertos)
        
    k += 1

plt.figure(figsize=(10, 6))

x_values = np.linspace(0.1, 0.9, 9)
labels = ["MultinomialNB", "GaussianNB", "ComplementNB", "BernoulliNB"]
k = 0
for key, values in d.items():
    plt.plot(x_values, values, label=labels[k])  # Plotando cada lista
    k += 1

# Configurações do gráfico
plt.title("Tamanho do test com o tipo de modelo")
plt.xlabel("Tamanho do split de test")
plt.ylabel("% de acerto")
plt.legend()    # Mostra a legenda
plt.grid(True)  # Adiciona uma grade
plt.savefig("images/grafico.jpeg")