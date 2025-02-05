import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

file_path = 'proj1Dataset.xlsx'


class ClosedRegression:
    def __init__(self):
        self.df = pd.read_excel(file_path)
        # print(f"Data shape: {self.df.shape}")
        if self.df.isnull().values.any():
            # print("Data contains missing values")
            self.df = self.df.dropna()

        # print(f"Data shape after deleting missing values: {self.df.shape}")    

        self.preprocessing()
        self.pseudo_inverse()
        self.predict()
        # self.plotting()

    def preprocessing(self):
        self.features = self.df.iloc[:, :-1].values
        self.target = self.df.iloc[:, -1].values    
        self.x = np.asmatrix(self.features, dtype=float)
        self.t = np.asmatrix(self.target, dtype=float)
        self.t = self.t.T
        self.ones = np.ones((self.x.shape[0]), dtype=float)
        self.ones = np.asmatrix(self.ones).T
        self.X_ = np.hstack((self.x, self.ones))    

    def pseudo_inverse(self):
        first_part = np.dot(self.X_.T, self.X_)
        second_part = np.dot(self.X_.T , self.t)
        self.w = np.dot(np.linalg.pinv(first_part), second_part)   
        # print(f"Weight vector: {self.w}")

    def predict(self):
        self.y = np.dot(self.X_, self.w)
        # print(f"Predicted values: {self.y}")
    # def plotting(self):
    #       plt.scatter(self.features, self.target, color='red', marker='x')
    #       plt.plot(self.features, self.y, color='blue', label='Closed Form')
    #       plt.title("Matlab\'s \"carbig\" dataset")
    #       plt.xlabel('Weight')
    #       plt.ylabel('Horsepower')
    #       plt.legend(loc = "upper right")
    #       plt.show()




class GradientDescentRegression:
        
    def __init__(self):
        self.df = pd.read_excel(file_path)
        if self.df.isnull().values.any():
            self.df = self.df.dropna() 
        self.preprocessing()
        self.W = np.random.rand(self.X_.shape[1], 1)
        # self.W = np.ones((self.X_.shape[1], 1))  # Set initial weights to [1,1,...]
        # print(f"Weight vector shape: {self.W.shape}")
        self.calculate_gradient()
        self.gradient_descent()
        self.predict()
        # self.plotting()

    def preprocessing(self):
        self.features = self.df.iloc[:, :-1].values
        self.target = self.df.iloc[:, -1].values
        # Scale features: (x - mean) / std deviation
        feature_means = np.mean(self.features, axis=0)
        feature_stds = np.std(self.features, axis=0)
        self.x = (self.features - feature_means) / feature_stds
        self.x = np.asmatrix(self.x, dtype=float)
        self.t = np.asmatrix(self.target, dtype=float)
        # By definition of target vector should be column vector
        self.t = self.t.T
        self.ones = np.ones((self.x.shape[0], 1), dtype=float)
        self.ones = np.asmatrix(self.ones)
        self.X_ = np.hstack((self.x, self.ones))
    
    def calculate_gradient(self):
        # print(f"W : {self.W}")
        first_part = np.dot(self.W.T, self.X_.T)
        first_part = 2 * np.dot(first_part, self.X_)
        second_part = 2 * np.dot(self.t.T, self.X_)
        gradient = first_part - second_part
        return gradient

    def gradient_descent(self):
        learning_rate = 0.001
        for i in range(10):
            gradient = self.calculate_gradient().T
            self.W -= learning_rate * gradient

    def predict(self):

        self.y = np.dot(self.X_, self.W)
        # print(f"Predicted values: {self.y}")

    # def plotting(self):
    #     plt.scatter(self.features, self.target, color='red', marker='x')
    #     plt.plot(self.features, self.y, color='green', label='Gradient Descent')
    #     plt.title("Matlab's \"carbig\" dataset")
    #     plt.xlabel('Weight')
    #     plt.ylabel('Horsepower')
    #     plt.legend(loc="upper right")
    #     plt.show()
  


                 

                    
if __name__ == "__main__":

    object_regression = ClosedRegression()
    object_gradient = GradientDescentRegression()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 1st model
    axes[0].scatter(object_regression.features, object_regression.target, color='red', marker='x')
    axes[0].plot(object_regression.features, object_regression.y, color='blue', label='Closed Form')
    axes[0].set_title("Matlab's \"carbig\" dataset")
    axes[0].set_xlabel('Weight')
    axes[0].set_ylabel('Horsepower')
    axes[0].legend(loc="upper right")

    #2nd model
    axes[1].scatter(object_gradient.features, object_gradient.target, color='red', marker='x')
    axes[1].plot(object_gradient.features, object_gradient.y, color='green', label='Gradient Descent')
    axes[1].set_title("Matlab's \"carbig\" dataset")
    axes[1].set_xlabel('Weight')
    axes[1].set_ylabel('Horsepower')
    axes[1].legend(loc="upper right")
    plt.show()



