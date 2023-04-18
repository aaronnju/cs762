# Plotting a Pairplot of Titanic Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(
    'https://github.com/datagy/data/raw/main/titanic.csv',
    usecols=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
data = data.dropna()

sns.pairplot(data=data, hue='Survived')
plt.show()
