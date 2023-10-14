from sklearn.linear_model import LinearRegression
import pandas as pd

df_train = pd.read_csv('data/fish_train.csv')
df_test = pd.read_csv('data/fish_reserved.csv')

def clean(df):
    df[['Width', 'Height', 'Length1', 'Length2', 'Length3']] = df[['Width', 'Height', 'Length1', 'Length2', 'Length3']].apply(lambda x: x**2)
    dummies = pd.get_dummies(df['Species'], drop_first=True)
    df[list(dummies.columns)] = dummies
    df.drop(['Species'], axis=1, inplace=True)

clean(df_train)
clean(df_test)

x_train = df_train.drop(['Weight'], axis=1)
y_train = df_train['Weight']
x_test = df_test

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print(list(y_pred))