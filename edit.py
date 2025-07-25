import pandas as pd

df = pd.read_csv('egitim_verisi.csv')
df['GeneralCategory'] = df['GeneralCategory'].replace('ArtAndCulture', 'Art & Culture')
df['GeneralCategory'] = df['GeneralCategory'].replace('HealthAndWellness', 'Health & Wellness')
df['GeneralCategory'] = df['GeneralCategory'].replace('FoodAndDrink', 'Food & Drink')

df.to_csv('egitim_verisi_yeni.csv', index=False)
