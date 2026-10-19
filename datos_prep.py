
import pandas as pd

def limpiar_datos(ruta):
    df = pd.read_csv(ruta)
    df = df.drop_duplicates()
    df = df.dropna()
    return df
