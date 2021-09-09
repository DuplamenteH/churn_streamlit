from typing import List
from pandas.core.frame import DataFrame
import pickle
#from sklearn.impute import SimpleImputer
import pandas as pd

class Data_Prepared ():
    def __init__(self):
        self.le = pickle.load(open('label_encoder.pkl','rb'))
        self.mms = pickle.load(open('mms.pkl','rb'))
        #api/api_v1/model/mms.pkl
        #self.simp = SimpleImputer(strategy="mean")

    def get_df_transform(self,df:DataFrame):
        """
        parameter:
            df-> base de dados enviada pelo usuario

            Função irá retornar um df ja tratado
        """
        print('[INFO]--> Iniciando o tratamento dos seus dados ....')

        print('[INFO]--> Transformando os valores objects em valores numericos')
        for col in df.columns:
            if df[col].dtypes == object:
                df[col] = self.le.transform(df[col])


        print('[INFO]--> Verificando se há dados faltantes.')
        #if df.isnull().sum().sum()!=0:
        #    print("[INFO]--> Os dados apresentam valores faltantes.")
        #    print("[INFO]--> Fazendo as transformações dos devido dados faltante ")
        #    df = self.simp.fit_transform(df)
        #    print("[INFO]--> Dados faltantes foram substituidos por a media de sua coluna")
        #else:
        #    print("[INFO]--> Não tem dados faltantes")
        
        print("[INFO]--> deixando os dados em uma escala de 0 a 1")
        df_mms = self.mms.transform(df)
        print("[INFO]--> Processo de tratamento terminado, dados sendo retornados")
        df_final = pd.DataFrame(data=df_mms,columns=df.columns)
        #df_final['churn'] = pd.to_numeric(df_final['churn'],downcast='integer')
        return df_final
        
    def get_remove_cols(self,colunas:List,df:DataFrame):
        """
            args: cols-> Lista de colunas a serem removidas,
                        *importante*: por mais que seja so uma coluna passe uma lista.
                        df -> dataframe que as colunas seram removidas.
        """

        df_final=df.drop(columns=colunas)
        return df_final


  
    


    # Resolver este problema
        