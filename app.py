import pickle
import pandas as pd
from pandas import DataFrame
from data_prep import Data_Prepared
import streamlit as st


df = pd.read_csv('https://raw.githubusercontent.com/DuplamenteH/machine-learning-projects/main/Churn/dados/customer-churn-prediction-2020/train.csv')
df.drop(columns=['churn'],inplace=True);
#load model


def get_model():
    model = pickle.load(open('churn_randomFlorest.pkl','rb'))
    return model

modelo = get_model()

data_pre = Data_Prepared()
#predict
def predict(df_raw:DataFrame):
    colunas_remover = ['state','area_code']
    df_ = data_pre.get_remove_cols(colunas=colunas_remover,df = df_raw)
    df_final = data_pre.get_df_transform(df_)

    pred = modelo.predict(df_final)
    df_final['predicao'] = pred


    st.dataframe(df_final)

    return pred

def predict_prob(df_raw:DataFrame):
    colunas_remover = ['state','area_code']
    df_ = data_pre.get_remove_cols(colunas=colunas_remover,df = df_raw)
    df_final = data_pre.get_df_transform(df_)

    prob = modelo.predict_proba(df_final)
    aux1 = prob[0][0]
    aux2 = prob[0][1]


    return aux1,aux2







#side bar
st.sidebar.header("Sobre")
st.sidebar.markdown("[artigo](https://portfolio.stacktecnologias.com/cmatheus/19/)")
st.sidebar.subheader("Autor: Carlos Matheus")



st.title("Churn Predict")
st.image('churn.jpg')


st.text("Preencha o formulário abaixo para realizar sua preedição : ")

with st.form(key='form-predict'):
    state = st.text_input("estado",value="MA")
    account_length = st.number_input(label='account_length',value=120)
    area_code = st.text_input("codigo da area",value='area_code_444')
    internacional_plan = st.radio("Possui plano internacional", options=('yes','no'))
    voice_mail_plan = st.radio("Possui serviço de correio de voz", options=('yes','no'))
    number_vmail_messages = st.number_input(label='Digite o numero de mensagens de voz que você tem caso não tenha nenhuma apenas ignore',value=0)
    total_day_minutes = st.number_input(label='Minutagem diaria',value=106.6)
    total_day_calls = st.number_input(label='Chamadas diaria',value=70)
    total_day_charge = st.number_input(label='Recargas diaria',value=42.50)
    total_eve_minutes = st.number_input(label='Minutagem vespertino',value=42.50)
    total_eve_calls = st.number_input(label='Chamadas vespertino',value=40)
    total_eve_charge = st.number_input(label='Recarga vespertino',value=40)
    total_night_minutes = st.number_input(label='Minutagem nortunas',value=40)
    total_night_calls =st.number_input(label='Chamadas nortunas',value=40)
    total_night_charge = st.number_input(label='Recargas nortunas',value=40)
    total_intl_minutes =st.number_input(label='Total de minutos',value=40)
    total_intl_calls =st.number_input(label='Total de chamadas',value=40)
    total_intl_charge =st.number_input(label='Total de recargas',value=40)
    number_customer_service_calls =st.number_input(label='Numeros de serviços de chamada',value=0, min_value=0, max_value=8)

    
    data = [state,account_length,area_code,internacional_plan,voice_mail_plan,number_vmail_messages,total_day_minutes,
        total_day_calls, total_day_charge, total_day_minutes,total_eve_calls,total_eve_charge, total_night_minutes,total_night_calls,total_night_charge,
        total_intl_minutes,total_intl_calls, total_intl_charge,number_customer_service_calls
    ]
    cols = ['state', 'account_length', 'area_code', 'international_plan','voice_mail_plan', 'number_vmail_messages', 'total_day_minutes','total_day_calls', 'total_day_charge', 'total_eve_minutes','total_eve_calls', 'total_eve_charge', 'total_night_minutes','total_night_calls', 'total_night_charge', 'total_intl_minutes','total_intl_calls', 'total_intl_charge','number_customer_service_calls']
    submitBnT = st.form_submit_button(label='Predict')

    if submitBnT:
        df_form = pd.DataFrame([data],columns=cols)
        st.dataframe(df_form)

        pred =predict(df_form)
        prob1,prob2 = predict_prob(df_form)

        if pred==0:
            st.success("O seu cliente permanecerá na sua empresa, procure dar benefícios , assim ele irá sentir-se valorizado")
            st.info("Probabilidades da predição quanto ao cliente {}%  de ficar , {}%  de sair".format(round(prob1*100,3),round(prob2*100,3)))
        else:
            st.error("Cliente saiu da sua empresa.")
            st.info("Probabilidades da predição quanto ao cliente:  {}%  de ficar , {}%  de sair".format(round(prob1*100,3),round(prob2*100,3)))
            


