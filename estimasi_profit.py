import pickle
import streamlit as st

model = pickle.load(open('estimasi_profit.sav', 'rb'))

st.title('Estimasi Profit')

Day = st.number_input('Masukan Jumlah Hari', min_value=1, step=1)
Year = st.selectbox('Input Tahun',('2011', '2012', '2013','2014', '2015', '2016'))
Customer_Age = st.number_input('Input Umur Pelanggan', min_value=1, step=1)
Unit_Cost = st.text_input('Input Biaya unit')
Unit_Price = st.text_input('Input Harga Per unit')
Cost = st.text_input('Input Harga')

predict = ''

if st.button('Estimasi Profit'):
    predict = model.predict(
        [[ float(Day), float(Year), float(Customer_Age), float(Unit_Cost), float(Unit_Price), float(Cost)]]
    )
    st.write ('Estimasi Profit : ', predict)


