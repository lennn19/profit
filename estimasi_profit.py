import pickle
import streamlit as st

model = pickle.load(open('estimasi_profit.sav', 'rb'))

st.title('Estimasi Profit')

Day = st.number_input('Masukan Jumlah Hari', min_value=1, step=1)
Year = st.number_input('Masukan Jumlah Tahun', min_value=1, step=1)
Customer_Age = st.number_input('Masukan Umur Pelanggan', min_value=1, step=1)
Unit_Cost = st.number_input('Masukan Biaya unit')
Unit_Price = st.number_input('Masukan Harga Per unit')
Cost = st.number_input('Input Harga')

predict = ''

if st.button('Estimasi Profit'):
    predict = model.predict(
        [[ Day, Year, Customer_Age, Unit_Cost, Unit_Price, Cost]]
    )
    st.write ('Estimasi Profit : ', predict)

    import streamlit as st


