import pickle
import streamlit as st

model = pickle.load(open('estimasi_profit.sav', 'rb'))

st.title('Estimasi Profit')

Day = st.number_input('Masukan Total Kill', min_value=1, step=1)
Year = st.number_input('Masukan Total Death Played', min_value=1, step=1)
Customer_Age = st.number_input('Masukan Total Kemenangan Team', min_value=1, step=1)
Unit_Cost = st.number_input('Masukan Total Kill Death Played')
Unit_Price = st.number_input('Masukan Ratio Headshot')
Cost = st.number_input('Masukan Ratio Headshot')

predict = ''

if st.button('Estimasi Profit'):
    predict = model.predict(
        [[ Day, Year, Customer_Age, Unit_Cost, Unit_Price, Cost]]
    )
    st.slider ('Estimasi Profit : ', predict)

    import streamlit as st


