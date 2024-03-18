import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import plotly.express as px

st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded",)

summ =0

# global variables
uploaded_file = None



# Создание переменных session state
if 'df_input' not in st.session_state:
    st.session_state['df_input'] = pd.DataFrame()

if 'df_predicted' not in st.session_state:
    st.session_state['df_predicted'] = pd.DataFrame()

if 'tab_selected' not in st.session_state:
    st.session_state['tab_selected'] = None

def reset_session_state():
    st.session_state['df_input'] = pd.DataFrame()
    st.session_state['df_predicted'] = pd.DataFrame()


# Prepare data section

# ML section start
numerical = ['tenure','warehouse_to_home','hour_spend_on_app','number_of_device_registered','number_of_address','order_amount_hike_fromlast_year',
             'coupon_used','order_count','day_since_last_order','cashback_amount']
categorical = ['preferred_payment_mode','preferred_login_device', 'gender','city_tier','prefered_order_cat','marital_status','complain','satisfaction_score'] 

le_enc_cols = ['preferred_login_device','gender']
gender_map = {'male': 0, 'female': 1}
device_map = {'phone': 1, 'computer': 0}

# logistic regression model
model_file_path = 'models/churn_prediction_model.sav'
model = pickle.load(open(model_file_path, 'rb'))   

# encoding model DictVectorizer
encoding_model_file_path = 'models/churn_prediction_model_2.sav'
encoding_model = pickle.load(open(encoding_model_file_path, 'rb'))

# Кэширование функции предсказания
@st.cache_data
def predict_churn(df_input, treshold):
    # Функция для предсказания оттока
    scaler = MinMaxScaler()

    df_original = df_input.copy()
    df_input[numerical] = scaler.fit_transform(df_input[numerical])

    for col in le_enc_cols:
        if col == 'gender':
            df_input[col] = df_input[col].map(gender_map)
        else:
            df_input[col] = df_input[col].map(device_map)

    dicts_df = df_input[categorical + numerical].to_dict(orient='records')
    X = encoding_model.transform(dicts_df)
    # X[np.isnan(X)] = 0
    y_pred = model.predict_proba(X)[:, 1]
    churn_descision = (y_pred >= treshold).astype(int)
    df_original['churn_predicted'] = churn_descision
    df_original['churn_predicted_probability'] = y_pred

    return df_original

@st.cache_data
def convert_df(df):
    # Функция для конвертации датафрейма в csv
    return df.to_csv(index=False).encode('utf-8')

with st.sidebar:
    st.title('Ввод данных')
    tab1, tab2 = st.tabs(["_Из файла_", "_Вручную_"])

    with tab1:
        st.header("Импорт базы данных 📑  .xlsx")
        uploaded_file = st.file_uploader("Выберите файл:",on_change=reset_session_state)
        if uploaded_file is not None:
            treshold = st.slider('Порог вероятности оттока', 0.0, 1.0, 0.5, 0.1, key='slider1')
            prediction_button = st.button('Предсказать', type='primary', use_container_width=True, key='button1')
            global df_input
            st.session_state['df_input'] = pd.read_excel(uploaded_file, sheet_name='E Comm', index_col= 0)
            if prediction_button:
                # Предсказание и сохранение в session state
                st.session_state['df_predicted'] = predict_churn(st.session_state['df_input'], treshold)
                st.session_state['tab_selected'] = 'tab1'


    with tab2:
        st.header("Ручной ввод параметров ✌🏻")
        st.write('Заполните поля данными для получения предсказания оттока клиентов.')
         # Вкладка с вводом данных вручную, выбором порога и кнопкой предсказания (вкладка 2)
        customer_id = st.text_input('Customer ID', placeholder='00000', help='Введите ID клиента')
        tenure = st.number_input('Срок пользования клиентом сервиса, (мес)', min_value=0, max_value=100, value=0)  
        warehouse_to_home = st.number_input ('Расстояние от дома до склада, (км)',  min_value=2, max_value=150, value=2)
        hour_spend_on_app = st.number_input ('Время проведенное в приложении, (час)', min_value=0, max_value= 6, value=0)
        number_of_device_registered = st.number_input ('Количество зарегистрированных устройств', min_value=1, max_value= 10, value=1)
        satisfaction_score = st.number_input ('Оценка удовлетворенности клиента', min_value=1, max_value= 5, value=1)
        number_of_address = st.number_input ('Количество зарегистрированных адресов клиента', min_value=1, max_value= 25, value=1)
        order_amount_hike_fromlast_year = st.number_input('Увеличение суммы заказа по сравнению с прошлым месяцем, (%)',min_value=10, max_value= 30, value=10)
        coupon_used = st.number_input ('Количество использованных купонов', min_value=0, max_value= 15, value=0)
        order_count = st.number_input ('Количество заказов в месяц', min_value=0, max_value= 20, value=0)
        day_since_last_order = st.number_input ('Количество дней прошедших от последнего заказа', min_value=0, max_value= 60, value=0)
        cashback_amount = st.number_input ('Сумма начисленного кэшбэка', min_value=0, max_value= 350, value=0)
        complain = st.selectbox ('Наличие жалобы', (' yes','no'))   # В словарике сделать как в примере с 'seniorcitizen': 1 if senior_citizen == 'Да' else 0,
        city_tier = st.selectbox ('Зона города',(1,2,3))
        preferred_login_device = st.selectbox ('Устройство входа клиента',('computer','phone')) 
        preferred_payment_mode = st.selectbox ('Метод оплаты',('debit_card','credit_card','upi','cc','cod','e_wallet','cash_on_delivery'))
        gender = st.selectbox( 'Пол', ('female', 'male'))
        marital_status = st.selectbox( 'Семейное положение', ('single', 'married','divorced'))
        prefered_order_cat = st.selectbox('Предпочтительная категория заказа клиента', ('laptop__accessory', 'mobile', 'mobile_phone', 'fashion', 'grocery', 'phone'))    
        
        # Если введен ID клиента, то показываем слайдер с порогом и кнопку предсказания
        if customer_id != '':
            treshold = st.slider('Порог вероятности оттока', 0.0, 0.497, 0.5, 0.01, key='slider2')
            prediction_button_tab2 = st.button('Предсказать', type='primary', use_container_width=True, key='button2')
            
            if prediction_button_tab2:
                st.session_state['tab_selected'] = 'tab2'
                # Сохраняем введенные данные в session state в виде датафрейма
                st.session_state['df_input'] = pd.DataFrame({
                    'customer_id': customer_id,
                    'tenure': tenure,
                    'warehouse_to_home': warehouse_to_home,
                    'hour_spend_on_app': hour_spend_on_app,
                    'number_of_device_registered': number_of_device_registered,
                    'satisfaction_score': satisfaction_score,
                    'number_of_address': number_of_address,
                    'order_amount_hike_fromlast_year': order_amount_hike_fromlast_year,
                    'coupon_used': coupon_used,
                    'order_count': order_count,
                    'day_since_last_order': day_since_last_order,
                    'cashback_amount': cashback_amount,
                    'complain': 1 if complain == 'yes' else 0,
                    'city_tier': city_tier,
                    'preferred_login_device' : preferred_login_device,
                    'preferred_payment_mode': preferred_payment_mode,
                    'gender' : gender,
                    'marital_status': marital_status,
                    'prefered_order_cat': prefered_order_cat}, index=[0])

                # Предсказание и сохранение в session state
                st.session_state['df_predicted'] = predict_churn(st.session_state['df_input'], treshold)
           
# Sidebar section end

# Main section start
# Основной блок
st.title('Прогноз оттока клиентов 🔀')
st.write('Прогнозирование величин, на основе математической модели логистической регрессии')    


tab3, tab4, tab5, tab6  = st.tabs(["Импортированные данные","Графики и диаграммы","Результаты прогнозирования","Комментарии и выводы"])
with tab3:
    if uploaded_file == None:
        st.write('Пока нет данных')
    else: # Производим автоматическую коорекцию названий столбцов:
        if uploaded_file == None:
            pass
        else:
            indx = []
            listing = []
            list_result = []
            index_value = ''
            # Перебираем названия столбцов датасета
            for j in range (0, len(st.session_state['df_input'].columns)):
                if j == 0:
                    for n in range (len(st.session_state['df_input'].columns[j])): # Диапазон до длины названия первого столбца
                        listing.insert (n,((st.session_state['df_input'].columns[j])[n])) # Преобразуем в список, из букв, разделеннных запятой. list - пока содержит название первого столбца
                    for i in range (len(listing)):
                        if listing[i].isupper(): # Если в списке найдена большая буква, заменяется на '_'+ 'исходная малая буква'
                            list_result.extend('_') # Записываем новый список в list_result
                            k = listing[i].lower()
                            list_result.extend(k)
                        else:
                            list_result.extend(listing[i]) # Остальные буквы пишем как есть
                    del list_result[0]
                    del list_result[10] # Удаляем первый и двенадцатый знак '_'
                    index_value =''.join(map(str, list_result))
                    del listing[:]
                    del list_result[:]
                    indx.append(index_value)
                    index_value =''

                else:
                    for n in range (len(st.session_state['df_input'].columns[j])): # Диапазон до длины названия остальных столбцов
                        listing.insert (n,((st.session_state['df_input'].columns[j])[n])) # Преобразуем в список, из букв, разделеннных запятой. list - пока содержит название первого столбца
                    for i in range (len(listing)):
                        if listing[i].isupper(): # Если в списке найдена большая буква, заменяется на '_'+ 'исходная малая буква'
                            list_result.extend('_') # Записываем новый список в list_result
                            k = listing[i].lower()
                            list_result.extend(k)
                        else:
                            list_result.extend(listing[i]) # Остальные буквы пишем как есть
                    del list_result[0] # Удаляем только первый знак '_'
                    index_value =''.join(map(str, list_result)) # Переводим список в строку
                    del listing[:]
                    del list_result[:]
                    indx.append(index_value)
        st.session_state['df_input'].columns = indx
        st.session_state['df_input'] = st.session_state['df_input'].reset_index()
        st.session_state['df_input'] = st.session_state['df_input'].drop('index', axis = 1)
        st.session_state['df_input'] 
    
     
with tab4:
    if uploaded_file == None:
        st.write('Пока нет данных')
    else:
        st.write( 1) 
with tab5:
    
     # Если предсказание еще не было сделано, то выводим входные данные в общем виде
    if len(st.session_state['df_predicted']) == 0:
        st.write('Пока нет данных')
    else:
        # Если предсказание уже было сделано, то выводим входные данные в expander
        with st.expander("Входные данные"):
            st.write(st.session_state['df_input'])
    # Примеры визуализации данных
    # st.line_chart(st.session_state['df_input'][['tenure', 'cashback_amount']])
    # st.bar_chart(st.session_state['df_input'][['contract']])
            
    # Выводим результаты предсказания для отдельного клиента (вкладка 2)
    if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab2':
        if st.session_state['df_predicted']['churn_predicted'][0] == 0:
            st.image('https://gifdb.com/images/high/happy-face-steve-carell-the-office-057k667rwmncrjwh.gif', width=200)
            st.subheader(f'Клиент :green[остается] c вероятностью {(1 - st.session_state["df_predicted"]["churn_predicted_probability"][0]) * 100:.2f}%')
        else:
            st.image('https://media.tenor.com/QFnU4bhN8gMAAAAd/michael-scott-crying.gif', width=200)
            st.subheader(f'Клиент :red[уходит] c вероятностью {(st.session_state["df_predicted"]["churn_predicted_probability"][0]) * 100:.2f}%')

with tab6:
    st.write('Пока нет данных')

# Выводим результаты предсказания для клинтов из файла (вкладка 1)
if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab1':
    # Результаты предсказания для всех клиентов в файле
    st.subheader('Результаты прогнозирования')
    st.write(st.session_state['df_predicted'])
    # Скачиваем результаты предсказания для всех клиентов в файле
    res_all_csv = convert_df(st.session_state['df_predicted'])
    st.download_button(
        label="Скачать все предсказания",
        data=res_all_csv,
        file_name='df-churn-predicted-all.csv',
        mime='text/csv',
    )

    # Гистограмма оттока для всех клиентов в файле
    fig = px.histogram(st.session_state['df_predicted'], x='churn_predicted', color='churn_predicted')
    st.plotly_chart(fig, use_container_width=True)

    # Клиенты с высоким риском оттока
    risk_clients = st.session_state['df_predicted'][st.session_state['df_predicted']['churn_predicted'] == 1]
    # Выводим клиентов с высоким риском оттока
    if len(risk_clients) > 0:
        st.subheader('Клиенты с высоким риском оттока')
        st.write(risk_clients)
        # Скачиваем клиентов с высоким риском оттока
        res_risky_csv = convert_df(risk_clients)
        st.download_button(
            label="Скачать клиентов с высоким риском оттока",
            data=res_risky_csv,
            file_name='df-churn-predicted-risk-clients.csv',
            mime='text/csv',
        )