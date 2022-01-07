import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


st.title("Cellphone Price Range Prediction")
st.caption("Made by. **Amar Ma'ruf** (Student of Hacktiv8 Data Science Fulltime Program)")

st.markdown("This is data that we used for training")

df = pd.read_csv('train.csv')
df_copied = df.copy()

df_copied['price_range'] = df_copied['price_range'].map({0: '< 1 jt', 1: '< 4 jt ', 2: '< 10 jt', 3: '>= 10 jt'})
# st.caption("Data that we used for training")
st.write(df_copied.head())

col1, col2, col3 = st.columns(3)
with col1: 
    st.markdown("External Specifications")
    battery = st.slider('Battery Capacity', 500, 2000, 500 )
    weight = st.slider("Weight", 80, 200, 80)
    sc_h = st.slider('Screen Height (cm)', 5, 19, 5)
    sc_w = st.slider('Screen Width (cm)', 0, sc_h, 0)
    px_w = st.slider('Pixel Width (pixel)', 500, 2000, 500)
    px_h = st.slider('Pixel Height (pixel)', 0, px_w, 0)
    depth = st.selectbox('Mobile Depth (cm)', (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0., 0.9, 1.0))

with col2:
    st.markdown("Internal Specifications")
    clock = st.slider('Clock Speed', 0.5, 3.0, 0.5, 0.1 )
    fc = st.slider('Front Camera', 0,20,0)
    pc = st.slider('Primary Camera', 0,20,0)
    memory = st.slider('Internal Memmory', 2, 64, 2, 2)
    ram = st.slider('Ram ', 256.0, 4000.0, 256.0,0.5)
    talk_time = st.slider('Battery life when call', 5, 20, 5)
    core = st.selectbox("Cores of Processor", (1, 2, 3, 4, 5, 6, 7, 8))

with col3:
    st.markdown("Support Feature")
    blue = st.radio('Bluetooth', ('Support', 'Not Support'))
    dual = st.radio('Dual Sim', ('Support', 'Not Support'))
    g4 = st.radio('4G', ('Support', 'Not Support'),)
    g3 = st.radio('3G', ('Support', 'Not Support'))
    touch = st.radio('Touch Screen', ('Support', 'Not Support'))
    wifi = st.radio('WiFi', ('Support', 'Not Support'))


new_data = {}
new_data['battery_power'] = battery

if blue == 'Support':
    new_data['blue'] = [1]
else:
    new_data['blue'] = [0]

new_data['clock_speed'] = clock

if dual == 'Support':
    new_data['dual_sim'] = [1]
else:
    new_data['dual_sim'] = [0]

new_data['fc'] = fc

if g4 == 'Support':
    new_data['four_g'] = [1]
else:
    new_data['four_g'] = [0]

new_data['int_memory'] = memory
new_data['m_dep'] = depth
new_data['mobile_wt'] = weight
new_data['n_cores'] = core
new_data['pc'] = pc
new_data['px_height'] = px_h
new_data['px_width'] = px_w
new_data['ram'] = ram
new_data['sc_h'] = sc_h
new_data['sc_w'] = sc_w
new_data['talk_time'] = talk_time

if g3 == 'Support':
    new_data['three_g'] = [1]
else:
    new_data['three_g'] = [0]

if touch == 'Support':
    new_data['touch_screen'] = [1]
else:
    new_data['touch_screen'] = [0]

if wifi == 'Support':
    new_data['wifi'] = [1]
else:
    new_data['wifi'] = [0]


# st.write(new_data)

new_df = pd.DataFrame(data=new_data)
new_df['dimension'] = np.sqrt((new_df['sc_h']**2) + (new_df['sc_w']**2)).astype(int)
new_df.drop(columns=['sc_h', 'sc_w'], inplace=True)

st.markdown("Is this your cellphone specifications?")
st.write(new_df)

## Modeling
X = df.drop(columns='price_range')
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10)

#DChange colom sc_h & sc_w into dimension feature
X_train['dimension'] = np.sqrt((X_train['sc_h']**2) + (X_train['sc_w']**2)).astype(int)
X_train.drop(columns=['sc_h', 'sc_w'], inplace=True)

#Model Definitiom using pipeline
pipe_svc = make_pipeline(PCA(n_components=17), StandardScaler(), SVC(C= 100, gamma=0.001))
pipe_svc.fit(X_train, y_train)

predict = pipe_svc.predict(new_df)

if predict == 0:
    prediction = '< 1 jt'
elif predict == 1:
    prediction = '< 4 jt'
elif predict == 2:
    prediction = '< 10 jt'
else:
    prediction = '>= 10 jt'
df_copied['price_range'] = df_copied['price_range'].map({0: '< 1 jt', 1: '< 4 jt ', 2: '< 10 jt', 3: '>= 10 jt'})
# new_df['Prediction'] = predict
# df.loc[len[df]]= new_df
# df_copied['class'] = df_copied['price_range']
button = st.button('Proceed!')

if button:
    st.write("Your cellphone price is in range of ", prediction)

# plt.figure(figsize=(12,5))
# sns.scatterplot(data = df, x= 'battery_power', y='ram', s=50, hue='price_range', alpha=0.1)
# sns.scatterplot(data = new_df, x= 'battery_power', y='ram', s=100, hue=predict, color='red')
# sns.scatterplot(data = new_df.iloc[-1, :], x= 'battery_power', y='ram', s=50, hue=predict, color='red')
# st.pyplot()