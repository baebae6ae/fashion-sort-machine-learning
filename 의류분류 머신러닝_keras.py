#앞에서 만들었던거 사용함
import numpy as np
import matplotlib.pyplot as plt

#의류 이미지 데이터 세트 불러오기
import keras
import tensorflow as tf
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(x_train_all.shape, y_train_all.shape)

x_show = x_test

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)

x_train = x_train/255
x_val = x_val/255 #이미지는 픽셀마다 0~255 사이의 값을 가지므로 255로 나누면 0~1 사이로 표준화한 것임.

x_train = x_train.reshape(-1, 784)
x_val = x_val.reshape(-1, 784) #이미지가 28x28의 이차원 배열이라 1차원으로 바꾸고 784 길이로 펼치는 것임 (784=28x28)
x_test = x_test.reshape(-1, 784)
print(x_test.shape)


y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val) #타깃 데이터가 0~9 사이의 정수로 확인되기 때문에 원-핫 인코딩으로 10개의 원소를 가진 배열로 바꿔준다 6 = [0.0.0.0.0.0.1.0.0.0.]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_train, y_train)

print('테스트 정확도 : ', test_acc)

plt.imshow(x_show[30],cmap='gray')
plt.show

predictions = model.predict(x_test)
l=np.argmax(predictions[30]) # 모델의 예측 레이블 확인
class_names = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌들', '셔츠', '스니커즈', '가방', '앵클부츠']
print(class_names[l])
