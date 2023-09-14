import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input


model = tf.keras.models.load_model('./food_recognition_model2.h5')
label_to_int = {'갈비찜': 0, '감자볶음': 1, '고구마': 2, '고추 장아찌': 3, '국수': 4, '단호박찜': 5, '달걀찜': 6, '달걀후라이': 7, '닭갈비': 8, '된장국': 9, '된장찌개': 10, '떡국': 11, '만두국': 12, '무김치': 13, '미역국': 14, '미역나물': 15, '바나나': 16, '밥': 17, '방울토마토': 18, '배추김치': 19, '부침개': 20, '브로콜리': 21, '비엔나 소시지': 22, '빵': 23, '사과': 24, '삶은 달걀': 25, '삼겹살구이': 26, '상추': 27, '샌드위치': 28, '생선구이': 29, '순두부찌개': 30, '스크램블 에그': 31, '야채볶음': 32, '양배추쌈': 33, '양상추샐러드': 34, '어묵국': 35, '오리구이': 36, '오믈렛': 37, '오이': 38, '우유': 39, '잡채': 40, '조미김': 41, '죽': 42, '찜닭': 43, '참외': 44, '치킨너겟': 45, '콩나물 무침': 46, '토마토': 47, '파프리카': 48, '포도': 49, '해물볶음': 50, '훈제연어': 51}


def predict_food_category(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    img = preprocess_input(img)

    prediction = model.predict(img)[0]
    predicted_class_index = np.argmax(prediction)

    for key, value in label_to_int.items():
        if value == predicted_class_index:
            return key
        else:
            pass


def predict_food():
    image_dir = './crop/'
    src = []
    file_name = []
    result = []
    for file in os.listdir(image_dir):
        src.append(image_dir + file)
        file_name.append(file)
        result.append(predict_food_category(image_dir + file))
    
    for i in range(len(result)):
        print(file_name[i] + " : , Predict : "+ str(result[i]))
    # 중복값 제거
    food_name = set(result)
    
    return food_name
