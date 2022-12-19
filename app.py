import streamlit as st
from PIL import Image
from PIL import ImageOps
from PIL import UnidentifiedImageError
import base64
import os
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import numpy as np
import pandas as pd
import pickle
import cv2 as cv
import joblib

model = joblib.load('model/model.sav')
def vectorize(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    classes = model.predict(descriptors)
    hist, _ = np.histogram(classes, np.arange(513))
    return hist / hist.sum()


def db_create(folder_dir):
    vectors, links = [], []
    for image in os.listdir(folder_dir):
        if image.endswith(".jpg"):
            vectors.append(base64.b64encode((vectorize(cv.imread('images/voc/' + image)))))
            links.append(image)
    return pd.DataFrame({"embedding": vectors, "link": links})


def get_k_neighbours(vector, df, number_of_neighbours):
    neigh = NearestNeighbors(n_neighbors=number_of_neighbours, metric=lambda a, b: distance.cosine(a, b))
    neigh.fit(df['embedding'].to_numpy().tolist())
    return neigh.kneighbors([vector], number_of_neighbours, return_distance=False)


def get_neighbours_links(df, neighbors):
    similar = df.iloc[neighbors[0]]
    return similar['link'].to_numpy().tolist()


st.set_option('deprecation.showfileUploaderEncoding', False)
db = pd.read_csv('output1.csv', delimiter=',')
db['embedding'] = db['embedding'].apply(lambda x: np.frombuffer(base64.b64decode(bytes(x[2:-1], encoding='ascii')), dtype=np.int32))

def main():
    st.header("Download image")
    uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        pass
    else:
        image = Image.open(uploaded_file).convert("RGB")
        image = ImageOps.exif_transpose(image)
        st.image(image)
        img_opencv = np.array(image)
        img_opencv = img_opencv[:, :, ::-1].copy()
        links = get_neighbours_links(db, get_k_neighbours(vectorize(img_opencv), db, 3))
        st.success("Similar images from the dataset:")
        col = st.columns(3)
        for i in range(len(links)):
            try:
                with col[i]:
                    similar_image = Image.open('images/voc/' + links[i])
                    st.image(similar_image, width=200)
            except UnidentifiedImageError:
                pass


if __name__ == '__main__':
    main()
