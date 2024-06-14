import streamlit as st
from link import *
import pickle
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import cosine_similarity
import json
from langchain_google_genai import ChatGoogleGenerativeAI

@st.cache_resource
def load_models():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("MODEL TRANSFORMER: all-MiniLM-L6-v2 ------------> READY")
    sentence_embed_model = SentenceTransformer("firqaaa/indo-sentence-bert-base")
    print("MODEL TRANSFORMER: indo-sentence-bert-base ------------> READY")
    return model, sentence_embed_model


def predict_major(input_text: str, model_path="model_lstm_bert.h5", embed_model_name=None, top_n=1):
    with open('embedding_dataset.pkl', 'rb') as f:
        _, _, label_encoder = pickle.load(f)
    
    model = tf.keras.models.load_model(model_path)

    # Embed the input text
    embedded_text = embed_model_name.encode([input_text])
    
    # Reshape embedded_text to match the model's input shape
    embedded_text = np.reshape(embedded_text, (1, 1, -1))
    
    # Predict the probabilities
    pred = model.predict(embedded_text)[0]
    
    # Get the indices of the top_n predictions
    top_indices = pred.argsort()[-top_n:][::-1]
    
    # Decode the top_n labels
    labels = label_encoder.inverse_transform(top_indices)
    
    return labels[0]


similarity_vector_data = {
    "Teknik Informatika": "data_meta/informatika_embed.json",
    "Manajemen Bisnis": "data_meta/bisnis_embed.json",
    "Desain Grafis": "data_meta/desain_embed.json"
}

get_csv_data = {
    "Teknik Informatika": "data/Freelance_Teknik_Informatika_Data.csv",
    "Manajemen Bisnis": "data/Freelance_Manajemen_Bisnis_Data.csv",
    "Desain Grafis": "data/Freelance_Desain_Grafis_Data.csv"
}
def recommendation_freelance(predict, user_input, model):
    # Load the corresponding data
    with open(similarity_vector_data[predict], "r") as f:
        data = json.load(f)
    
    # Ensure data is a 2D array
    data = np.array(data)
    
    # Translate and encode the user input
    user_input_translated = GoogleTranslator(target='en').translate(user_input)
    user_embedding = model.encode([user_input_translated])
    
    # Ensure the user embedding is 2D
    user_embedding = np.array(user_embedding).reshape(1, -1)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_embedding, data)
    
    # Find the index of the most similar project
    top_3_indices = np.argsort(similarities[0])[-3:][::-1]
    
    df = pd.read_csv(get_csv_data[predict])
    df_recommendation = df.loc[top_3_indices]
    
    return df_recommendation


def continue_convertation(predict_recoment, last_three_convertation):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyA1vSebG1XsacRyaTP4gilYBzDSvhxz0gs")
    llm = llm.invoke(f"""
[CONTEXT]
Rekomendasi Projek Freelance:
PROJEK 1:
{str(predict_recoment.iloc[0]["title"])}
{str(predict_recoment.iloc[0]["price"])}
{str(predict_recoment.iloc[0]["description"])}
{str(predict_recoment.iloc[0]["link"])}

PROJEK 2:
{str(predict_recoment.iloc[1]["title"])}
{str(predict_recoment.iloc[1]["price"])}
{str(predict_recoment.iloc[1]["description"])}
{str(predict_recoment.iloc[1]["link"])}
[/CONTEXT]

==============================

{last_three_convertation}

==============================

Jawab pertanyaan lanjutan dari user, namun referensi atau patokan dari CONTEXT. Jika diluar CONTEXT bilang saja 'saya hanya menjawab berdasarkan hasil rekomendasi, tekan `new chat` untuk rekomendasi baru'.
Namun jika masih berhubungan dengan CONTEXT misal mengenai teknologinya, cara pembuatan, pengertian-pengertian, dll. Jawab dengan kreatif dan singkat saja.
""")
    
    st.session_state.messages.append({"role": "assistant", "content": llm.content})
    
    return llm.content

def recommendation():
    model, sentence_embed_model = load_models()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    intro = "Hallo saya bisa merekomendasikan proyek freelance untuk mu, namun sebelum itu bisa kamu ceritakan kepadaku terkait keahlian dan skill yang kamu miliki?"
    if not any(msg["content"] == intro for msg in st.session_state.messages):
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.chat_message("assistant").markdown(intro)
        
    prompt = st.chat_input("Ketik disini!")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        
        if len(st.session_state.messages) == 2:
            message_role = "Menarik! saya akan merekomendasikan projek freelance yang mungkin cocok denganmu!"
            st.session_state.messages.append({"role": "assistant", "content": message_role})
            st.chat_message("assistant").markdown(message_role)
                
            with st.spinner("Mohon tunggu..."):
                predict = predict_major(prompt, model_path="model_lstm_bert.h5", embed_model_name=sentence_embed_model)
                global recomment
                recomment = recommendation_freelance(predict, prompt, model)
                
                link_full_predict = {
                    "Teknik Informatika": link_teknik_informatika,
                    "Manajemen Bisnis": link_manajemen_bisnis,
                    "Desain Grafis": link_desain_grafis
                }
                
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyA1vSebG1XsacRyaTP4gilYBzDSvhxz0gs")
                llm = llm.invoke(f"""
[CONTEXT]
Rekomendasi Projek Freelance:
PROJEK 1:
{str(recomment.iloc[0]["title"])}
{str(recomment.iloc[0]["price"])}
{str(recomment.iloc[0]["description"])}
{str(recomment.iloc[0]["link"])}

PROJEK 2:
{str(recomment.iloc[1]["title"])}
{str(recomment.iloc[1]["price"])}
{str(recomment.iloc[1]["description"])}
{str(recomment.iloc[1]["link"])}
[CONTEXT]

==============================

POV: Anda seorang yang merekomendasikan Projek Freelance.

Tolong rekomendasikan projek freelance kepada user berdasarkan CONTEXT di atas, beri tau kalau dia mungkin cocok dengan rekomendasi yang diberikan
berikan dalam bentuk list, lampirkan Judul dan link untuk menuju projek freelance tersebut.

Berikut format jawaban yang perlu di tuliskan
1. judul rekomendasi pertama
harga projek,
deskripsi singkat,
jelaskan kenapa direkomendasikan itu,
link.

dan seterusnya untuk yang kedua sama! tidak perlu ada kesimpulan! dan buat formatnya terlihat rapih dan profesional!

Beri tahu user jika ingin tau lebih terkait 2 projek di atas bisa lanjut chat, dan jika ingin mendapat rekomendasi lainnya bisa klik `new chat` untuk memulai dari awal.

Dan di akhir tolong berikan link berikut {link_full_predict[predict]} untuk memberi tahu jika ingin melihat projek relevan lainnya bisa mengakses link tersebut.
""")
                
                st.session_state.messages.append({"role": "assistant", "content": llm.content})
                st.chat_message("assistant").markdown(llm.content)
                
                
        elif len(st.session_state.messages) >= 4:
            # Mengambil 3 data terbaru dari session state messages
            last_three_messages = st.session_state.messages[-3:]

            # Mengkonversi data tersebut ke dalam format string JSON
            json_string = json.dumps(last_three_messages)
            
            st.chat_message("assistant").markdown(continue_convertation(recomment, json_string))