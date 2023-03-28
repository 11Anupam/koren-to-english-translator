# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(
    page_title="쿼카고", layout="wide", initial_sidebar_state="expanded"
)

@st.cache
def load_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model

tokenizer = AutoTokenizer.from_pretrained("QuoQA-NLP/KE-T5-Ko2En-Base")
ko2en_model = load_model("QuoQA-NLP/KE-T5-Ko2En-Base")
en2ko_model = load_model("QuoQA-NLP/KE-T5-En2Ko-Base")


st.title("🐻 쿼카고 번역기")
st.write("좌측에 번역 모드를 선택하고, CTRL+Enter(CMD+Enter)를 누르세요 🤗")
st.write("Select Translation Mode at the left and press CTRL+Enter(CMD+Enter)🤗")

translation_list = ["한국어에서 영어 | Korean to English", "영어에서 한국어 | English to Korean"]
translation_mode = st.sidebar.radio("번역 모드를 선택(Translation Mode):", translation_list)


default_value = '신한카드 관계자는 "과거 내놓은 상품의 경우 출시 2개월 만에 적금 가입이 4만여 좌에 달할 정도로 인기를 끌었다"면서 "금리 인상에 따라 적금 금리를 더 올려 많은 고객이 몰릴 것으로 예상하고 있다"고 말했다.'
src_text = st.text_area(
    "번역하고 싶은 문장을 입력하세요:",
    default_value,
    height=300,
    max_chars=200,
)
print(src_text)



if src_text == "":
    st.warning("Please **enter text** for translation")
else: 
    # translate into english sentence
    if translation_mode == translation_list[0]:
        model = ko2en_model
    else: 
        model = en2ko_model

    translation_result = model.generate(
        **tokenizer(
            src_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        ),
        max_length=64,
        num_beams=5,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
    )
    translation_result = tokenizer.decode(
        translation_result[0],
        clean_up_tokenization_spaces=True,
        skip_special_tokens=True,
    )

    print(f"{src_text} -> {translation_result}")

    st.write(translation_result)
    print(translation_result)
