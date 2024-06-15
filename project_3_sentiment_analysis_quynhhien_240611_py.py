import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from pyspark.sql import SparkSession
import joblib
import base64

# Hàm để tải hình ảnh và trả về mã base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
# Đường dẫn đến hình ảnh của bạn
img_path = 'background1.jpg'
img_base64 = get_base64_of_bin_file(img_path)
# Thiết lập tiêu đề và subheader
# Sử dụng CSS để thay đổi màu tiêu đề
st.markdown(
    """
    <style>
    .title {
        color: #FF6347;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    
# Sử dụng Markdown để thiết lập tiêu đề với lớp CSS
st.markdown('<h1 class="title">SENTIMENT ANALYSIS</h1>', unsafe_allow_html=True)
st.subheader("""
             PROJECT 3: SHOPEE FOOD
             Quốc Khánh - Quỳnh Hiên
             """)
st.write("___________________________")
# Thiết lập hình nền bằng CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Hiển thị thanh menu ngang

menu = [
    "Project Purpose",
    "About Dataset",
    "Analysis Showcase",
    "Prediction Tool"
]
#icons=[
    #"info-circle", 
    #"database", 
    #"bar-chart", 
    #"tools"]
choice = st.radio("Menu", menu, horizontal=True)
st.write("___________________________")

# Project Purpose

if choice == 'Project Purpose':
    st.subheader("Sentiment Analysis for Businesses")
    
    # Đoạn văn 1: Giới thiệu Sentiment Analysis
    st.write("""
    **Giới thiệu Sentiment Analysis:**
    
    Sentiment Analysis (Phân tích cảm xúc) là quá trình sử dụng ngôn ngữ tự nhiên, thống kê học và học máy để phân tích và phát hiện cảm xúc của con người thông qua các bình luận, đánh giá hoặc ý kiến. Công nghệ này giúp doanh nghiệp hiểu rõ hơn về phản hồi của khách hàng, từ đó cải thiện sản phẩm, dịch vụ và tăng cường mối quan hệ với khách hàng.
    """)

    # Đoạn văn 2: Lợi ích cho Doanh nghiệp
    st.write("""
    **Lợi ích cho Doanh nghiệp:**

    - **Hiểu rõ tâm lý khách hàng:** Nhận diện cảm xúc tích cực, tiêu cực hoặc trung lập trong các bình luận.
    - **Nâng cao chất lượng dịch vụ:** Dựa trên phản hồi để cải thiện các điểm yếu.
    - **Tăng cường chiến lược tiếp thị:** Phân tích cảm xúc giúp điều chỉnh chiến lược tiếp thị phù hợp với khách hàng.
    - **Giám sát thương hiệu:** Theo dõi và phản hồi kịp thời các đánh giá tiêu cực về thương hiệu.
    """)

    # Đoạn văn 3: Hình ảnh hoặc Đồ thị
    st.write("""
    **Hình ảnh hoặc Đồ thị:**
    
    - Infographic về lợi ích của Sentiment Analysis.
    - Biểu đồ tròn minh họa tỷ lệ các loại cảm xúc (tích cực, tiêu cực, trung lập) mà doanh nghiệp có thể phân tích.
    """)
# About Dataset

elif choice == 'About Dataset':
    st.subheader("Understanding Our Data")
    st.write("""
    - **Sentiment Analysis (Phân tích cảm xúc)** là quá trình sử dụng ngôn ngữ tự nhiên, thống kê học và học máy để phân tích và phát hiện cảm xúc của con người thông qua các bình luận, đánh giá hoặc ý kiến.
	- Công nghệ này giúp doanh nghiệp hiểu rõ hơn về phản hồi của khách hàng, từ đó cải thiện sản phẩm, dịch vụ và tăng cường mối quan hệ với khách hàng.
	""")
    
    st.write("""
    **Lợi ích cho Doanh nghiệp:**
    
    - **Hiểu rõ tâm lý khách hàng:** Nhận diện cảm xúc tích cực, tiêu cực hoặc trung lập trong các bình luận.
    - **Nâng cao chất lượng dịch vụ:** Dựa trên phản hồi để cải thiện các điểm yếu.
    - **Tăng cường chiến lược tiếp thị:** Phân tích cảm xúc giúp điều chỉnh chiến lược tiếp thị phù hợp với khách hàng.
    - **Giám sát thương hiệu:** Theo dõi và phản hồi kịp thời các đánh giá tiêu cực về thương hiệu.
""")
    
    st.write("""
    **Hình ảnh hoặc Đồ thị:**
    - Infographic về lợi ích của Sentiment Analysis.
    - Biểu đồ tròn minh họa tỷ lệ các loại cảm xúc (tích cực, tiêu cực, trung lập) mà doanh nghiệp có thể phân tích.
""") 
    st.write("**Reviews Dataset**")
    st.image("Review_Dataset.png")
    st.write("**Restaurant Dataset**")
    st.image("Restaurant_Dataset.png")
    st.write("**Merged Dataset**")
    st.image("Merged_Dataset.png")
    st.write("**Location distribution of restaurant**")
    st.image("Location_Distribution_Of_Restaurant.png")
    st.image("Location_Distribution_In_Pie.png")
    st.write("**Price Distribution by District**")
    st.image("Price_Distribution_by_District.png")
    st.write("**Biểu đồ phân phối comment theo thời gian**")
    st.image("Comment_Distribution_by_Time.png")
    st.write("**Dine_In_Delivery_Ratio In Pie**")
    st.image("Dine_In_On_Delivery.png")
    st.write("**Biểu đồ Word Cloud của Processed Comment**")
    st.image("Preprocessed_Comment_Wordcloud.png")
    st.write("Biểu đồ phân phối cảm xúc của comment")
    st.image("Overall_Sentiment_Distribution.png")

#Analysis Showcase

elif choice == 'Analysis Showcase':
    st.write("""
    **Insights from Consumer Feedback**
    **Nội dung:**
""")
    st.write("""
    - **Demo mô hình:**
    
    - Mô hình phân tích cảm xúc của chúng tôi sử tự luật, dụng học máy và dữ liệu lớn để phân tích các bình luận và đánh giá của khách hàng.
    - Mô hình có thể phát hiện các khía cạnh khác nhau của sản phẩm hoặc dịch vụ (như chất lượng, giá cả, dịch vụ, vị trí và không gian) và đánh giá cảm xúc tương ứng.
""")
    st.write("""
    - **Các biểu đồ và Insight:**
    
    - Biểu đồ tròn hiển thị tỷ lệ cảm xúc chung của khách hàng (tích cực, tiêu cực, trung lập).
	- Biểu đồ thanh hiển thị số lượng bình luận tích cực và tiêu cực cho từng khía cạnh của sản phẩm hoặc dịch vụ.
	- Word cloud hiển thị các từ khóa phổ biến nhất trong các bình luận tích cực và tiêu cực.
	- Biểu đồ đường theo dõi xu hướng cảm xúc của khách hàng theo thời gian.
 """)
    st.write("""
    - **Hình ảnh hoặc Đồ thị:**
    - Word Cloud về các từ khóa tích cực và tiêu cực.
	- Biểu đồ tròn về tỷ lệ cảm xúc (Positive, Negative, Neutral).
	- Biểu đồ thanh về cảm xúc theo từng khía cạnh (FOOD, PRICE, SERVICE, LOCATION, AMBIENCE).
   """)
    st.write("**Kết quả mô hình Machine Learning**")
    st.image("Machine_Learning_Model_Results.png")
    st.write("**Kết quả mô hình PySpark Big Data**")
    st.image("PySpark_Big_Data_Result.png")
    st.write("**Biều đồ phân phối Sentiment của Comment**")
    st.image("Sentiment_Distribution.png")
    st.write("**Biểu đồ Word Cloud của comment sau khi đã thực hiện xử lý ngôn ngữ**")
    st.image("Preprocessed_Comment_Wordcloud.png")
    
    st.subheader("**Báo cáo theo RestaurantID 1**")
    
    st.write("**Biểu đồ phân phối cảm xúc theo RestaurantID 1**")
    st.image("Sentiment_Distribution_for_RestaurantID_1.png")
    st.write("**Biểu đồ phân phối cảm xúc theo RestaurantID 1 in Pie**")
    st.image("Sentiment_Distribution_for_RestaurantID_1_Pie.png")
    st.write("**Word Cloud của RestaurantID 1**")
    st.image("Wordcloud_Comment_for_RestaurantID_1.png")
    st.write("**Positive Word Cloud của RestaurantID 1**")
    st.image("Positive_Wordcloud_for_RestaurantID_1.png")
    st.write("**Negative Word Cloud của RestaurantID 1**")
    st.image("Negative_Wordcloud_for_RestaurantID_1.png")
    st.write("**Dine In/Delivery Ratio của RestaurantID 1**")
    st.image("Dine_In_Delivery_Ratio_forRestaurantID_1.png")
    st.write("**Báo cáo tổng hợp theo RestaurantID 1**")
    st.image("Bao_cao_tong_hop.png")

# Prediction Tool

elif choice == 'Prediction Tool':
    st.title("Prediction Tool")

    st.write("""
    - **Hình ảnh hoặc Đồ thị:**
    - Word Cloud về các từ khóa tích cực và tiêu cực.
    - Biểu đồ tròn về tỷ lệ cảm xúc (Positive, Negative, Neutral).
    - Biểu đồ thanh về cảm xúc theo từng khía cạnh (FOOD, PRICE, SERVICE, LOCATION, AMBIENCE).
""")

    tab1, tab2 = st.tabs(["Customer", "Restaurant owner"])
    with tab1:

    # Hàm xử lý ngôn ngữ
    # Tải teencode
        def load_teencode(file_path):
            with open(file_path, 'r', encoding="utf8") as file:
                teen_lst = file.read().split('\n')
            teen_dict = {}
            for line in teen_lst:
                if '\t' in line:  # Check if the delimiter is present
                    key, value = line.split('\t')
                    teen_dict[key] = str(value)
                # Handle lines without the delimiter (optional):
                # else:
                #     print(f"Skipping line: {line}")
            return teen_dict

        def load_emoji(file_path):
            with open(file_path, 'r', encoding="utf8") as file:
                emoji_lst = file.read().split('\n')
            emoji_dict = {}
            for line in emoji_dict:
                if '\t' in line:  # Check if the delimiter is present
                    key, value = line.split('\t')
                    emoji_dict[key] = str(value)
                # Handle lines without the delimiter (optional):
                # else:
                #     print(f"Skipping line: {line}")
            return emoji_dict

        # Tải từ điển Anh-Việt
        def load_english_vnmese(file_path):
            with open(file_path, 'r', encoding="utf8") as file:
                english_lst = file.read().split('\n')
            english_dict = {}
            for line in english_lst:
                key, value = line.split('\t')
                english_dict[key] = str(value)
            return english_dict

        # Tải wrong words
        def load_wrong_words(file_path):
            with open(file_path, 'r', encoding="utf8") as file:
                wrong_lst = file.read().split('\n')
            return wrong_lst

        # Hàm xử lý teencode và từ sai
        def fix_teencode_and_wrong_words(text, teen_dict):
            for key, value in teen_dict.items():
                text = re.sub(r'\b{}\b'.format(re.escape(key)), value, text)
            return text

        # Chuẩn hóa unicode tiếng việt
        def loaddicchar():
            uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
            unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

            dic = {}
            char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
                '|')
            charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
                '|')
            for i in range(len(char1252)):
                dic[char1252[i]] = charutf8[i]
            return dic

        # Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
        def covert_unicode(txt):
            dicchar = loaddicchar()
            return re.sub(
                r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
                lambda x: dicchar[x.group()], txt)

        # Tải dữ liệu
        teen_dict = load_teencode('teencode.txt')
        emoji_dict = load_emoji('emojicon.txt')
        english_dict = load_english_vnmese('english-vnmese.txt')
        wrong_lst = load_wrong_words('wrong-word.txt')

        # Hàm tiền xử lý văn bản
        def preprocess_text(text, teen_dict, wrong_lst, english_dict, emoji_dict):
            text = covert_unicode(text)
            text = text.lower()
            text = fix_teencode_and_wrong_words(text, teen_dict)

            # Chỉ giữ lại các từ tiếng Việt và tiếng Anh
            pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
            text = ' '.join(re.findall(pattern, text))

            # Dịch từ tiếng Anh sang tiếng Việt
            words = text.split()
            words = [english_dict[word] if word in english_dict else word for word in words]
            text = ' '.join(words)

            # CONVERT EMOJICON
            text = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(text))
            
            # Loại bỏ các từ sai
            words = text.split()
            words = [word for word in words if word not in wrong_lst]

            return ' '.join(words)



        # Tạo từ điển từ viết tắt
        abbreviation_dict = {
            "mk": "mình",
            "ko": "không",
            "thik": "thích",
            "lm": "làm",
            "j": "gì",
            "ca": "cả",
            "tưk": "tức",
            "khum": "không",
            "thk": "thích",
            "k": "không",
            "qtqđ": "quá trời quá đất",
            "vs": "với",
            "th": "thì",
            "kh": "không",
            "v": "với",
            # Thêm các từ viết tắt khác vào từ điển
        }

        # Hàm mở rộng từ viết tắt
        def expand_abbreviations(text, abbreviation_dict):
            words = text.split()
            expanded_words = [abbreviation_dict.get(word.lower(), word) for word in words]
            return ' '.join(expanded_words)

        # Tạo danh sách tùy chỉnh cho các cụm từ phổ biến
        positive_list = [
            "ấn tượng", "ấm áp", "ấm cúng", "bắt mắt", "cảm động", "chất lượng", "chấp nhận được", "chuyên nghiệp", "đồng ý", "chúc mừng",
            "dễ dàng", "dễ thương", "dễ tìm", "đáng giá", "đáng tin cậy", "đáng tiền", "đẹp", "dịch vụ tốt", "tuyệt", "nhanh nhẹn", "điểm 10",
            "đơn giản", "đúng giờ", "dễ ăn", "giá rẻ", "giòn", "giòn rụm", "giòn tan", "gọn gàng", "gây ấn tượng", "sẽ quay lại", "chuẩn",
            "hài lòng", "hợp khẩu vị", "hợp lý", "hoà nhã", "hoàn hảo", "nhanh", "mát mẻ", "mới", "ăn ngon", "ngon lắm", "ưng", "ưng ý",
            "nóng hổi", "nổi trội", "nức mũi", "phong cách", "phục vụ tốt", "phục vụ tận tâm", "rất ngon", "rất tốt", "rẻ", "rẻ hơn", "mê",
            "rẻ nhất", "rộng rãi", "sạch sẽ", "sang trọng", "thành thạo", "thân thiện", "thích", "thoải mái", "tốt", "xịn", "ngon", "ngon lành",
            "thuận tiện", "tươi", "tươi mới", "tuyệt vời", "tận tâm", "vừa vặn", "vui vẻ", "xuất sắc", "giảm", "sẽ ủng hộ", "ủng hộ", "vừa ý",
            "mình thích", "cười", "yêu"
        ]

        negative_list = [
            "ăn không ngon","bẩn", "bẩn thỉu", "chật", "chật chội", "chán", "chậm", "chậm chạp", "chưa đẹp", "chưa chín", "dơ", "dơ dáy",
            "dụng cụ dơ", "giá cao", "giá đắt", "hôi thối", "không đáng giá", "không đáng tiền", "không hài lòng", "dẹp", "dẹp luôn", "bỏ ý định",
            "không tốt", "kém", "kém chất lượng", "kém vệ sinh", "lười", "lừa đảo", "mắc", "ngột ngạt", "nguyên vật liệu cũ", "khóc ròng",
            "nguội", "phản hồi chậm", "rất dỡ", "rối rắm", "tanh", "thất vọng", "thiếu", "tối", "tối tăm", "tồi tệ", "sợ luôn", "huỷ",
            "tệ", "thật tệ", "tức giận", "yếu", "không thoải mái", "khó chịu", "không ngon", "món ăn thật tệ", "không ngon bằng", "mệt mỏi",
            "dở", "rất dơ", "nấu còn ngon hơn", "rất ngọt", "ăn kiểu gì", "không quay lại", "rất dở", "bỏ", "bỏ luôn", "không thích", "sợ",
            "không vui", "bùn", "giận dữ", "tức", "tệ", "không thề", "lo lắng",
        ]

        # Từ khóa cho từng khía cạnh
        aspect_keywords = {
            'FOOD': ['món ăn', 'thức ăn', 'đồ ăn', 'ngon', 'không ngon', 'dở', 'tươi', 'mùi vị', 'thơm', 'kém', 'thiu', 'chất lượng'],
            'PRICE': ['chi phí', 'rẻ', 'đắt', 'giảm', 'miễn phí', 'tặng', 'hời', 'không đáng', 'mắc', 'ủng hộ'],
            'SERVICE': ['phục vụ', 'nhân viên', 'tốt', 'thân thiện', 'chuyên nghiệp', 'nhiệt tình', 'chu đáo', 'lạnh nhạt', 'thô lỗ', 'không tốt', 'ủng hộ'],
            'LOCATION': ['địa điểm', 'vị trí', 'dễ tìm', 'gần', 'xa', 'thuận tiện', 'khó tìm', 'không tiện', 'yên tĩnh','ủng hộ'],
            'AMBIENCE': ['không gian', 'không khí', 'phong cách', 'thiết kế', 'trang trí', 'âm nhạc', 'ánh sáng', 'ngột ngạt', 'ồn ào', 'ủng hộ']
        }

        # Hàm sử dụng từ điển tùy chỉnh để xác định cảm xúc
        def custom_sentiment_analysis(text, positive_list, negative_list):
            text = text.lower()  # Chuyển thành chữ thường

            # Kiểm tra từ trong negative_list trước
            for phrase in negative_list:
                if phrase in text:
                    print(f"Matched negative phrase: '{phrase}'")
                    return 'negative'

            # Kiểm tra từ trong positive_list
            for phrase in positive_list:
                if phrase in text:
                    print(f"Matched positive phrase: '{phrase}'")
                    return 'positive'

            return 'neutral'

        # Hàm tách câu thành các vế nhỏ hơn
        def split_clauses(sentence):
            connectors = ['nhưng', 'và', 'tuy nhiên', 'mặc dù']
            clauses = [sentence]
            for connector in connectors:
                temp_clauses = []
                for clause in clauses:
                    parts = clause.split(connector)
                    temp_clauses.extend(parts)
                clauses = temp_clauses
            return clauses

        # Hàm xác định khía cạnh của một câu với từ điển tùy chỉnh
        def get_aspect_sentiment(sentence, positive_list, negative_list):
            sentence = sentence.lower()  # Chuyển thành chữ thường
            aspect_sentiments = {'FOOD': 'neutral', 'PRICE': 'neutral', 'SERVICE': 'neutral', 'LOCATION': 'neutral', 'AMBIENCE': 'neutral'}
            for aspect, keywords in aspect_keywords.items():
                for keyword in keywords:
                    if keyword in sentence:
                        sentiment = custom_sentiment_analysis(sentence, positive_list, negative_list)
                        print(f"Sentence: '{sentence}' -> Keyword '{keyword}' -> Aspect: '{aspect}' -> Sentiment: '{sentiment}'")
                        aspect_sentiments[aspect] = sentiment
            return aspect_sentiments

        # Tải danh sách stop words
        with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as file:
            stop_words = file.read().splitlines()

        stop_words.extend(['nhà hàng', 'quán ăn', 'tiệm', 'quán', 'cửa hàng', 'chổ', 'chổ này', 'quán này', 'ở đây', 'tiệm này', 'chỗ này',
                        'chỗ', 'cỏ', 'nhân viên', 'kết hợp', 'lắm', 'bị', 'đặc', 'chi nhánh', 'chủ', 'kêu', 'hơi', 'nè', 'gói', 'bịch',
                        'hộp', 'dĩa', 'tô', 'chén', 'đĩa', 'đến', 'ghé', 'qua', 'đi', 'ở', 'tới', 'có', 'lúc', 'đem', 'kiểu',
                        ])
        stop_words = set(stop_words)

        # Hàm nối từ 'không'
        def process_special_word(text):
            new_text = ''
            text_lst = text.split()
            i = 0
            while i < len(text_lst):
                word = text_lst[i]
                if word in ['không', 'chẳng', 'chả', 'éo', 'đéo']:
                    next_idx = i + 1
                    if next_idx < len(text_lst):
                        word = word + '_' + text_lst[next_idx]
                        i = next_idx
                new_text += word + ' '
                i += 1
            return new_text.strip()

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=1005)


        # Đường dẫn tới mô hình đã lưu
        MODEL_PATH = 'svm_model.joblib'

        # Load SVM model
        loaded_SVM = joblib.load('svm_model.joblib')

        # Giao diện Khách Hàng
        st.title("Sentiment Analysis Tool")

        st.header("Khách hàng: Dự đoán sentiment của bình luận")
        comment = st.text_area("Nhập vào câu bình luận của bạn:")
        if st.button("Predict"):
            if comment:
                processed_comment = preprocess_text(comment, teen_dict, wrong_lst, english_dict, emoji_dict)  
                comment_vector = vectorizer.fit_transform([processed_comment])
                vector_size = comment_vector.shape[1]
                new_vector = np.zeros((1,1005))
                for i in range(vector_size):
                    new_vector[0,i] = comment_vector[0,i]
                
                sentiment = loaded_SVM.predict(new_vector)
                sentiment_label = ("Tích cực" if sentiment == 2 else "Trung lập" if sentiment == 1 else "Tiêu cực")
                st.write(f"Sentiment của bình luận là: {sentiment_label}")
            else:
                st.write("Vui lòng nhập bình luận để dự đoán.")

    with tab2:

    # Đọc dữ liệu từ file CSV
        result_df = pd.read_csv("result_df.csv")

        st.header("Chủ nhà hàng: Báo cáo tổng hợp")
        restaurant_id = st.text_input("Nhập vào ID nhà hàng:")

        def analyze_business(result_df, restaurant_id):
            restaurant_id = int(restaurant_id)
            restaurant_reviews = result_df[result_df['IDRestaurant'] == restaurant_id]
            
            if restaurant_reviews.empty:
                return None
            
            restaurant_name = restaurant_reviews['Restaurant'].iloc[0]
            
            total_comments = restaurant_reviews['Processed_Comment_2'].count()
            sentiment_counts = restaurant_reviews['sentiment'].value_counts()
            positive_comments = (restaurant_reviews['sentiment'] == 'positive').sum()
            negative_comments = (restaurant_reviews['sentiment'] == 'negative').sum()

            aspects = ['FOOD', 'PRICE', 'SERVICE', 'LOCATION', 'AMBIENCE']

            negative_aspects_count = {aspect: restaurant_reviews.loc[restaurant_reviews['sentiment'] == 'negative', aspect].sum() for aspect in aspects}
            most_negative_aspect = max(negative_aspects_count, key=negative_aspects_count.get)
            comment_by_most_neg_aspect = restaurant_reviews.loc[restaurant_reviews[most_negative_aspect] == 'negative', 'Processed_Comment_2']

            positive_aspects_count = {aspect: restaurant_reviews.loc[restaurant_reviews['sentiment'] == 'positive', aspect].sum() for aspect in aspects}
            most_positive_aspect = max(positive_aspects_count, key=positive_aspects_count.get)
            comment_by_most_pos_aspect = restaurant_reviews.loc[restaurant_reviews[most_positive_aspect] == 'positive', 'Processed_Comment_2']

            eat_in_keywords = 'ăn tại chỗ|ghé|đến|không khí| không gian| dễ tìm|vị trí|đông|vắng|ngồi'
            restaurant_reviews['Eat_In'] = restaurant_reviews['Processed_Comment_2'].str.contains(eat_in_keywords, case=False, na=False)
            restaurant_reviews['Take_Away'] = ~restaurant_reviews['Eat_In']
            dine_in_count = restaurant_reviews['Eat_In'].sum()
            takeaway_count = restaurant_reviews['Take_Away'].sum()

            dine_in_ratio = dine_in_count / total_comments if total_comments > 0 else 0
            takeaway_ratio = takeaway_count / total_comments if total_comments > 0 else 0

            Restaurant_report = pd.DataFrame({
                'IDRestaurant': [restaurant_id],
                'Restaurant': [restaurant_name],
                'Total_Comments': [total_comments],
                'Positive_Comments': [positive_comments],
                'Negative_Comments': [negative_comments],
                'Most_Negative_Aspect': [most_negative_aspect],
                'Negative_Comment_By_Aspect': [comment_by_most_neg_aspect],
                'Most_Positive_Aspect': [most_positive_aspect],
                'Positive Comment by aspect': [comment_by_most_pos_aspect],
                'Highest_Rated_Aspect': [most_positive_aspect],
                'Dine_In_Ratio': [dine_in_ratio],
                'Takeaway_Ratio': [takeaway_ratio]
            })

            return Restaurant_report

        def plot_sentiment_distribution_for_restaurant(result_df, restaurant_id):
            sentiment_counts = result_df[result_df['IDRestaurant'] == restaurant_id]['sentiment'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis', ax=ax)
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            ax.set_title(f'Sentiment Distribution for Restaurant ID {restaurant_id}')
            st.pyplot(fig)
        def generate_wordcloud(text):
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        

            positive_comments_text = " ".join(report['Positive Comment by aspect'].iloc[0])
            negative_comments_text = " ".join(report['Negative_Comment_By_Aspect'].iloc[0])
            
            st.write("Word Cloud for Positive Comments:")
            positive_wordcloud = generate_wordcloud(positive_comments_text)
            fig, ax = plt.subplots()
            ax.imshow(positive_wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            st.write("Word Cloud for Negative Comments:")
            negative_wordcloud = generate_wordcloud(negative_comments_text)
            fig, ax = plt.subplots()
            ax.imshow(negative_wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        if st.button("Tạo báo cáo"):
            if restaurant_id:
                report = analyze_business(result_df, restaurant_id)
            
                if report is not None:
                    st.write(f'Báo cáo cho nhà hàng ID {restaurant_id}')
                    st.write(report)

                    # Plot sentiment distribution
                    sentiment_fig = plot_sentiment_distribution_for_restaurant(result_df, restaurant_id)
                    st.pyplot(sentiment_fig)
                    
                    # Display word clouds
                    positive_comments_text = " ".join(report['Positive Comment by aspect'].iloc[0])
                    negative_comments_text = " ".join(report['Negative_Comment_By_Aspect'].iloc[0])

                    st.write("Word Cloud for Positive Comments:")
                    positive_wordcloud = generate_wordcloud(positive_comments_text)
                    fig, ax = plt.subplots()
                    ax.imshow(positive_wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

                    st.write("Word Cloud for Negative Comments:")
                    negative_wordcloud = generate_wordcloud(negative_comments_text)
                    fig, ax = plt.subplots()
                    ax.imshow(negative_wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            else:
                st.write(f'Không tìm thấy dữ liệu cho Restaurant ID {restaurant_id}. Vui lòng kiểm tra lại ID nhà hàng.')
        else:
            st.write("Vui lòng nhập vào ID nhà hàng.")
