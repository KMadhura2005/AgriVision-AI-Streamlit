\# 🌱 AgriVision-AI



\*\*AI-Based Crop Disease Detection System using Streamlit and Deep Learning\*\*



AgriVision-AI is a deep learning-based web application that helps identify crop diseases from leaf images. The application uses a trained TensorFlow/Keras model and provides predictions through an easy-to-use Streamlit interface.



\## 🚀 Features



\* 🌿 Upload a crop leaf image

\* 🤖 AI-based disease prediction

\* 📊 Displays the predicted disease/class

\* 🖥️ Interactive Streamlit web interface

\* 🌾 Supports multiple crop disease classes

\* ⚡ Fast image-based prediction

\* 🎨 Agriculture-themed user interface



\## 🛠️ Technologies Used



\* \*\*Python\*\*

\* \*\*Streamlit\*\*

\* \*\*TensorFlow / Keras\*\*

\* \*\*NumPy\*\*

\* \*\*Pillow\*\*

\* \*\*JSON\*\*

\* \*\*Git \& GitHub\*\*



\## 📂 Project Structure



```text

AgriVision-AI-Streamlit/

│

├── app.py

├── agri\_vision\_model\_9class.keras

├── class\_names.json

├── agri-background-image.png

├── requirements.txt

├── .gitignore

└── README.md

```



\## 🧠 How It Works



1\. The user uploads an image of a crop leaf.

2\. The image is processed and resized according to the model's requirements.

3\. The trained deep learning model analyzes the image.

4\. The model predicts the corresponding disease/class.

5\. The predicted result is displayed in the Streamlit application.



\## 🌾 Supported Classes



The application uses a \*\*9-class trained model\*\*. The exact class names used by the application are stored in:



```text

class\_names.json

```



This keeps the class labels synchronized with the trained model.



\## ⚙️ Installation



Clone the repository:



```bash

git clone https://github.com/KMadhura2005/AgriVision-AI-Streamlit.git

```



Move into the project directory:



```bash

cd AgriVision-AI-Streamlit

```



Install the required dependencies:



```bash

pip install -r requirements.txt

```



\## ▶️ Run the Application



Start the Streamlit application:



```bash

streamlit run app.py

```



The application will open in your browser.



\## 📸 Application



Upload a crop leaf image through the Streamlit interface and the AI model will analyze the image and display its predicted disease/class.



\## 🎯 Project Objective



The main objective of AgriVision-AI is to demonstrate how artificial intelligence and deep learning can be applied to agriculture for automated crop disease identification.



Early identification of crop diseases can help farmers take appropriate action and potentially reduce crop losses.



\## 🔮 Future Scope



\* Add more crops and disease classes

\* Improve model accuracy with a larger dataset

\* Add disease treatment recommendations

\* Add multilingual support

\* Deploy the application online

\* Add confidence scores and prediction explanations

\* Integrate additional agricultural information



\## 👩‍💻 Author



\*\*Madhura Kadam\*\*



Computer Engineering Student



\---



⭐ If you find this project useful, consider giving the repository a star!



