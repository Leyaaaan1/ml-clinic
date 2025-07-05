# 🫀 Heart Disease Risk Prediction System

A machine learning-powered web application that predicts heart disease risk based on patient health data and lifestyle factors.

## 🚀 Features

- **Interactive Web Interface**: Built with Streamlit for easy patient data input
- **Machine Learning Model**: Random Forest classifier trained on health indicators
- **Risk Assessment**: Provides confidence scores and personalized recommendations
- **Medical Guidelines**: Offers clinical recommendations based on predictions

## 📋 Project Structure

```
heart_disease_predictor/
├── app.py                    # Streamlit web application
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── heart_disease_model.pkl  # Trained ML model
├── model_features.pkl       # Feature list for model
└── README.md               # Project documentation
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Leyaaaan1/machine-clinic.git
   cd machine-clinic
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (optional):**
   ```bash
   python train_model.py
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 🔧 Usage

1. Open the web application in your browser
2. Fill out the patient information form:
   - Personal details (BMI, age, sex, race)
   - Health metrics (physical/mental health days, sleep)
   - Medical history (smoking, alcohol, stroke, diabetes)
   - Lifestyle factors (physical activity, general health)
3. Click "Predict" to get the risk assessment
4. Review the confidence score and recommendations

## 📊 Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 14 health and lifestyle indicators
- **Training**: Uses simulated health data (replace with real dataset)
- **Output**: Binary classification (at risk / not at risk)

## 🩺 Medical Recommendations

The system provides different recommendations based on the prediction:

**High Risk Patients:**
- Immediate medical consultation
- Lifestyle changes (smoking cessation, exercise)
- Regular monitoring of vital signs
- Cardiology referral if needed

**Low Risk Patients:**
- Continue healthy habits
- Annual wellness screenings
- Cardiovascular education

## ⚠️ Important Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🔮 Future Enhancements

- Integration with real medical datasets
- Advanced model validation and testing
- Multi-class risk categorization
- Export functionality for medical reports
- Integration with healthcare systems
