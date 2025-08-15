# Sepsis Prediction Model using MIMIC Dataset

A machine learning project that predicts sepsis in hospitalized patients using the MIMIC (Medical Information Mart for Intensive Care) dataset. This project implements a Random Forest classifier with comprehensive data visualization and model evaluation.

## 🎯 Project Overview

Sepsis is a life-threatening condition that requires early detection for optimal patient outcomes. This project uses machine learning to predict sepsis risk based on patient demographics and clinical data from the MIMIC dataset.

### Key Features
- ✅ Automated data preprocessing and cleaning
- ✅ Comprehensive exploratory data analysis with visualizations
- ✅ Random Forest classifier with class imbalance handling
- ✅ Extensive model evaluation metrics and plots
- ✅ Error analysis and model interpretability

## 📊 Dataset

This project uses the [MIMIC Dataset](https://mimic.physionet.org/), specifically:
- `admissions.csv` - Hospital admission records
- `patients.csv` - Patient demographic information
- `diagnoses_icd.csv` - ICD diagnosis codes
- `d_icd_diagnoses.csv` - ICD code descriptions
- `labevents.csv` - Laboratory test results (optional)

### Sepsis Definition
Sepsis cases are identified using the following ICD codes:
- `99591` - Sepsis
- `99592` - Severe sepsis
- `78552` - Septic shock
- `R6520` - Severe sepsis without septic shock
- `R6521` - Severe sepsis with septic shock
- `A419` - Sepsis, unspecified organism

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.7
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/sepsis-prediction.git
cd sepsis-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download MIMIC dataset and place CSV files in the `data/` directory

4. Update the file path in the script:
```python
base_path = r"path/to/your/data/folder"
```

5. Run the model:
```bash
python sepsis_prediction.py
```

## 📁 Project Structure
```
sepsis-prediction/
│
├── data/                          # MIMIC dataset files
│   ├── admissions.csv
│   ├── patients.csv
│   ├── diagnoses_icd.csv
│   ├── d_icd_diagnoses.csv
│   └── labevents.csv (optional)
│
├── sepsis_prediction.py           # Main script
├── debug_data_loading.py          # Debug script for troubleshooting
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── results/                       # Generated plots and results
    ├── eda_plots.png
    ├── model_evaluation.png
    └── additional_analysis.png
```

## 🔬 Model Performance

The Random Forest classifier achieves the following performance metrics (example results):

| Metric | Score |
|--------|-------|
| ROC AUC | 0.785 |
| Precision | 0.721 |
| Recall | 0.658 |
| F1-Score | 0.688 |
| Accuracy | 0.892 |

*Note: Actual performance may vary depending on your specific dataset and preprocessing.*

## 📈 Visualizations

The project generates comprehensive visualizations including:

### Exploratory Data Analysis
- Class distribution (sepsis vs non-sepsis)
- Age distribution by sepsis status
- Gender distribution analysis
- Age vs gender scatter plots
- Sepsis rates by age groups
- Feature correlation heatmap

### Model Evaluation
- Confusion matrix
- ROC curve with AUC score
- Precision-recall curve
- Feature importance plot
- Prediction probability distributions
- Performance metrics comparison

### Advanced Analysis
- Model calibration plots
- Error analysis by demographics
- Learning curves
- Prediction confidence analysis

## 🛠️ Features

### Data Preprocessing
- Automatic data type conversion and validation
- Missing value handling
- Data quality checks
- Age group binning and feature engineering

### Model Training
- Random Forest with class imbalance handling
- Stratified train-test split
- Cross-validation ready structure
- Feature importance analysis

### Evaluation
- Multiple evaluation metrics
- Visual performance analysis
- Error analysis by patient subgroups
- Model interpretability features

## 🐛 Troubleshooting

### Common Issues

1. **File Path Errors**
   - Ensure all CSV files are in the correct directory
   - Check file names match exactly (case-sensitive)
   - Use absolute paths if relative paths fail

2. **Data Type Errors**
   - Run the debug script first: `python debug_data_loading.py`
   - The script automatically handles most data type conversions

3. **Memory Issues**
   - Consider using a subset of the data for initial testing
   - Optimize pandas operations for large datasets

4. **No Sepsis Cases Found**
   - Verify ICD codes match your dataset version (ICD-9 vs ICD-10)
   - Check the sample ICD codes printed by the debug script

## 📋 Requirements

Create a `requirements.txt` file:
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 🔄 Future Enhancements

- [ ] Add more clinical features (lab values, vital signs)
- [ ] Implement deep learning models (LSTM, CNN)
- [ ] Add real-time prediction capability
- [ ] Integrate with hospital information systems
- [ ] Add model explainability (SHAP, LIME)
- [ ] Implement ensemble methods
- [ ] Add cross-validation and hyperparameter tuning
- [ ] Create web interface for predictions

## 📚 References

1. Johnson, A., Pollard, T., & Mark, R. (2019). MIMIC-III Clinical Database (version 1.4). PhysioNet.
2. Singer, M., et al. (2016). The third international consensus definitions for sepsis and septic shock (Sepsis-3). JAMA, 315(8), 801-810.
3. Seymour, C. W., et al. (2016). Assessment of clinical criteria for sepsis: for the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA, 315(8), 762-774.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Notes

- **Data Privacy**: Ensure compliance with HIPAA and other data privacy regulations
- **Clinical Use**: This model is for research purposes only and should not be used for clinical decision-making without proper validation
- **Dataset Access**: MIMIC dataset requires credentialed access through PhysioNet

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/sepsis-prediction](https://github.com/yourusername/sepsis-prediction)

## 🙏 Acknowledgments

- [MIT Laboratory for Computational Physiology](https://lcp.mit.edu/) for the MIMIC dataset
- [PhysioNet](https://physionet.org/) for data hosting and access
- The open-source community for the amazing tools and libraries

---

