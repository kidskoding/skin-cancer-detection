# skin-cancer-detection

<strong> This GitHub repository was the project I submitted for the 2022 Regeneron High School Science Fair. </strong>

## Abstract

Realizing that Skin Cancer was one of the most prominent forms of cancer in today's world, 
I sought out to build a simple skin-care-detection software by utilizing advanced Machine Learning techniques, including TensorFlow and Keras, 
in order to accurately detect skin cancer among patients, by training it with hundreds of thousands of skin cancer lesion images.
<br /> <br />
<strong>This project won an honorable mention medal at the district-level science fair competition, along with winning the NASA Jacobs Explorer Award at the city-level science fair competition.</strong>
<hr />

## Project Presentation
The full description of the project presentation, including background, approach, hypotheses, procedures, and in depth results can be found by viewing this ![presentation]([https://github.com/kidskoding/skin-cancer-detection/blob/master/LICENSE](https://github.com/kidskoding/skin-cancer-detection/blob/master/SEFH%202022%20-%20Skin%20Cancer.pdf))

## Overview
Skin cancer is a prevalent form of cancer, with over 9,000 daily diagnoses in the US alone. 
This project leverages Artificial Intelligence (AI) and Machine Learning (ML) to predict skin cancer, 
aiming to improve early detection and access to care, especially in rural areas. 

## Features
- Utilizes ML models like KNN, CNN, RFC, and DTC for skin cancer detection.
- Analyzes image resolutions, blurring effects, batch sizes, and epochs for optimization.
- Measures performance using accuracy, Receiver Operator Curves (ROCs), and Area Under ROC Curves (AUCs).
    - A great accuracy and AUC score would be above 70%, or .70

## Software Used
- <strong>Programming Tools</strong>: Python, Pandas, Tensorflow, Keras, scikit-learn, and OpenCV
- <strong>Data Visualization with ROCs and AUCs</strong>: Matplotlib and Seaborn

## Dataset
This project uses the ISIC skin cancer dataset consisting of approximately 10,000 images along with metadata for training and testing the ML models.

## Usage
- Set up the Python environment and install the required packages.
- Download the ![ISIC dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) and preprocess the data as needed.
- Train the ML models using different parameters and evaluate performance.
- Analyze the results and determine whether the accuracy and AUC scores of the Machine Learning models are effective enough (greater than 0.70 or 70%) to accurately determine skin cancer in a patient.

## License
This project is licensed under the ![MIT License](https://opensource.org/licenses/MIT) - see the ![LICENSE](https://github.com/kidskoding/skin-cancer-detection/blob/master/LICENSE) file for details.
