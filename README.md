# Safe-Space-_Ai-model-for-mental-health-Text-analyzer
SafeSpace AI is an AI-powered mental health text analysis app that analyzes how users express their feelings and predicts the most likely mental health category using NLP, TF-IDF, and XGBoost. It provides personalized guidance and an AI Helper to encourage awareness, early attention, and seeking appropriate support when needed.
# SafeSpace AI

> **Understand what your words may be saying before they become a warning sign.**

SafeSpace AI is a bilingual **NLP + Machine Learning + Generative AI** application designed around one simple idea: sometimes the words people say or write can carry signals that deserve attention before they become a bigger warning sign.

The project is built for **awareness, reflection, and supportive interaction — not diagnosis**. A single emotion such as sadness should never be treated as proof of depression. Instead, the application analyzes the overall linguistic pattern of a user's text and presents probability-based results across seven learned categories.

---

## Why I Created This Project

People say things every day without realizing how important those words may be.

Someone may describe stress, fear, hopelessness, emotional exhaustion, or other difficult experiences without directly saying that they need help. The idea behind SafeSpace AI is to make people more aware of the language they use and of changes in the way people around them express themselves.

If we pay closer attention to these signals, we may be able to recognize that something deserves care earlier and encourage the person to seek appropriate support.

At the same time, **being sad, stressed, tired, or overwhelmed is normal**. SafeSpace AI therefore does not use simple rules such as:

```text
"sad"      → Depression
"worried"  → Anxiety
```

The machine-learning model learns patterns from the training data instead of relying on individual keywords.

---

# Project Idea

SafeSpace AI gives a user a space to express a thought, feeling, or experience in their own words.

The application then:

1. Accepts free-form text.
2. Normalizes the input.
3. Converts Arabic input to English when necessary because the trained classifier works with English text.
4. Extracts numerical NLP features using TF-IDF.
5. Adds two simple text-level features: **character count** and **word count**.
6. Sends the resulting feature vector to the trained XGBoost classifier.
7. Calculates probabilities for seven categories.
8. Displays the strongest predicted category and the probability distribution.
9. Generates supportive AI feedback based on the actual result.
10. Provides practical guidance selected for the predicted category rather than showing one random or permanently fixed message.
11. Gives the user access to a separate real conversational **AI Helper** connected to the OpenAI API.

### Seven prediction categories

- Anxiety
- Bipolar
- Depression
- Normal
- Personality Disorder
- Stress
- Suicidal

The output is a **model prediction, not a clinical diagnosis**.

---

# How the Application Works

## 1. Bilingual Interface

The application was designed to be comfortable for both Arabic- and English-speaking users.

### Language controls

The interface provides a direct language switch between:

**English ↔ Arabic**

The selected language affects the interface, guidance, AI support, and AI Helper conversation.

When Arabic text is entered, the application can translate it to English before sending it through the trained English NLP model. This keeps the model pipeline consistent while still allowing the user to communicate naturally in Arabic.

---

## 2. Comfortable User Interface

The application includes a **Light / Dark theme control** so the interface can be adjusted for visual comfort and different lighting conditions.

Other UI decisions were made to keep the experience simple and focused:

- Clear input area
- One-click analysis
- Language switching
- Theme switching
- Separate AI Helper page
- Clear probability visualization
- Session analysis history
- Responsive Streamlit layout

---

## 3. Free-Form Text Input

The main text area is intentionally designed for **free expression**.

There is no artificial requirement to write a fixed number of words or characters. The user can describe what they are feeling in their own way rather than being forced into a questionnaire-style format.

The application also accepts natural writing, including informal wording, punctuation, and common spelling variations.

The goal is to preserve the user's natural expression as much as possible before analysis.

---

# Machine Learning Pipeline

The machine-learning part of the project was developed as a complete NLP classification pipeline.

```text
Mental-Health Text Dataset
          ↓
     Data Loading
          ↓
    Data Inspection
          ↓
   Text Preprocessing
          ↓
 Tokenization / Normalization
          ↓
       Stemming
          ↓
        TF-IDF
          ↓
 + Character Count
 + Word Count
          ↓
   Feature Matrix
          ↓
 Train Multiple ML Models
          ↓
   Model Comparison
          ↓
     XGBoost Selected
          ↓
      Save Model
          ↓
   Streamlit Deployment
```

---

# 4. Data Loading

The first stage was importing the mental-health text dataset and understanding its structure before building the model.

The data-loading stage included:

- Reading the dataset
- Inspecting columns
- Checking the target labels
- Reviewing text examples
- Checking missing values
- Understanding the distribution of categories
- Preparing the data for supervised learning

This stage is important because model performance depends heavily on the quality and structure of the training data.

---

# 5. Text Preprocessing

Raw natural language contains noise and variation that can make traditional NLP models less effective.

The preprocessing stage was used to create a more consistent textual representation.

The pipeline includes operations such as:

- Normalizing text
- Cleaning unnecessary characters
- Tokenization
- Removing textual noise
- Stemming

The purpose is not to remove the meaning from the text, but to make similar linguistic patterns easier for the model to learn.

---

# 6. TF-IDF Feature Extraction

After preprocessing, the text is converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The final saved model uses a `TfidfVectorizer` configured with:

| Setting | Value |
|---|---:|
| N-gram range | `(1, 2)` |
| Maximum TF-IDF features | `20,000` |
| Vocabulary size in saved model | `20,000` |

This means the vectorizer learns both:

- individual words (**unigrams**)
- two-word combinations (**bigrams**)

TF-IDF gives greater importance to terms that are useful for distinguishing documents while reducing the influence of terms that occur broadly across the corpus.

---

# 7. Feature Engineering

The application does not rely only on the TF-IDF representation.

Two additional numerical features are added to the TF-IDF matrix:

- **Character count**
- **Word count**

The production feature vector therefore contains:

```text
20,000 TF-IDF features
        +
1 character-count feature
        +
1 word-count feature
        =
20,002 total input features
```

This is also reflected in the saved XGBoost model, which expects **20,002 input features**.

---

# 8. Model Comparison

Instead of selecting an algorithm without testing alternatives, the project compared **four machine-learning models** on the classification task and selected the strongest-performing model.

The confirmed final result is:

### 🏆 XGBoost — 96.95% Accuracy

XGBoost was selected as the final production classifier because it achieved the highest accuracy in the model comparison.

> **Important:** The current deployment files contain the final XGBoost model, but they do not contain the original four-model training/evaluation output with the other three exact accuracy values. I therefore do not list invented numbers for those models. The confirmed project value is **96.95% for XGBoost**.

### Final model configuration

The saved XGBoost model contains these key settings:

| Parameter | Value |
|---|---:|
| Model | XGBClassifier |
| Objective | `multi:softprob` |
| Evaluation metric | `mlogloss` |
| Number of estimators | `100` |
| Max depth | `5` |
| Learning rate | `0.1` |
| Random state | `101` |
| Tree method | `hist` |
| Number of input features | `20,002` |
| Number of classes | `7` |

These values are taken from the saved model artifact used by the application.

---

# 9. Saving the Model

After training and selecting the best model, the components required for prediction were serialized into a model package.

The saved package contains:

```text
model.pkl
├── model
├── vectorizer
├── label_encoder
└── stemmer
```

This allows the deployed application to load the trained pipeline instead of retraining the model every time a user opens the application.

The production application loads the model with Joblib and then reuses the saved vectorizer and label encoder during prediction.

---

# 10. Prediction in the Application

When the user presses **Analyze Text**, the application follows the same general feature-construction logic used during model development.

```text
User Input
   ↓
Normalize Text
   ↓
Translate Arabic → English when needed
   ↓
TF-IDF Vectorization
   ↓
Character Count + Word Count
   ↓
20,002-Feature Vector
   ↓
XGBoost
   ↓
7 Class Probabilities
   ↓
Prediction + Confidence
```

The application uses `predict_proba()` to obtain a probability distribution across all seven classes.

This is why the result is not limited to a simple `Yes / No` answer.

---

# 11. Probability-Based Results

The application displays the probability associated with each category.

This is important because a machine-learning prediction should not be presented as an absolute fact about a person.

For example, the application can show that several categories received different probabilities rather than pretending that one label is a certain diagnosis.

The main principle is:

> **The model provides a learned statistical indication from text — not a medical diagnosis.**

---

# 12. Personalized Guidance

One of the important parts of SafeSpace AI is that the guidance is **not random and not one fixed paragraph reused for every user**.

The application selects guidance according to the category produced by the model.

For example, different categories have different practical guidance, such as:

- Anxiety → grounding and breathing techniques
- Stress → identifying stressors and taking manageable breaks
- Depression → small actions, connection, and professional support when needed
- Bipolar → consistent routines, sleep tracking, and professional care
- Personality disorder → supportive coping and professional support
- Normal → maintaining healthy routines and wellbeing practices
- Suicidal → immediate human and emergency support

The project also includes references to public guidance from organizations such as **WHO** and **NIMH** for the practical guidance section.

This creates a result that is connected to the model's output instead of displaying a random motivational message.

---

# 13. AI-Generated Support

In addition to the fixed category-specific guidance, SafeSpace AI can generate a short supportive response using a generative AI model.

The AI support receives the user's text and the model's predicted category and generates a response that is:

- Supportive
- Practical
- Non-judgmental
- Brief
- Not presented as a diagnosis

If the generative AI service is unavailable, the application has a built-in supportive fallback so the core experience can continue.

For high-risk suicidal predictions, the application uses a dedicated crisis-focused response instead of relying on a generic generated message.

---

# 14. Real AI Helper Chatbot

SafeSpace AI also contains a separate **AI Helper** designed as a real conversational assistant rather than a static FAQ or a collection of saved responses.

The AI Helper is connected to the **OpenAI API** and can maintain the conversation context during the current session.

### What the AI Helper does

- Holds a real back-and-forth conversation
- Uses the selected language
- Responds to the context of previous messages
- Uses neutral, non-judgmental language
- Does not assume the user's gender
- Does not diagnose mental-health conditions
- Does not provide medication instructions
- Prioritizes human and emergency support if immediate self-harm risk is expressed

The chat is intentionally separated from the prediction screen so the user can move into a dedicated conversational experience without cluttering the analysis interface.

```text
              SafeSpace AI
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
 Machine Learning        Generative AI
        │                     │
   Text Analysis          AI Helper Chat
        │                     │
 7-Class Probability    OpenAI API
```

---

# 15. Language Switching

The application provides a direct **English ↔ Arabic** experience.

The language switch is not only visual. The application also adapts the generated support and AI Helper responses to the selected language.

For Arabic input, translation can be used before classification so that the English-trained model can process the text consistently.

---

# 16. Theme Switching

A Light / Dark theme option was added to make the application more comfortable to use across different environments.

This is a small UI feature, but it supports the project's overall goal of creating a calm and accessible space for users to express themselves.

---

# 17. Session History

The application keeps the user's previous analyses visible during the current session.

A user can review:

- Previous text analyses
- Predicted categories
- Confidence values
- Probability distributions

The history can also be cleared from the interface.

---

# 18. Streamlit Application

After the machine-learning model was trained, evaluated, saved, and integrated, the project was converted into an interactive web application using **Streamlit**.

The application brings together:

- NLP preprocessing
- Translation support
- Machine-learning inference
- Probability visualization
- Category-specific guidance
- Generative AI support
- Real AI Helper chat
- English / Arabic switching
- Light / Dark theme
- Session history
- Crisis-oriented handling for high-risk predictions

This turned the project from a model into an actual interactive AI application.

---

# Complete Project Workflow

```text
Dataset
   ↓
Data Loading & Inspection
   ↓
Cleaning / Preprocessing
   ↓
Tokenization / Normalization / Stemming
   ↓
TF-IDF Feature Extraction
   ↓
Feature Engineering
   │
   ├── 20,000 TF-IDF features
   ├── Character count
   └── Word count
   ↓
20,002 Features
   ↓
Train 4 ML Models
   ↓
Compare Accuracy
   ↓
Select XGBoost
   ↓
96.95% Accuracy
   ↓
Save model + vectorizer + encoder + stemmer
   ↓
Build Streamlit Application
   ↓
Add Arabic / English Support
   ↓
Add Personalized Guidance
   ↓
Add OpenAI AI Support
   ↓
Build Real AI Helper Chatbot
   ↓
Add Light / Dark Theme
   ↓
Deploy on Streamlit Community Cloud
```

---

# Code Architecture

The production application is organized around several main responsibilities.

### Model loading

A cached model-loading function loads the serialized model once and reuses it during the Streamlit session lifecycle.

### Text normalization

A dedicated normalization function cleans user input before it enters the prediction pipeline.

### Translation

Arabic text can be translated to English when required because the trained classifier uses English NLP features.

### Prediction

The prediction function:

1. Loads the model package.
2. Gets the saved vectorizer, model, and label encoder.
3. Transforms the text with TF-IDF.
4. Calculates character count.
5. Calculates word count.
6. Concatenates the features using a sparse matrix.
7. Verifies the feature count expected by XGBoost.
8. Calculates class probabilities.
9. Decodes the predicted class.
10. Returns the result to the interface.

### Guidance engine

The guidance dictionary maps each predicted category to its own supportive explanation, practical steps, and public reference sources.

### Generative AI support

The OpenAI integration generates a short supportive response based on the user's actual text and the model result.

### AI Helper

The chatbot stores the current conversation messages and sends recent context to the OpenAI API so responses can remain conversational and relevant.

### Streamlit UI

The UI layer controls:

- Language
- Theme
- Text input
- Translation
- Analysis
- Results
- Guidance
- AI support
- AI Helper
- History
- Developer contact

---

# Technologies Used

| Technology | Role in the project |
|---|---|
| Python | Main development language |
| Pandas | Dataset handling during model development |
| NumPy | Numerical/data operations during development |
| Scikit-learn | TF-IDF, preprocessing and ML utilities |
| TF-IDF | Text-to-numeric feature extraction |
| XGBoost | Final 7-class classifier |
| SciPy Sparse | Combining TF-IDF with numerical features efficiently |
| Joblib | Saving and loading the trained model package |
| OpenAI API | Generative AI support and AI Helper conversation |
| Deep Translator | Arabic-to-English translation support |
| Streamlit | Web application interface and deployment |

---

# Project Structure

The **public repository is intentionally documentation-focused**.

The production source code and trained model are kept private rather than published in the showcase repository.

```text
SafeSpace-AI-public/
│
└── README.md
```

The private deployment repository contains the files required to run the application, while this public repository explains the project, methodology, architecture, and final result.

---

# Deployment

After the final model was saved and integrated into the application, SafeSpace AI was deployed using **Streamlit Community Cloud**.

## Live Application try it 

**[https://safespace-ai-mental-health-ai-model-deployed-by-app-pbu5ec5hnx.streamlit.app/]**

## Video Demo

**[https://youtu.be/_t7rH7UcjP0?si=lMOmBD-rzDkUO5ze]**

---

# Important Limitation

SafeSpace AI is an **educational and awareness-oriented AI project**. It is not a medical or psychological diagnostic system.

Mental-health conditions cannot be reliably diagnosed from one sentence or one machine-learning prediction. The probabilities shown by the application describe what the trained model predicts from the text; they do not establish that a person has a condition.

If someone appears to be in immediate danger or may harm themselves, professional or emergency support should be contacted instead of relying on an AI system.

---

# What This Project Demonstrates

SafeSpace AI demonstrates an end-to-end AI development workflow:

- Dataset exploration
- Text preprocessing
- NLP
- TF-IDF feature engineering
- Sparse feature construction
- Training multiple machine-learning models
- Model comparison
- XGBoost classification
- Probability-based prediction
- Model serialization
- Bilingual application design
- Translation integration
- Personalized category-specific guidance
- Generative AI integration
- Real conversational AI
- Streamlit application development
- Cloud deployment

---

# Final Thought

SafeSpace AI started from a simple observation:

> **Sometimes we say things without realizing how important those words may be.**

The purpose of the project is not to label people or turn normal emotions into diagnoses.

It is about **awareness** — understanding the language we use, paying attention to meaningful patterns, and creating a supportive space where a person can express what may be difficult to say out loud.

---

## Author

**Habiba Ahmed Talat**  
 AI & Data Science Engineer

Interested in:
-Data Analysis
-Big Data
- Artificial Intelligence
- Machine Learning
- Natural Language Processing
- 
- Data Visualization
- AI Application Development
<img width="1920" height="1015" alt="image" src="https://github.com/user-attachments/assets/7b7158c8-9957-44bb-901f-39a3e3edbeb5" />
      
<img width="1920" height="1008" alt="image" src="https://github.com/user-attachments/assets/0a24d1f1-4f03-4a55-ae6b-ae0f2f229efa" />

<img width="1918" height="979" alt="image" src="https://github.com/user-attachments/assets/668bef72-e23f-405a-b927-cd660323e1d4" />

<img width="1920" height="1009" alt="image" src="https://github.com/user-attachments/assets/7cb6e4c3-68db-4eee-8740-9e2b8bde436d" />

<img width="1894" height="1015" alt="image" src="https://github.com/user-attachments/assets/bda6c014-6cce-4bb5-b520-ac21a653006b" />
<img width="1920" height="1014" alt="image" src="https://github.com/user-attachments/assets/10700334-8055-4f97-9b5d-43c7b869bca0" />
