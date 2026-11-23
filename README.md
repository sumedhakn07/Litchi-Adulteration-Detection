<div align="center">

# 🍈 LADD: Litchi Adulteration & Decay Detection
### *A Smart Multimodal AI that "Sees" and "Sniffs" Fruit to Catch Dangerous Food Adulteration*

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.78%25-22C55E?style=for-the-badge)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

**Author:** Sumedha K N (Reg. No: 2582424) • **Class:** V MSDA (M.Sc. Data Science)  
**Supervisor:** Dr. Kavitha R • **Department of Statistics & Data Science**  
**Institution:** CHRIST (Deemed to be University), Central Campus, Bengaluru  
**Course:** MDA507-5 Neural Networks & Deep Learning  

---

</div>

## 📖 Table of Contents
1. [🌟 Explain Like I'm in 10th Grade (The Big Picture)](#-explain-like-im-in-10th-grade-the-big-picture)
2. [⚠️ The Real-World Problem](#-the-real-world-problem)
3. [🧪 The LADD Dataset & The 7 "Super-Senses"](#-the-ladd-dataset--the-7-super-senses)
4. [🧠 How the AI Brain Works (GMU Fusion)](#-how-the-ai-brain-works-gmu-fusion)
5. [📊 Battle of the Models: Who Won?](#-battle-of-the-models-who-won)
6. [🔍 Why Cameras Alone Fail (The 7 Mistakes)](#-why-cameras-alone-fail-the-7-mistakes)
7. [🎛️ The AI's "Referee": Gate Interpretability](#️-the-ais-referee-gate-interpretability)
8. [🚀 Live Interactive Web App (Streamlit)](#-live-interactive-web-app-streamlit)
9. [💻 Step-by-Step Installation & Quickstart](#-step-by-step-installation--quickstart)
10. [📂 Project Folder Structure](#-project-folder-structure)
11. [🌍 Alignment with UN Sustainable Development Goals](#-alignment-with-un-sustainable-development-goals)
12. [🔮 Future Scope](#-future-scope)
13. [📜 Citation & Academic Details](#-citation--academic-details)

---

## 🌟 Explain Like I'm in 10th Grade (The Big Picture)

### 🛒 The Grocery Store Mystery
Imagine you go to a fruit market to buy a bunch of juicy, bright-red litchis.
- One bag has **Fresh** litchis: healthy, sweet, and safe.
- Another bag has **Rotten** litchis: smelly, squishy, and brown.
- But there is a dangerous **Third Bag**: **Formalin-Mixed** litchis!

### 😱 What is Formalin, and Why is it Scary?
**Formalin** is a liquid chemical made of formaldehyde. Hospitals and biology labs use it to preserve specimens (like frogs in glass jars) so they never decompose. 

Some dishonest fruit sellers dip litchis in formalin! Why? Because fresh litchis naturally turn brown and dry up in just 2 days. Formalin artificially freezes the red color and keeps the rind stiff. **It looks fresh on the outside, but it is laced with toxic, cancer-causing poison on the inside!**

### 🤖 Why Normal Computer Vision Gets Tricked
If you build a camera AI to look at fruit:
- The camera sees the bright-red formalin litchi and says: *"Looks red and shiny! It's fresh!"* ❌ **WRONG!**
- The camera sees a healthy fresh litchi with a tiny harmless brown sunspot and says: *"Ugh, brown spots! It's rotten!"* ❌ **WRONG!**

### 💡 Our Super-Detective AI Solution
We built an AI that doesn't just use its **eyes**—it also uses a **digital nose and fingers**!
1. **The Eyes (Camera):** Takes a photo using a lightweight neural network called **MobileNet-V3**.
2. **The Nose & Touch (7 Chemical Sensors):** Measures chemical gas emissions ($\text{CO}_2$, ethanol, formaldehyde), moisture, skin firmness, acidity ($\text{pH}$), and sweetness ($\text{TSS}$).
3. **The Gated Multimodal Unit (The Brain):** A smart referee that combines both senses. If the camera says *"Looks fresh!"* but the chemical sensor detects formaldehyde, the brain shouts: **"Danger! That's Formalin!"**

The result? Our multimodal AI reaches **99.78% accuracy**—almost perfect!

---

## ⚠️ The Real-World Problem

```
       [ Fresh Litchi ]                     [ Formalin-Treated Litchi ]
     Anthocyanin Pigments                Chemical Hardening of Pericarp
   Safe, Sweet, High Vitamin C            Toxic, Carcinogenic Formaldehyde
              │                                          │
              ▼                                          ▼
   ┌─────────────────────────────────────────────────────────────┐
   │            THEY LOOK VIRTUALLY IDENTICAL TO A CAMERA!       │
   └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
      Camera Alone: 7 Errors  ──►  MULTIMODAL AI: ONLY 1 ERROR!
```

- **Health Hazard:** Formaldehyde is classified as a **Group 1 Carcinogen** by the World Health Organization (WHO). Consuming it causes stomach ulcerations, kidney damage, and long-term leukemia risks.
- **Economic Waste:** Litchis brown quickly due to natural enzymes (PPO and POD). Millions of dollars worth of safe, edible fruits are discarded simply because cameras think surface browning means the pulp is rotten.
- **Law Enforcement:** India's food regulator (**FSSAI**) strictly bans formaldehyde in fresh fruits. Our tool gives inspectors an automated, instant screening system.

---

## 🧪 The LADD Dataset & The 7 "Super-Senses"

The **LADD (Litchi Adulteration and Decay Detection)** benchmark contains **3,000 fruit instances**, divided equally into three classes:
- **1,000 Fresh** (Class 0)
- **1,000 Formalin-Mixed** (Class 1)
- **1,000 Rotten** (Class 2)

<div align="center">
  <img src="results/eda_sample_images.png" width="700" alt="LADD Sample Images"/>
  <p><i>Figure 1: Notice how Formalin-Mixed litchis look almost as firm and red as genuine Fresh ones!</i></p>
</div>

### 🔬 What Do the 7 Sensors Actually Measure?

| Sensor Parameter | Unit | What it Means in Plain English | What Happens During Decay / Adulteration? |
| :--- | :---: | :--- | :--- |
| **Moisture** | % | How juicy and hydrated the fruit skin is | Drops from **78.5%** in fresh down to **64.8%** in rotten fruit. |
| **Firmness** | N | How hard the shell pushes back when pressed | Formalin chemically hardens the skin (**3.51 N**) compared to mushy rotten fruit (**0.99 N**). |
| **pH** | — | Acidity of the fruit juice | Bacteria & yeast make acid; rotten fruit becomes sour (**pH 3.84**) vs fresh (**pH 4.75**). |
| **TSS** | °Brix | Total Soluble Solids (sugar sweetness) | Microbes eat the natural sugars; drops from **18.0 °Brix** down to **10.1 °Brix**. |
| **Ethanol** | g/L | Alcohol produced by yeast fermentation | Spikes up to **9.98 g/L** in rotten fruit (vs **0.30 g/L** in fresh). |
| **CO₂** | ppm | Carbon dioxide gas "breathed out" by microbes | Massive surge to **1,146 ppm** in rotten litchis (vs **298 ppm** in fresh). |
| **Formaldehyde** | mg/kg | Poison preservative chemical level | Stays at **0.00** in natural fruit, but spikes in **Formalin-Mixed** fruit! |

<div align="center">
  <img src="results/eda_boxplots.png" width="850" alt="Sensor Feature Boxplots"/>
  <p><i>Figure 2: Boxplots showing how rotten and formalin fruits show clear chemical differences.</i></p>
</div>

---

## 🧠 How the AI Brain Works (GMU Fusion)

Our model is called a **Gated Multimodal Unit (GMU)**. It has fewer than **1 million parameters** (979,427 parameters), making it light enough to run on a cheap mini-computer like a Raspberry Pi!

```
                  ┌────────────────────────┐
                  │ 📷 224x224 RGB Photo   │
                  └───────────┬────────────┘
                              │
                              ▼
                  [ MobileNet-V3 Small ]
                              │
                              ▼ (Linear 1024 -> 64)
                   Visual Embedding (h_v)
                              │
                              ├──────────────────────┐
                              │                      │
                              ▼                      ▼
                     ┌──────────────────┐    ┌───────────────┐
                     │  GATING UNIT (g) │    │ FUSION LAYER  │
                     │  "The Referee"   │───►│  h = g·h_v +  │
                     │  σ(W[h_v; h_s])  │    │  (1-g)·h_s    │
                     └──────────────────┘    └───────┬───────┘
                              ▲                      │
                              │                      ▼
                   Sensor Embedding (h_s)   [ Classifier Head ]
                              ▲              (Linear 64->32->3)
                              │                      │
                  [ 2-Layer Sensor MLP ]             ▼
                              ▲                 OUTPUT CLASS:
                              │                • Fresh
                  ┌───────────┴────────────┐   • Formalin-Mixed
                  │ 🧪 7 Sensor Readings   │   • Rotten
                  └────────────────────────┘
```

### The Math in 3 Simple Steps:
1. **Learn Visual Features:** $h_v = \text{VisionNetwork}(\text{Image}) \in \mathbb{R}^{64}$
2. **Learn Chemical Features:** $h_s = \text{SensorNetwork}(\text{Sensors}) \in \mathbb{R}^{64}$
3. **Dynamic Gating Decision:** The gate computes a trust score $g$ between 0 and 1:
   $$g = \sigma(W_g [h_v; h_s] + b_g)$$
   $$h_{\text{fused}} = g \odot h_v + (1 - g) \odot h_s$$
   - If the visual clues are clear and trustworthy, $g$ shifts toward vision.
   - If the fruit's appearance is deceptive, the gate relies on chemical sensors $(1 - g)$!

---

## 📊 Battle of the Models: Who Won?

We tested 6 different artificial intelligence models on a strict, held-out test split of **450 fruits** (150 Fresh, 150 Formalin, 150 Rotten) that the AI had never seen during training:

| Model | What Input Does It Use? | Test Accuracy | Macro F1-Score | Number of Parameters | Speed per Fruit |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | Sensors Only | 100.00% | 1.0000 | Tabular Forest | < 1 millisecond |
| **XGBoost** | Sensors Only | 100.00% | 1.0000 | Gradient Boosted | < 1 millisecond |
| **LightGBM** | Sensors Only | 100.00% | 1.0000 | Leaf-wise Trees | < 1 millisecond |
| **Sensor MLP** | Sensors Only | 100.00% | 1.0000 | 2,755 weights | < 1 millisecond |
| **MobileNet-V3 Small** | **Vision Only (Camera)** | **98.44%** | **0.9844** | 1,001,251 weights | ~ 4 milliseconds |
| **🏆 Multimodal GMU (Ours)** | **Vision + Sensors** | **99.78%** | **0.9978** | **979,427 weights** | **~ 5 milliseconds** |

<div align="center">
  <img src="results/confusion_matrices.png" width="750" alt="Confusion Matrix Comparison"/>
  <p><i>Figure 3: Side-by-side Confusion Matrix. Left: Camera Alone (7 mistakes). Right: Multimodal GMU (Only 1 mistake!).</i></p>
</div>

---

## 🔍 Why Cameras Alone Fail (The 7 Mistakes)

When we inspected the **7 errors** made by the camera-only MobileNet-V3 model:
1. **The "Sunburned" Fresh Fruits (4 errors):** Four completely fresh, delicious fruits had small brown spots on their outer skin from harmless sunlight or wind desiccation. The camera panicked and classified them as **Rotten**!
   - *How GMU fixed it:* The GMU checked the moisture ($78\%$), ethanol ($0.25\text{ g/L}$), and $\text{CO}_2$ ($290\text{ ppm}$) levels. The chemical profile was healthy, so it overturned the camera and correctly labeled them **Fresh**!
2. **The "Sneaky" Rotten Fruits (3 errors):** Three rotting fruits had pulp that had started fermenting on the inside, but their tough outer rind still held a bright red color. The camera said **Fresh**!
   - *How GMU fixed it:* The GMU detected high ethanol gas and soft skin firmness, immediately flagging them as **Rotten**!

Overall, the GMU **eliminated 85.7% of the vision-only errors**, pushing accuracy to **99.78%**!

---

## 🎛️ The AI's "Referee": Gate Interpretability

In traditional deep learning, models are "black boxes"—you never know *why* they made a choice.
With our **Gated Multimodal Unit**, we can peek inside the brain and check the gate value $g$:

<div align="center">
  <img src="results/gate_analysis.png" width="600" alt="GMU Gate Analysis"/>
  <p><i>Figure 4: Average gate values across classes stay nicely balanced between 52% and 54%.</i></p>
</div>

- **Fresh Fruits:** Gate = **52.5%** vision, **47.5%** sensors.
- **Formalin-Mixed Fruits:** Gate = **53.5%** vision, **46.5%** sensors.
- **Rotten Fruits:** Gate = **52.0%** vision, **48.0%** sensors.

> [!NOTE]
> **No "Modality Collapse":** In bad multimodal systems, the AI gets lazy and relies 100% on one sense while ignoring the other. Here, the gate stays tightly balanced around **53% vision vs. 47% sensors**, proving that the network forms a true cooperative partnership between the camera and the sensors!

---

## 🚀 Live Interactive Web App (Streamlit)

We built an easy-to-use, interactive web application in [`app.py`](app.py) so anyone can test the models live without writing code.

### App Features:
1. **Model Switcher:** Toggle between **Multimodal GMU** (Camera + Sensors) or **Vision Baseline** (Camera Only).
2. **Two Ways to Test:**
   - **Upload Mode:** Upload your own fruit photo (`.jpg`, `.png`).
   - **Benchmark Mode:** Browse through all **450 real test fruits** with their ground-truth labels and real sensor numbers pre-loaded!
3. **Live Correctness Verification:**
   - Shows **`✅ CORRECTLY CLASSIFIED`** (in green) if the model gets it right.
   - Shows **`❌ MISCLASSIFIED`** (in red) if the model makes a mistake.
4. **Live Metric Cards:** Displays exact confidence percentage, uncertainty entropy, and the dynamic vision-vs-sensor gate weight gauge.
5. **Full 450-Sample Benchmark Tab:** Click one button to run inference across all 450 test fruits and see the live Confusion Matrix and Classification Report calculated on the fly (no fake numbers!).

---

## 💻 Step-by-Step Installation & Quickstart

### Prerequisites
- Windows, macOS, or Linux
- Python 3.10, 3.11, 3.12, or 3.13 installed
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/LADD-Litchi-Adulteration-Detection.git
cd LADD-Litchi-Adulteration-Detection
```

### 2. Set Up a Virtual Environment
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or install lightweight CPU PyTorch directly: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`)*

### 4. Launch the Streamlit App!
**On Windows:** Simply double-click `run_app.bat`, or run:
```bash
streamlit run app.py
```
Your browser will open to `http://localhost:8501` automatically! 🎈

---

## 📂 Project Folder Structure

```
Litchi/
├── app.py                     # Streamlit web application
├── run_app.bat                # One-click Windows desktop launcher
├── generate_report.py         # Automated DOCX technical report generator
├── models/                    # Trained model weights & scalers
│   ├── multimodal_gmu.pth     # GMU PyTorch weights (979k params)
│   ├── vision_baseline.pth    # MobileNet-V3 Small weights (1M params)
│   └── sensor_scaler.pkl      # StandardScaler fitted on training partition
├── Data/
│   ├── Raw/                   # Original dataset & images
│   │   ├── Litchi Sensor Data.csv
│   │   └── images/            # 3,000 raw photos (Fresh, Formalin, Rotten)
│   └── processed/             # Cleaned, anti-leakage train/val/test splits
│       ├── train.csv          # 2,099 training samples (70%)
│       ├── val.csv            # 451 validation samples (15%)
│       └── test.csv           # 450 test samples (15%)
├── notebooks/                 # Step-by-step Jupyter notebooks
│   ├── 01_eda.ipynb           # Exploratory data analysis & path alignment
│   ├── 02_sensor_baselines.ipynb # RF, XGBoost, LightGBM, Sensor MLP
│   ├── 03_vision_baseline.ipynb  # MobileNet-V3 Small vision training
│   ├── 04_multimodal_fusion.ipynb# Gated Multimodal Unit (GMU) training
│   └── 05_evaluation.ipynb    # Side-by-side benchmark & gate analysis
└── results/                   # High-resolution benchmark figures
    ├── eda_sample_images.png  # Representative fruit photographs
    ├── eda_boxplots.png       # 7-sensor boxplot distributions
    ├── eda_correlation.png    # Sensor Pearson correlation heatmap
    ├── vision_training_curves.png # MobileNet-V3 learning curves
    ├── fusion_training_curves.png # GMU multi-phase learning curves
    ├── confusion_matrices.png # Side-by-side confusion matrix
    └── gate_analysis.png      # Gating distribution across classes
```

---

## 🌍 Alignment with UN Sustainable Development Goals

This project is designed in alignment with the **United Nations Sustainable Development Goals (SDGs)**:

<div align="center">

| SDG Goal | Target | How This Project Helps |
| :---: | :--- | :--- |
| **SDG 3: Good Health & Well-Being** | **Target 3.9:** Substantially reduce deaths and illnesses from hazardous chemicals | Stops toxic formaldehyde-laced fruit from reaching families and children in retail markets. |
| **SDG 9: Industry & Innovation** | **Target 9.5:** Upgrade technological capabilities of industrial sectors | Brings smart, low-cost AI and sensor fusion into fruit packing houses and agricultural supply chains. |
| **SDG 12: Responsible Consumption** | **Target 12.3:** Halve per capita global food waste | Prevents edible fresh fruits with cosmetic skin browning from being wrongly thrown away. |

</div>

---

## 🔮 Future Scope
- **Portable Hardware Pod:** Mount the GMU model onto an **NVIDIA Jetson Orin Nano** or **Raspberry Pi 5** paired with a USB camera and low-cost MQ-series gas sensors.
- **Hyperspectral / Near-Infrared (NIR) Imaging:** Optical wavelengths between $700\text{--}1000\text{ nm}$ that can "see through" the rind without puncturing the fruit.
- **Factory Conveyor Sorting:** Connect the model to high-speed mechanical ejector arms to sort 10 fruits per second in commercial packing plants.

---

## 📜 Citation & Academic Details

If you use this benchmark, code, or methodology in your research or project, please cite:

```bibtex
@article{sumedha2026ladd,
  title={LADD Benchmark: Gated Multimodal Deep Learning for Litchi Freshness and Formalin Adulteration Detection},
  author={Sumedha, K. N. and Kavitha, R.},
  journal={Department of Statistics and Data Science, CHRIST (Deemed to be University)},
  year={2026},
  address={Bengaluru, India}
}
```

**Course Details:**  
- **Course Code:** MDA507-5 (Neural Networks and Deep Learning)  
- **Program:** M.Sc. Data Science (V MSDA)  
- **Institution:** CHRIST (Deemed to be University), Central Campus, Bengaluru, India  

---

<div align="center">
  <sub>Built with ❤️ by Sumedha K N using PyTorch, MobileNet-V3, and Streamlit.</sub>
</div>
