# nlp_a7

```markdown
# Hate Speech Detector

This repository implements a hate speech detection system using BERT-based models, comparing knowledge distillation and LoRA approaches. The best-performing model (odd-layer distilled) is deployed via a Streamlit web app.

[Performance Plot](Screenshot (501).png)
[Performance Plot](Screenshot (502).png)
---

## Project Overview

- **Objective**: Classify tweets as "Non-Hate", "Offensive", or "Hate".
- **Dataset**: HateXplain (`hate_speech_offensive`) from Hugging Face (~24,783 samples).
- **Models**:
  - **Teacher**: BERT-base-uncased (12 layers, 3 labels).
  - **Odd-Layer Student**: 6-layer BERT distilled from odd layers (0, 2, 4, 6, 8, 10).
  - **Even-Layer Student**: 6-layer BERT distilled from even layers (1, 3, 5, 7, 9, 11).
  - **LoRA Student**: 6-layer BERT with LoRA (r=8, targeting `query`, `value`).
- **Training**: 
  - Distillation: Cross-entropy + KL divergence + cosine embedding loss.
  - LoRA: Standard training without distillation.
  - 5 epochs, batch size 32, AdamW (lr=5e-5), linear scheduler.
- **Results**:
  - Odd-Layer: **0.9660** test accuracy (deployed).
  - Even-Layer: 0.9630 test accuracy.
  - LoRA: 0.7730 test accuracy.
- **Deployment**: Streamlit app with the odd-layer model.

---

## Repository Structure

```
nlp_a7/
├── app.py                  # Streamlit app script
├── requirements.txt        # Dependencies for deployment
├── student_odd.pth         # Odd-layer model weights
├── student_even.pth        # Even-layer model weights
├── student_lora.pth        # LoRA model weights
├── special_tokens_map.json # Tokenizer file
├── tokenizer_config.json   # Tokenizer file
├── vocab.txt              # Tokenizer file
├── download (9).png        # Performance plot (optional)
├── nlp_a7.ipynb            # Training script
├── README.md               # Project documentation
```

---

## Prerequisites

- Python 3.11+
- CUDA-enabled GPU (optional, recommended for training)
- Git

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/your-username/nlp_a7.git
cd nlp_a7
```

### Install Dependencies
#### For Training
```bash
pip install torch transformers datasets peft sklearn tqdm matplotlib
```

#### For Streamlit App
```bash
pip install -r requirements.txt
```

---

## Training

1. **Run the Training Script**:
   - Open `nlp_a7.ipynb` in Jupyter Notebook or Google Colab.
   - Execute all cells to:
     - Load and preprocess the dataset.
     - Train the three models.
     - Evaluate on test data.
     - Save models and tokenizer.

2. **Outputs**:
   - `.pth` files: `student_odd.pth`, `student_even.pth`, `student_lora.pth`.
   - Tokenizer files: Saved in the root directory.
   - Plot: Displayed (save manually as `performance_plot.png` if desired).

3. **Save Plot** (optional):
   ```python
   plt.savefig("performance_plot.png")
   ```

---

## Local Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   - Ensure files are in `C:\Users\HP\Downloads\nlp_a7-main\nlp_a7-main\` (or update paths in `app.py`).
   - Access at `http://localhost:8501`.

2. **Customize Paths** (if needed):
   ```python
   model_path = r"C:\path\to\student_odd.pth"
   tokenizer_path = r"C:\path\to\tokenizer\files"
   plot_path = r"C:\path\to\download (9).png"
   ```

3. **Test**:
   - Input: "I hate you"
   - Expected Output: "Hate" (or similar, based on training).

---

## Deployment on Streamlit Cloud

### Prepare Files
Ensure the following are in the repo root:
```
app.py
requirements.txt
student_odd.pth
special_tokens_map.json
tokenizer_config.json
vocab.txt
download (9).png  # Or performance_plot.png
```

### Push to GitHub
```bash
git add .
git commit -m "Add app and model files"
git push origin main
```
For large `.pth` files (>100 MB):
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes *.pth
git commit -m "Track .pth files with LFS"
git push origin main
```

### Deploy
1. Visit [Streamlit Community Cloud](https://streamlit.io/cloud).
2. Connect your GitHub repo.
3. Set branch to `main` and main file to `app.py`.
4. Deploy and access the app URL.

---

## Usage

- **Interface**: Enter text in the provided text area.
- **Prediction**: Click "Classify" to see if the text is "Non-Hate", "Offensive", or "Hate".
- **Plot**: Displays model performance (if `download (9).png` is the correct plot).

---

## Results

| Model         | Test Accuracy |
|---------------|---------------|
| Odd-Layer     | 0.9660        |
| Even-Layer    | 0.9630        |
| LoRA          | 0.7730        |

The odd-layer model is deployed due to its superior performance.

![Performance Plot](download (9).png)
## Model Performance

| Model Type    | Training Loss (Epoch 5) | Test Set Performance |
|---------------|-------------------------|----------------------|
| Odd-Layer     | 0.2931                  | 0.9660              |
| Even-Layer    | 0.2940                  | 0.9630              |
| LoRA          | 0.5243                  | 0.7730              |

- **Training Loss**: Average loss from the final epoch (Epoch 5).
- **Test Set Performance**: Accuracy on a 1000-sample test split.


## Troubleshooting

- **Model Loading Error**:
  - Check `student_odd.pth` compatibility with `BertForSequenceClassification`.
  - `strict=False` in `load_state_dict` handles missing keys (e.g., `position_ids`).
- **Plot Error**:
  - Verify `download (9).png` exists or rename to `performance_plot.png`.
- **Dependency Conflicts**:
  - Align versions with `requirements.txt` (e.g., `torch==2.6.0`, `transformers==4.48.3`).

---
