<h1>ISIC Skin Lesion Classification with PyTorch</h1>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.9+</li>
  <li>Optional: NVIDIA GPU for faster training</li>
</ul>

<h2>Installation</h2>
<h3>1. Clone the repository</h3>

```
git clone <the-repo-url>
cd Med_AI_Sys
```

<h3>2. Create a virtual environment</h3>

```
python -m venv venv
```

<h3>3. Activate the virtual environment</h3>

```
venv\Scripts\activate       (Windows)
source venv/bin/activate     (Mac/Linux)
```

<h3>4. Install libraries</h3>

```
pip install -r requirements.txt
```

<h3>5. Configure environment variables</h3>

* Copy `.env.example` to `.env`

```
copy .env.example .env      (Windows)
cp .env.example .env        (Mac/Linux)
```

* Edit `.env` to set the correct paths for your data, images, splits, plots, and models.

<h2>Executing the scripts</h2>

* Open a terminal in VS Code

* Run the data preparation / splitting script:

```
python prepare_splits.py
```

* Run exploratory data analysis:

```
python eda.py
```

* Train the model:

```
python train_classification.py
```

* The trained model will be saved in `models/resnet18_isic.pth` and plots in `data/plots/`

