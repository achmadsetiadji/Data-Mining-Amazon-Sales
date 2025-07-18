# Amazon Product Rating Prediction App

A Streamlit web application for predicting Amazon product ratings using Random Forest Regression, with interactive data exploration and business insights. Built for educational purposes (Data Mining coursework at Universitas Indraprasta PGRI).

---

## Features
- Predict Amazon product ratings based on product features and reviews
- Interactive data exploration and visualization
- Compare machine learning methods
- Business insights and summary tab

---

## Directory Structure
```
├── app.py                  # Main Streamlit app entry point
├── app/
│   ├── __init__.py
│   ├── config.py           # App configuration (logo path, title)
│   ├── ml.py               # Machine learning logic
│   ├── utils.py            # Data loading, cleaning, logo display
│   ├── visual.py           # Visualization and tab logic
├── amazon.csv              # Main dataset (required)
├── image/
│   └── logo_unindra.png    # Logo image (required)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.dev.yml  # Docker Compose for local development
├── docker-compose.prod.yml # Docker Compose for production
```

---

## Requirements
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/)
- `amazon.csv` dataset in the project root (sourced from Kaggle, see below)
- `image/logo_unindra.png` logo image

---

## Dataset Source
- The dataset used in this project is sourced from Kaggle: [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
- Download the dataset from Kaggle and place `amazon.csv` in the project root directory.

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/achmadsetiadji/Data-Mining-Amazon-Sales.git
cd Data-Mining-Amazon-Sales
```

### 2. Prepare Required Files
- Ensure `amazon.csv` is present in the project root.
- Ensure `image/logo_unindra.png` exists.

### 3. Local Development (with Hot Reload)
```bash
docker-compose -f docker-compose.dev.yml up --build
```
- Access the app at: [http://localhost:8501](http://localhost:8501)
- Changes to code will auto-reload.

### 4. Production Deployment
```bash
docker-compose -f docker-compose.prod.yml up --build -d
```
- Access the app at: [http://localhost:8501](http://localhost:8501)
- Runs in detached mode, optimized for production.

### 5. Stopping the App
```bash
docker-compose -f docker-compose.prod.yml down
# or for dev
docker-compose -f docker-compose.dev.yml down
```

---

## Configuration
- **Logo Path:** Set in `app/config.py` as `image/logo_unindra.png`.
- **Dataset:** The app expects `amazon.csv` in the project root with appropriate columns (see code for details).

---

## Usage
- The sidebar provides project info and data status.
- Use the navigation tabs for summary, data exploration, prediction, and method comparison.

---

## Troubleshooting
- **FileNotFoundError:** Ensure `amazon.csv` and `image/logo_unindra.png` exist in the correct locations.
- **Port 8501 in use:** Stop other Streamlit apps or change the port in the compose file.
- **Data loading errors:** Check the CSV format and required columns.

---

## License / Attribution
- This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
- Dataset: [Amazon Sales Dataset on Kaggle](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
- Developed for Data Mining coursework at Universitas Indraprasta PGRI.

---

## MIT License

```
MIT License

Copyright (c) 2024 Achmad Setiadji

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
``` 