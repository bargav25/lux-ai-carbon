
# Running in Local

## Environment Setup

### Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/bargav-bunny/lux-ai-carbon.git
cd lux-ai-carbon
```

### Create a Python Virtual Environment

Create a virtual environment to manage the project's dependencies separately:

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows:**

  ```bash
  .\venv\Scripts\activate
  ```

- **macOS and Linux:**

  ```bash
  source venv/bin/activate
  ```

### Install Required Packages

Install all necessary Python packages using the `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data Setup

### Download Training Data

Download the training data directly from Google Drive link (https://drive.google.com/drive/folders/1AungbiFTCkO1bsxIAOiOPrGkgm5Qtvd1?usp=drive_link) and store it in the `top_agents` directory within your project:


## Running the Training Script

With the environment and data ready, you can run the training script as follows:

```bash
python train.py
```

This command will start the training process, and the resulting model file will be saved within your project directory.

## Conclusion

Ensure that each step is followed carefully to set up the project correctly. If you run into any issues, We have the notebook as well, which is highly recommended to run in Kaggle notebooks.

