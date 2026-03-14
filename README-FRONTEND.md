# StyleShield + React (`frontend`)

## 1. Python API

```bash
pip install -r api/requirements.txt
pip install numpy pandas scikit-learn
cd api
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## 2. React app

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (default **http://localhost:5174**) → upload CSV → **Run StyleShield**.

## CSV format

Same as `Styleshield_script.py`: long format with `account_id`, `post_text`, optional `posting_hour`.
