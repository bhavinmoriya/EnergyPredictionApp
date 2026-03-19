Deploying your FastAPI + Streamlit/HTML app to **Render.com** is straightforward. Here’s a step-by-step guide to get your app live and accessible 24/7:

---

## **Step 1: Prepare Your Project**
1. **Project Structure**
   Ensure your project has this structure:
   ```
   your_project/
   ├── main.py          # FastAPI app
   ├── static/          # HTML/CSS/JS files
   │   └── index.html
   ├── requirements.txt # Python dependencies
   └── .gitignore
   ```

2. **Create `requirements.txt`**
   Run this command to generate a `requirements.txt` file:
   ```bash
   uv pip freeze > requirements.txt
   ```
   Or manually create the file with:
   ```
   fastapi
   uvicorn
   python-multipart
   polars
   xgboost
   scikit-learn
   matplotlib
   numpy
   ```

3. **Create a `.gitignore` File**
   Add this to ignore unnecessary files:
   ```
   .venv/
   __pycache__/
   *.pyc
   *.pyo
   *.pyd
   .env
   ```

4. **Commit to GitHub**
   Push your project to a GitHub repository.

---

## **Step 2: Set Up Render.com**

### **1. Sign Up / Log In**
- Go to [render.com](https://render.com) and sign up (or log in).

### **2. Create a New Web Service**
- Click **"New"** → **"Web Service"**.
- Connect your GitHub account and select your repository.

### **3. Configure the Service**
- **Name**: Give your service a name (e.g., `energy-forecast-app`).
- **Region**: Choose the closest region to your users.
- **Branch**: Select the branch to deploy (e.g., `main`).
- **Root Directory**: Leave blank if your files are in the root.
- **Environment**: Python 3.
- **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port $PORT
  ```
  *(Render automatically sets the `$PORT` environment variable.)*

### **4. Environment Variables**
- If your app uses environment variables (e.g., API keys), add them under **"Environment"** in the Render dashboard.

### **5. Deploy**
- Click **"Create Web Service"**.
- Render will build and deploy your app. This may take a few minutes.

---

## **Step 3: Access Your Live App**
- Once deployed, Render will provide a URL (e.g., `https://energy-forecast-app.onrender.com`).
- Open this URL in your browser to see your app live!

---

## **Step 4: (Optional) Custom Domain**
If you want a custom domain (e.g., `forecast.yourdomain.com`):
1. Go to your Render dashboard.
2. Navigate to your service → **"Custom Domains"**.
3. Add your domain and follow the instructions to configure DNS.

---

## **Troubleshooting**
- **Build Fails**: Check the logs in the Render dashboard for errors. Common issues include missing dependencies or incorrect file paths.
- **App Crashes**: Ensure your `requirements.txt` is up-to-date and all dependencies are listed.
- **Static Files Not Loading**: Double-check the paths in your FastAPI app (e.g., `app.mount("/static", StaticFiles(directory="static"), name="static")`).

---

## **Example `main.py` for Render**
Ensure your `main.py` is set up to serve static files and handle the root route:
```python
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML page at the root
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("static/index.html")

# Your existing endpoints (/upload, /predict, /visualize, etc.)
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    # Your upload logic here
    return {"message": "File uploaded and model trained!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## **Final Notes**
- **Free Tier**: Render offers a free tier, but your app will spin down after inactivity. Upgrade to a paid plan for 24/7 availability.
- **Scaling**: Render automatically scales your app based on traffic.
- **Logs**: Use the Render dashboard to monitor logs and debug issues.

---
Your app is now live and accessible to anyone, anywhere! 🎉🚀
Open your Render URL and start sharing it with your users.
