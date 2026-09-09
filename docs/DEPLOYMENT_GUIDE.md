# VisionInspect AI — Render & Vercel Cloud Deployment Guide

This guide provides step-by-step instructions for deploying:
1. **Backend (FastAPI + OpenCV)** to **[Render.com](https://render.com)**
2. **Database (MongoDB Atlas)** to **[MongoDB Atlas](https://www.mongodb.com/cloud/atlas)**
3. **Frontend (React + Vite)** to **[Vercel.com](https://vercel.com)**

---

## 🛢️ Step 1: MongoDB Database (MongoDB Atlas)

1. Sign up for a free account at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas).
2. Create a **Free Shared Cluster (M0)**.
3. Under **Database Access**, create a Database User (e.g. username: `visionuser`, password: `your_password`).
4. Under **Network Access**, add IP `0.0.0.0/0` (allow connection from anywhere).
5. Click **Connect** $\rightarrow$ **Drivers** $\rightarrow$ Copy the connection string:
   `mongodb+srv://visionuser:<password>@cluster0.mongodb.net/visioninspect_ai?retryWrites=true&w=majority`

---

## 🚀 Step 2: Deploy Backend to Render.com

1. Sign up/Log in at [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** $\rightarrow$ Select **Web Service**.
3. Connect your GitHub repository: `https://github.com/maheshgutha/visioninspect-ai.git`.
4. Configure service settings:
   - **Name**: `visioninspect-backend`
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add **Environment Variables** under the Environment tab:
   - `MONGO_URI`: *Your MongoDB Atlas connection string from Step 1*
   - `DB_NAME`: `visioninspect_ai`
   - `JWT_SECRET`: `your_random_secret_key_string_here`
   - `JWT_ALGORITHM`: `HS256`
   - `ACCESS_TOKEN_EXPIRE_MINUTES`: `1440`
   - `UPLOAD_DIR`: `/tmp/uploads`
   - `CORS_ORIGINS`: `*`
6. Click **Create Web Service**.
7. Once deployed, Render will give you your backend URL:
   `https://visioninspect-backend.onrender.com`
   *(Test health check at: `https://visioninspect-backend.onrender.com/api/health`)*

---

## ⚡ Step 3: Deploy Frontend to Vercel.com

1. Sign up/Log in at [vercel.com](https://vercel.com).
2. Click **Add New...** $\rightarrow$ **Project**.
3. Import your GitHub repository: `https://github.com/maheshgutha/visioninspect-ai.git`.
4. Configure project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: Select `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. Expand **Environment Variables**:
   - `VITE_API_BASE_URL`: `https://visioninspect-backend.onrender.com` *(Use your Render backend URL)*
6. Click **Deploy**.
7. Vercel will build and host your production frontend application at:
   `https://visioninspect-ai.vercel.app`

---

## 🔒 Step 4: Final CORS Verification

After getting your Vercel frontend URL (`https://visioninspect-ai.vercel.app`), update the `CORS_ORIGINS` environment variable on Render to include your Vercel URL to secure API calls!
