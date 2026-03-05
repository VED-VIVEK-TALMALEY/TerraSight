# 🚀 QUICK START - WEB APP

## ONE-TIME SETUP (5 minutes)

### 1. Install Backend Dependencies
```powershell
.\venv\Scripts\Activate.ps1
pip install fastapi uvicorn strawberry-graphql[fastapi] python-multipart
```

### 2. Setup Frontend
```powershell
mkdir frontend
cd frontend
npx create-react-app .
npm install @apollo/client graphql react-leaflet leaflet
```

### 3. Copy Files
- `backend_graphql.py` → project root
- `frontend_App.js` → `frontend/src/App.js`
- `frontend_App.css` → `frontend/src/App.css`

---

## EVERY TIME YOU DEMO (30 seconds)

### Terminal 1 - Backend:
```powershell
cd C:\Users\talma\Desktop\EO-project\EO-GPTOSS
.\venv\Scripts\Activate.ps1
python backend_graphql.py
```
**Wait for:** "✓ Models loaded"

### Terminal 2 - Frontend:
```powershell
cd C:\Users\talma\Desktop\EO-project\EO-GPTOSS\frontend
npm start
```
**Browser opens automatically!**

---

## USING THE APP

1. **Click anywhere** on the map
2. **Wait 2-3 seconds** (analyzing...)
3. **See results:**
   - Baseline (red) vs Your Model (green)
   - Spectral keywords
   - NDVI predictions
   - Improvement metrics

---

## TROUBLESHOOTING

**Backend not starting?**
→ Check `checkpoints/best_model.pt` exists

**Frontend errors?**
→ Run: `npm install` in frontend directory

**No results showing?**
→ Check Terminal 1 for backend errors

---

## URLs

- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- GraphQL Playground: http://localhost:8000/graphql

---

**THAT'S IT! Enjoy your demo!** 🎉
