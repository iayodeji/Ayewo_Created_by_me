# Ayewo Frontend

React/Vite interface for the Ayewo FastAPI malaria screening API.

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

The default API address is `http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` in `.env` when the API is deployed elsewhere.

Run `npm run build` to create the deployable `dist` directory.
