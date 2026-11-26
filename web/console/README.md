# EFM Service Assistant Console

This Vite/React app is self-contained and can be copied or published as its own
repository. It uses the dev-only `X-Tenant-ID` and `X-Roles` headers and the
same-origin API paths `/api/v1/chat/stream` and `/api/v1/ingest`.

## Local development

Start the API on `http://127.0.0.1:8000` with `ENV=dev`, then run:

```sh
npm install
npm run dev
```

Open the URL printed by Vite. Its `/api` requests proxy to the local API.

## Production build

```sh
npm ci
npm run build
```

The standalone bundle is written to `dist/`. The repository's FastAPI app
serves that build at `/console/` when `dist/` exists; the React app can instead
be deployed separately behind a same-origin `/api` proxy.
