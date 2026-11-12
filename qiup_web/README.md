# QIUP web API (Django)

Django replaces the former Next.js/Node Python-spawn bridge. The simulator
logic remains in `qiup_ui_backend.py` at the repository root.

## Pages / endpoints

| Method | Path | Response |
|---|---|---|
| `GET` | `/` | Simulator UI (Django templates + static JS) |
| `GET` | `/api/defaults` | JSON profile/defaults document |
| `POST` | `/api/preview` | one `object` event |
| `POST` | `/api/run` | NDJSON event stream |

## Run

```bash
pip install -r requirements.txt
python qiup_web/manage.py runserver 8000
```

Open **http://127.0.0.1:8000/** — the UI and API both run on Django.

## Tests

```bash
python qiup_web/manage.py test api ui -v 2
python qiup-ui/scripts/smoke_test.py http://127.0.0.1:8000
```
