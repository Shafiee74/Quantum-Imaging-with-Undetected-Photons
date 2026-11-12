/**
 * Former Node→Python spawn bridge.
 * The simulator HTTP API is now served by Django (`qiup_web`).
 * Next.js only proxies `/api/*` to Django (see next.config.mjs).
 */
export const QIUP_BACKEND = "django";
