/** @type {import("next").NextConfig} */
const DJANGO_ORIGIN = process.env.QIUP_DJANGO_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // Frontend-only Next.js: proxy /api/* to the Django backend (no Node simulator bridge).
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${DJANGO_ORIGIN}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
