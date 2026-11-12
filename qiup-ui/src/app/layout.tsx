import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "QIUP Simulator · v3.4.1 (prototype / pre-validation)",
  description:
    "Quantum imaging with undetected photons — four-arrangement prototype simulator UI (not experimentally calibrated)",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
