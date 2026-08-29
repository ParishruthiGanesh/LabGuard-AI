import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LabGuard AI",
  description:
    "Autonomous research reliability platform: challenge the claim, protect the run, trust the result.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">{children}</body>
    </html>
  );
}
