import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroCode AI | On-Device Autonomous Software Engineering",
  description: "In-Memory Code Property Graph Reachability Slicing under Edge VRAM Constraints. 97.1% token reduction, zero OOM crashes, sub-millisecond graph reachability.",
  openGraph: {
    title: "NeuroCode AI | Heterogeneous Edge Code-Graph Slicer",
    description: "97.1% token reduction on SWE-bench Lite. Physical edge GPU execution under 4GB VRAM.",
    url: "https://neurocode.deepak-arkz.me",
    siteName: "NeuroCode AI",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased min-h-screen bg-[#070b14] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
