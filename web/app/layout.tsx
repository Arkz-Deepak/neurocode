import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroCode AI | On-Device Autonomous Software Engineering",
  description: "In-Memory Code Property Graph Reachability Slicing under Edge VRAM Constraints. 97.1% prompt token reduction, 0.62ms reachability closure, zero OOM crashes on consumer GPUs.",
  keywords: [
    "NeuroCode",
    "Autonomous Software Engineering",
    "Code Property Graph",
    "AST reachability slicing",
    "Edge VRAM constraints",
    "KV-cache memory wall",
    "SWE-bench Lite",
    "Model Context Protocol",
    "MCP server",
    "VS Code extension",
    "Deepak R",
    "Elsevier JPDC",
    "ICPC 2027"
  ],
  authors: [{ name: "Deepak R.", url: "https://www.deepak-arkz.me" }],
  creator: "Deepak R.",
  publisher: "Deepak R.",
  metadataBase: new URL("https://neurocode.deepak-arkz.me"),
  alternates: {
    canonical: "/",
  },
  openGraph: {
    title: "NeuroCode AI | Heterogeneous Edge Code-Graph Slicer",
    description: "97.1% token reduction on SWE-bench Lite. Physical edge GPU execution under 4GB VRAM.",
    url: "https://neurocode.deepak-arkz.me",
    siteName: "NeuroCode AI",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "NeuroCode AI | On-Device Autonomous Software Engineering",
    description: "97.1% token reduction, zero OOM crashes, sub-millisecond graph reachability on edge hardware.",
    creator: "@Arkz_Deepak",
  },
  robots: {
    index: true,
    follow: true,
    nocache: false,
    googleBot: {
      index: true,
      follow: true,
      noimageindex: false,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const jsonLd = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "SoftwareApplication",
        "name": "NeuroCode AI",
        "operatingSystem": "Windows, Linux, macOS",
        "applicationCategory": "DeveloperApplication",
        "offers": {
          "@type": "Offer",
          "price": "0",
          "priceCurrency": "USD"
        },
        "description": "In-Memory Code Property Graph Reachability Slicing under Edge VRAM Constraints. 97.1% prompt token reduction.",
        "author": {
          "@type": "Person",
          "name": "Deepak R.",
          "url": "https://www.deepak-arkz.me"
        },
        "url": "https://neurocode.deepak-arkz.me",
        "downloadUrl": "https://neurocode.deepak-arkz.me/downloads/NeuroCode-Setup.exe"
      },
      {
        "@type": "ScholarlyArticle",
        "headline": "On-Device Autonomous Software Engineering: In-Memory Code Property Graph Reachability Slicing under Edge VRAM Constraints",
        "author": {
          "@type": "Person",
          "name": "Deepak R."
        },
        "publisher": {
          "@type": "Organization",
          "name": "Elsevier / IEEE"
        },
        "url": "https://neurocode.deepak-arkz.me"
      }
    ]
  };

  return (
    <html lang="en" className="dark">
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body className="antialiased min-h-screen bg-[#070b14] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
