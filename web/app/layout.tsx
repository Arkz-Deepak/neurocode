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
  verification: {
    google: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION || "google-site-verification-token",
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
        "description": "In-Memory Code Property Graph Reachability Slicing under Edge VRAM Constraints. 97.1% prompt token reduction, zero OOM crashes on consumer GPUs.",
        "author": {
          "@type": "Person",
          "name": "Deepak R.",
          "url": "https://www.deepak-arkz.me"
        },
        "url": "https://neurocode.deepak-arkz.me",
        "downloadUrl": "https://neurocode.deepak-arkz.me/downloads/NeuroCode-Setup.exe",
        "featureList": [
          "In-Memory Code Property Graph (AST + CFG + DFG)",
          "Sub-1500 token bounded reachability slicing in 0.62ms",
          "97.1% token compression on SWE-bench Lite",
          "Transparent local OpenAI proxy on port 8000",
          "Auto-configuring VS Code & Cursor MCP extension"
        ]
      },
      {
        "@type": "FAQPage",
        "mainEntity": [
          {
            "@type": "Question",
            "name": "How to run local coding LLMs on 4GB GPU without CUDA OOM crashes?",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "NeuroCode offloads code graph construction to host CPU/DRAM using Tree-sitter and NetworkX, computing bidirectional reachability closures in 0.62ms to bound prompts under 1,500 tokens. This reduces peak KV-cache from 14.2 GB to ~400 MB, preventing CUDA Out-Of-Memory errors on 4GB GPUs like the NVIDIA RTX 3050."
            }
          },
          {
            "@type": "Question",
            "name": "How to reduce token usage and API costs in Cursor or Claude Desktop?",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "NeuroCode provides a transparent local OpenAI proxy (neurocode proxy --port 8000) and an auto-configuring MCP server for Cursor and Claude Desktop. By replacing whole-file reading with syntactically bounded AST slices, it cuts median prompt token consumption from 48,000 tokens to 1,380 tokens (a 97.1% reduction)."
            }
          },
          {
            "@type": "Question",
            "name": "Where can I download NeuroCode?",
            "acceptedAnswer": {
              "@type": "Answer",
              "text": "NeuroCode is available as a native Windows installer (NeuroCode-Setup.exe), a VS Code/Cursor extension (.vsix), and on PyPI via 'pip install neurocode-ai'."
            }
          }
        ]
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
          "name": "IEEE / ACM ICPC 2027"
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
