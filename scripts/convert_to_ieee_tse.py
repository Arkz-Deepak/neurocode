#!/usr/bin/env python3
"""
Converts MLSys LaTeX paper to IEEE Transactions on Software Engineering (TSE) / ASE IEEEtran format.
"""

from pathlib import Path
import re

mlsys_path = Path("paper_mlsys/neurocode_mlsys.tex")
tse_path = Path("paper_ieee_tse/neurocode_tse.tex")

with open(mlsys_path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace documentclass and packages
ieee_header = r"""\documentclass[journal]{IEEEtran}

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}
\usepackage{microtype}
\usepackage{balance}

\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}

\begin{document}

\title{NeuroCode: Heterogeneous Host-Device Co-Scheduling for Memory-Bounded Repository-Scale Software Engineering Agents}

\author{Deepak~R.,~\IEEEmembership{Student Member,~IEEE}%
\thanks{Deepak R. is with the Department of Robotics and Automation Engineering, Dhaanish Ahmed College of Engineering, Anna University, Chennai 601301, India (e-mail: wssedd18@gmail.com).}%
}

\markboth{IEEE Transactions on Software Engineering,~Vol.~XX, No.~X,~November~2026}%
{Deepak R.: NeuroCode: Heterogeneous Host-Device Co-Scheduling for Memory-Bounded SWE Agents}

\maketitle
"""

# Extract abstract
abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL)
abstract_content = abstract_match.group(1).strip() if abstract_match else ""

ieee_abstract = f"""
\\begin{{abstract}}
{abstract_content}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
Autonomous software engineering, Large Language Models, Code Property Multigraph, edge computing, program repair, Key-Value cache, hardware-software co-design.
\\end{{IEEEkeywords}}
"""

# Extract body (from \section{Introduction} to \bibliography)
body_match = re.search(r"(\\section\{Introduction\}.*?)\\bibliography\{neurocode\}", text, re.DOTALL)
if body_match:
    body_content = body_match.group(1).strip()
else:
    raise ValueError("Could not extract body content")

# Adjust first word for \IEEEPARstart
body_content = re.sub(
    r"Autonomous software engineering \(SWE\)",
    r"\\IEEEPARstart{A}{utonomous} software engineering (SWE)",
    body_content,
    count=1
)

ieee_footer = r"""
\bibliographystyle{IEEEtran}
\bibliography{neurocode}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{figures/kaggle_thumbnail_560x280.png}}]{Deepak R.}
is an undergraduate researcher pursuing a Bachelor of Engineering (B.E.) in Robotics and Automation Engineering at Dhaanish Ahmed College of Engineering, Anna University, Chennai, India. His research interests include machine learning systems, autonomous software engineering agents, heterogeneous edge computing, and static compiler analysis.
\end{IEEEbiography}

\end{document}
"""

full_ieee = ieee_header + ieee_abstract + "\n\n" + body_content + "\n\n" + ieee_footer

with open(tse_path, "w", encoding="utf-8") as f:
    f.write(full_ieee)

print(f"Successfully generated IEEE TSE / ASE manuscript: {tse_path} ({len(full_ieee.splitlines())} lines)")
