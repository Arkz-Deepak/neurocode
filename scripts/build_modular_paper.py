#!/usr/bin/env python3
"""
Builds the modular IEEE conference paper using the template from Few_shot_learning___Lightweight_VLA__GVA_.zip.
"""

import os
import shutil
import re
from pathlib import Path

src_zip_dir = Path("/tmp/vla_template")
out_dir = Path("paper_modular_ieee")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "sections").mkdir(exist_ok=True)
(out_dir / "figures").mkdir(exist_ok=True)

# Copy figures
for img in Path("paper_mlsys/figures").glob("*.png"):
    shutil.copy(img, out_dir / "figures" / img.name)

# Copy class and bst files
for f in ["ieeeconf.cls", "IEEEtranS.bst", "IEEEabrv.bib"]:
    if (src_zip_dir / f).exists():
        shutil.copy(src_zip_dir / f, out_dir / f)

# Copy bibliography
shutil.copy("paper_mlsys/neurocode.bib", out_dir / "references.bib")

# Read mlsys source
with open("paper_mlsys/neurocode_mlsys.tex", "r", encoding="utf-8") as f:
    mlsys_text = f.read()

# Extract Abstract
abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", mlsys_text, re.DOTALL)
abstract_text = abstract_match.group(1).strip() if abstract_match else ""

# Extract Introduction (from Introduction to Background)
intro_match = re.search(r"(\\section\{Introduction\}.*?)\\section\{Background", mlsys_text, re.DOTALL)
intro_text = intro_match.group(1).strip() if intro_match else ""

# Extract Method (from Background & Systems Architecture & Implementation to Empirical Evaluation)
method_match = re.search(r"(\\section\{Background.*?)\\section\{Empirical Evaluation\}", mlsys_text, re.DOTALL)
method_text = method_match.group(1).strip() if method_match else ""

# Extract Experiments
exp_match = re.search(r"(\\section\{Empirical Evaluation\}.*?)\\section\{Related Work\}", mlsys_text, re.DOTALL)
exp_text = exp_match.group(1).strip() if exp_match else ""

# Extract Related Work
related_match = re.search(r"(\\section\{Related Work\}.*?)\\section\{Conclusion\}", mlsys_text, re.DOTALL)
related_text = related_match.group(1).strip() if related_match else ""

# Extract Conclusion
conclusion_match = re.search(r"(\\section\{Conclusion\}.*?)\\bibliography", mlsys_text, re.DOTALL)
conclusion_text = conclusion_match.group(1).strip() if conclusion_match else ""

# Write section files
with open(out_dir / "sections" / "01_Introduction.tex", "w", encoding="utf-8") as f:
    f.write(intro_text + "\n")

with open(out_dir / "sections" / "02_RelatedWork.tex", "w", encoding="utf-8") as f:
    f.write(related_text + "\n")

with open(out_dir / "sections" / "03_Method.tex", "w", encoding="utf-8") as f:
    f.write(method_text + "\n")

with open(out_dir / "sections" / "04_Experiments.tex", "w", encoding="utf-8") as f:
    f.write(exp_text + "\n")

with open(out_dir / "sections" / "05_Conclusion.tex", "w", encoding="utf-8") as f:
    f.write(r"\FloatBarrier" + "\n" + conclusion_text + "\n")

# Build main.tex
main_tex = r"""\documentclass[letterpaper, 10 pt, conference]{ieeeconf}

\IEEEoverridecommandlockouts
\overrideIEEEmargins

\usepackage[dvipsnames]{xcolor}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{etoolbox}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{colortbl}
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}
\usepackage{placeins}
\usepackage{url}
\usepackage{microtype}
\usepackage{cite}

\title{\LARGE \bf
NeuroCode: Heterogeneous Host-Device Co-Scheduling for Memory-Bounded Repository-Scale Software Engineering Agents}

\author{Deepak R.$^{1}$%
\thanks{$^{1}$Deepak R. is with the Department of Robotics and Automation Engineering, Dhaanish Ahmed College of Engineering, Anna University, Chennai 601301, India (e-mail: \texttt{wssedd18@gmail.com}).}%
}

\begin{document}

\maketitle
\thispagestyle{empty}
\pagestyle{empty}

\begin{abstract}
""" + abstract_text + r"""
\end{abstract}

\input{sections/01_Introduction}
\input{sections/02_RelatedWork}
\input{sections/03_Method}
\input{sections/04_Experiments}
\input{sections/05_Conclusion}

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""

with open(out_dir / "main.tex", "w", encoding="utf-8") as f:
    f.write(main_tex)

print(f"Modular IEEE conference manuscript successfully written to {out_dir}/")
