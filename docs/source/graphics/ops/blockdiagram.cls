\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{blockdiagram}[2020/04/21]

\LoadClass{standalone}

\usepackage{tikz}
\usetikzlibrary{positioning, fit, arrows, calc}

\pgfmathsetmacro{\blockwidth}{2.5}
\pgfmathsetmacro{\blockheight}{\blockwidth / 2}
\pgfmathsetmacro{\horzportshift}{\blockwidth * 0.15}
\pgfmathsetmacro{\vertportshift}{\blockheight *0.15}

\tikzset{
    block/.style={
        minimum width=\blockwidth cm, 
        minimum height=\blockheight cm,
        inner sep=0mm, 
        outer sep=0mm, 
        draw, 
        align=left 
    },
    >=latex,
}

\newcommand{\code}[1]{\texttt{\detokenize{#1}}}

\newcommand{\blocklabel}[2]{
    \node[
        anchor=south west, 
        inner sep=1mm, 
        outer sep=0mm
    ] (#2 label) at (#2.north west) {\code{#1}};}

\newcommand{\superblock}[3][]{
    \node[
        block, 
        inner sep=2mm, 
        fit=#1
    ] (#3) {};
    \blocklabel{#2}{#3}
}

\endinput