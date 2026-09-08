# 🛰️ Deep Learning Super Resolution Mapping (SRM)
### From Medium-Resolution Sentinel-2 Satellite Imagery to Sub-4m Enhanced Products

[![NTRO](https://img.shields.io/badge/NTRO-SIH%202026-orange?style=flat-square&logo=satellite)](https://www.ntro.gov.in)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+cu124-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20RTX%204050%20Ada-76b900?style=flat-square&logo=nvidia)](https://www.nvidia.com)

> **SIH 2026 Project** | Problem Statement ID: **26142**
> **Organization:** National Technical Research Organisation (NTRO)
> **Team:** KCUF Coders

---

## 📋 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Key Innovations](#-key-innovations)
- [Architecture](#️-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Visual Results](#️-visual-results)
- [Evaluation Metrics](#-evaluation-metrics)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [References](#-references)

---

## 🌍 Overview

India and global defense/research agencies rely on Sentinel-2 satellite imagery for agriculture monitoring, land-cover mapping, urban planning, disaster assessment, and environmental observation. However, at **10–30 metre resolution**, fine-scale details such as narrow roads, small buildings, field boundaries, and localized damage remain undetectable.

This project builds a **deep learning super-resolution framework** that transforms Sentinel-2 imagery from **10m → 2.5m resolution (4× upscaling)** using **SwinIR Transformer + GAN training**, followed by **SegFormer semantic segmentation** for land cover classification.

---

## 🎯 The Problem

Without high-resolution satellite data:

- Fine-scale features (narrow roads, small buildings) are invisible
- Crop boundary detection is inaccurate
- Disaster damage assessment lacks spatial precision
- Urban mapping misses critical infrastructure details

Commercial high-resolution satellites (Pleiades, WorldView) are expensive and have limited coverage. Our approach **extracts more value from freely available Sentinel-2 data** using deep learning.

---

## 💡 Key Innovations