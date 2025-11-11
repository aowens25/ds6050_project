# Deep Learning for Predicting Chronic Wasting Disease (CWD) Spread in U.S. Deer Populations

**University of Virginia: DS 6050 Deep Learning**  
**Team:** Alexander Owens, Samuel Delaney, Tyler Kellogg  
**Instructor:** Heman Shakeri, PhD  
**Semester:** Fall 2025  

---

## Project Overview
This project builds a reproducible deep learning workflow to predict next-year county-level CWD risk using publicly available U.S. Geological Survey (USGS) data and environmental features.  
We compare a small multilayer perceptron (MLP) against logistic regression and random forest baselines.

---

## Dataset
- **Source:** [USGS CWD Distribution ver 3.0 (June 2025)](https://www.usgs.gov/data/chronic-wasting-disease-distribution-united-states-state-and-county-ver-30-june-2025)  
- **Additional features:** Land cover fractions, annual temperature, elevation, deer density proxy, human population density.  
- **Target:** Binary flag indicating whether a new county detection occurs in the following year.  

---

