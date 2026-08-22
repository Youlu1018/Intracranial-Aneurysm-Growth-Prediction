# Intracranial Aneurysm Growth Research Prototype

This Streamlit app deploys the locked SVM described in the revised manuscript.
It is an exploratory **research-use-only** prototype and is **not for clinical
diagnosis, treatment selection, surveillance planning, or patient-management
decisions**.

## Model scope

- Outcome: imaging-detected growth during available follow-up, defined in the
  study as a >=1 mm increase in maximum diameter, a >0.5 mm increase in two
  perpendicular diameters, or a >10% relative increase in volume under the
  same imaging modality.
- Population: conservatively managed saccular unruptured intracranial
  aneurysms matching the study eligibility criteria.
- Validation: single-center internal validation only. Independent multicenter
  and prospective validation is required before any clinical use.
- Time horizon: the displayed value is not a validated 12-month, 24-month, or
  lifetime risk estimate. Median cohort follow-up was 375 days.

The seven locked inputs are alcohol use, tobacco use, aneurysm diameter,
maximum diameter, aneurysm volume, undulation index (UI), and nonsphericity
index (NSI). Continuous inputs are restricted to the observed range in the
complete 784-aneurysm cohort:

| Input | Allowed range |
|---|---:|
| Aneurysm diameter (mm) | 1.43265-8.80232 |
| Maximum diameter (mm) | 2.45312-13.6368 |
| Aneurysm volume (mm3) | 1.38736-461.787 |
| UI | 0.0408163-0.34375 |
| NSI | 0.136346-0.502377 |

Alcohol and tobacco use are binary (No = 0, Yes = 1).

## GitHub and Streamlit Community Cloud deployment

1. Commit this directory to GitHub. The real key file and the encrypted
   source-data recovery archive are excluded by `.gitignore`.
2. In Streamlit Community Cloud, choose `app.py` as the entry point and select
   Python 3.13 in Advanced settings.
3. In Advanced settings -> Secrets, paste the following setting, replacing the
   placeholder with the value from `.svm_deployment_key_DO_NOT_UPLOAD.txt`:

   ```toml
   MODEL_ENCRYPTION_KEY = "PASTE_PRIVATE_KEY_HERE"
   ```

4. Deploy. The app checks the encrypted bundle's size and SHA-256 digest before
   authenticated Fernet decryption in memory.

Never commit or upload `.svm_deployment_key_DO_NOT_UPLOAD.txt`. Keeping the key
beside the app is for local handoff only; encryption provides no confidentiality
if the key and ciphertext are published together. Streamlit Community Cloud
must receive the key through its Secrets interface.

## Local run

Install dependencies and run from this directory:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

For local convenience, `app.py` can read the ignored key file in this folder.
It also supports the `MODEL_ENCRYPTION_KEY` environment variable and
Streamlit's `.streamlit/secrets.toml` mechanism.

## Data protection

The public deployment bundle contains only the locked SVM, the training-derived
transformer, 30 standardized K-means centers used as the SHAP background, and
precomputed permutation importance. It contains no individual-level source
records.

`SVMappdata/traindataSVM.joblib.enc` is an encrypted local recovery archive of
the original individual-level data. It is not read by the app and is excluded
from GitHub. Its matching integrity manifest is also local-only.

Verify or restore that local archive only outside the deployment directory:

```powershell
python manage_encrypted_assets.py verify
python manage_encrypted_assets.py decrypt --output "D:\private\traindataSVM.joblib"
```
