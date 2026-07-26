# Deployment Protocol: Streamlit Cloud Implementation

## 1. Abstract
This document delineates the procedure for deploying the TerraSight Earth Observation (EO) Intelligence Platform to Streamlit Cloud. The deployment utilizes the optimized production assets located within the `hf-spaces-demo/` directory.

## 2. Infrastructure Requirements
*   GitHub Repository: [VED-VIVEK-TALMALEY/TerraSight](https://github.com/VED-VIVEK-TALMALEY/TerraSight)
*   Streamlit Cloud Account (Community or Enterprise)
*   Standard Web Browser

## 3. Configuration Procedure

### 3.1 Account Integration
1.  Navigate to [share.streamlit.io](https://share.streamlit.io).
2.  Authenticate using the GitHub account associated with the project repository.

### 3.2 Application Initialization
1.  Select the "Create app" or "New app" interface.
2.  Utilize the "I already have an app" option if prompted.
3.  Designate the following parameters:
    *   **Repository**: `VED-VIVEK-TALMALEY/TerraSight`
    *   **Branch**: `main`
    *   **Main file path**: `hf-spaces-demo/app.py`

### 3.3 Advanced Configuration (Optional)
If the deployment requires environmental variables (e.g., specific API endpoints), access the "Settings" menu within the Streamlit deployment dashboard and input the necessary key-value pairs into the "Secrets" section.

## 4. Deployment Execution
1.  Select "Deploy!".
2.  The system will initialize the environment, resolve dependencies specified in `hf-spaces-demo/requirements.txt`, and execute the application.
3.  The public interface will be accessible via the generated subdomain (e.g., `terrasight.streamlit.app`).

## 5. Continuous Integration / Continuous Deployment (CI/CD)
The deployment is configured for automatic synchronization. Any commits pushed to the `hf-spaces-demo/` directory on the `main` branch will trigger an immediate update of the live production environment.

