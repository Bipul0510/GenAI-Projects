# Beginner Setup Guide: Research Paper RAG Assistant

This guide assumes that you are on Windows and are new to Python and Azure. Follow the steps in order. Do not skip the Azure configuration step: the application cannot answer questions until it has your Azure details.

## 1. Understand what you have

Your project folder is:

```text
C:\Users\bipul\Documents\Codex\2026-08-29\new-chat
```

Inside it you should see:

| Item | What it does | Do you need to edit it? |
| --- | --- | --- |
| `app.py` | The complete application code. | No. |
| `requirements.txt` | List of Python packages the app needs. | No. |
| `.env.example` | Safe template for your Azure details. | No; copy it to `.env`. |
| `.env` | Your private Azure settings. | Yes, once. Create it in Step 5. |
| `data/` | The three research papers used by the app. | Leave the PDFs here. |
| `storage/` | Created automatically after the first successful question. | Do not create it yourself. |
| `README.md` | Portfolio overview and technical documentation. | Optional reading. |

Important: do **not** rename or move the PDFs in `data/`. Never share or upload your `.env` file because it contains your Azure key.

## 2. Create or open your Azure OpenAI resource

If you already have an Azure OpenAI resource, open it in the Azure portal and continue with the next step. If you do not have one:

1. Sign in at [Azure Portal](https://portal.azure.com/).
2. Search for **Azure OpenAI** or open **Azure AI Foundry** from your Azure subscription.
3. Create or select an Azure OpenAI resource. You may need an active Azure subscription and permission from your Azure administrator.
4. Open the resource after it is created.

The exact Azure screens can change, but look for **Deployments**, **Model deployments**, or **Azure AI Foundry**. The important outcome is two model deployments: one chat model and one embedding model.

## 3. Create two model deployments

Create these two deployments inside the same Azure OpenAI resource:

1. **Chat deployment**: choose a small text/chat model that is available in your Azure region. Give it a simple deployment name, such as `rag-chat`.
2. **Embedding deployment**: choose a text embedding model available in your region. Give it a simple deployment name, such as `rag-embeddings`.

Write down the two **deployment names** exactly. A deployment name is the label you choose in Azure; it may be different from the model name. The app needs the deployment names, not just model names.

Azure OpenAI uses your resource endpoint, API key, and model deployment identifier for requests. [Official OpenAI Azure guidance](https://developers.openai.com/api/reference/ruby) shows this endpoint/key/deployment pattern.

## 4. Copy your Azure endpoint and key

In your Azure OpenAI resource:

1. Find **Keys and Endpoint** in the resource menu.
2. Copy the value labelled **Endpoint**.
3. Copy either **KEY 1** or **KEY 2**.
4. Keep the browser tab open, or paste these temporarily into a private note. Do not share the key in chat, email, GitHub, screenshots, or your resume.

## 5. Install Python, if needed

1. Press the Windows key, type `PowerShell`, and open it.
2. Run:

```powershell
py --version
```

3. If you see Python `3.10` or newer, move to Step 6.
4. If you see an error, install Python 3.10 or newer from [python.org](https://www.python.org/downloads/). During installation, tick **Add Python to PATH**. Close and reopen PowerShell afterwards, then run `py --version` again.

## 6. Open the project folder in PowerShell

Copy and run this command exactly:

```powershell
cd "C:\Users\bipul\Documents\Codex\2026-08-29\new-chat"
```

Check that you are in the correct folder:

```powershell
dir
```

You should see `app.py`, `requirements.txt`, `data`, and `README.md`.

## 7. Create a private Python environment and install packages

Run these commands one at a time:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

What this does:

- `.venv` is a private folder for this project’s Python packages.
- The second command activates that folder.
- The final command installs Streamlit, Azure OpenAI support, FAISS, PDF reading, and related packages.

If PowerShell says that scripts are disabled, run this once in the same PowerShell window, then repeat the activation command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 8. Create your private `.env` settings file

Run:

```powershell
Copy-Item .env.example .env
notepad .env
```

Notepad will open. Replace the placeholder text with your own Azure values. Here is an example layout - the names on the right are examples only:

```env
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE-NAME.openai.azure.com/
AZURE_OPENAI_API_KEY=paste-your-key-here
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=rag-chat
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=rag-embeddings
```

Rules:

- Do not put quotation marks around values.
- Do not add spaces before or after `=`.
- Replace `rag-chat` and `rag-embeddings` with your exact Azure deployment names.
- Keep the endpoint supplied by Azure, including `https://`.
- Start with the API version in the example. If Azure tells you it is unsupported, use the API version supported by your resource.

Save the file with **Ctrl + S**, then close Notepad. The `.env` file stays on your computer and is excluded from Git by `.gitignore`.

## 9. Start the application

In the same PowerShell window, where `(.venv)` is visible at the beginning of the line, run:

```powershell
streamlit run app.py
```

The terminal will show a local address, usually:

```text
http://localhost:8501
```

Open that address in your browser if it does not open automatically. Keep the PowerShell window open while you use the app. To stop it later, click the PowerShell window and press **Ctrl + C**.

## 10. Ask your first question

1. In the browser, click one of the sample questions, for example: **What is multi-head attention and why is it beneficial?**
2. Click **Ask question**.
3. On this first question, wait while the app reads the PDFs, creates embeddings through Azure, and saves a local search index. This can take a few minutes.
4. Read the answer.
5. Under **Sources retrieved**, confirm that a paper name and page number are shown.
6. Open **View retrieved excerpts** to see the exact supporting text.

After the first successful question, a `storage` folder will appear. The next app launch reuses this index, so it is faster and does not recreate embeddings unnecessarily.

## 11. Test the project before adding it to your portfolio

Run these questions one by one:

1. What are the main components of a RAG model, and how do they interact?
2. What are the two sub-layers in each encoder layer of the Transformer model?
3. Explain positional encoding in Transformers and why it is necessary.
4. What is multi-head attention and why is it beneficial?
5. What is few-shot learning, and how does GPT-3 implement it during inference?

For each answer, check that:

- It answers the question in simple language.
- The correct paper is shown in the source list.
- At least one page number is shown.
- The retrieved excerpt supports the answer.

Then test this unrelated question:

```text
What is the capital of France?
```

The app should say it cannot find the answer in the supplied papers. That is good RAG behavior: it avoids pretending it knows something unsupported by its documents.

## 12. Use Refresh index only when needed

Click **Refresh index** only after you add, replace, or change a PDF in the `data` folder. It sends the paper chunks to your Azure embedding deployment again, so it can create extra Azure usage costs.

## 13. Common problems and easy fixes

| What you see | Likely cause | What to do |
| --- | --- | --- |
| `py` is not recognized | Python is missing or not on PATH. | Install Python, enable **Add Python to PATH**, reopen PowerShell. |
| `streamlit` is not recognized | The private environment is not active or packages are not installed. | Run `.\.venv\Scripts\Activate.ps1`, then `pip install -r requirements.txt`. |
| The app says Azure configuration is missing | `.env` was not created or has blank values. | Repeat Step 8 and save `.env` in the same folder as `app.py`. |
| The app reports Azure connection/configuration error | Endpoint, API version, key, or deployment name is incorrect. | Carefully compare every `.env` value with Azure. Use deployment names exactly. |
| `DeploymentNotFound` | A model deployment name is wrong. | Check Azure deployments and correct the chat or embedding deployment name. |
| First question is slow | The local index is being built. | Wait for it to finish. Future runs will reuse `storage/`. |
| PDF errors | A source PDF was moved or renamed. | Restore the three PDFs in the `data` folder with their original filenames. |

## 14. Prepare your portfolio evidence

When the app works, take four screenshots:

1. The home screen with a sample question.
2. A complete cited answer.
3. The retrieved-excerpts section expanded.
4. The sidebar showing **Index: Ready**.

Before pushing to GitHub, make sure `.env`, `.venv`, and `storage` are not added. Your `.gitignore` is already set up to exclude them.

## Your normal daily routine

After the first-time setup, each day you want to run the project:

```powershell
cd "C:\Users\bipul\Documents\Codex\2026-08-29\new-chat"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

That is all. You only need to recreate the environment or edit `.env` if your computer, Azure key, or Azure deployment changes.
