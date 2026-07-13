# Gradio server reference (Windows)

Use PowerShell for the commands below.

## 1. Open the project directory

```powershell
Set-Location "C:\Users\Jack\Documents\GitHub\Chef-s-Assistant-AI-Driven-Recipe-Generation-for-Dynamic-Menu-Planning"
```

## 2. Start the server

```powershell
.\run_gradio.ps1
```

Keep that PowerShell window open. When startup finishes, open:

<http://127.0.0.1:7860>

The complete recipe index is opened when a dataset feature is first used. If the
index does not exist yet, the application builds it automatically from the CSV;
this one-time first run takes longer. The Llama model is loaded into GPU memory when
an AI feature is first used, so the first AI request also takes longer than later
requests.

If PowerShell blocks local scripts because of its execution policy, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gradio.ps1
```

## 3. Stop the server normally

Return to the PowerShell window running the server and press:

```text
Ctrl+C
```

Wait until the PowerShell prompt returns. Closing that terminal also stops a
foreground server, but `Ctrl+C` is the preferred clean shutdown method. Stopping
the server releases the GPU memory used by the model.

## 4. Restart the server

Stop it with `Ctrl+C`, wait for the prompt, and run:

```powershell
.\run_gradio.ps1
```

Restart the server after changing Python code, dependency versions, environment
variables, the base model, or the LoRA adapter.

## 5. Check whether it is running

Open <http://127.0.0.1:7860>, or check from another PowerShell window:

```powershell
(Invoke-WebRequest -Uri "http://127.0.0.1:7860" -UseBasicParsing).StatusCode
```

`200` means the web server is responding.

To inspect the process listening on port 7860:

```powershell
Get-NetTCPConnection -LocalPort 7860 -State Listen |
    Select-Object LocalAddress, LocalPort, OwningProcess
```

## 6. Recover when port 7860 is already in use

First inspect the process ID:

```powershell
$listener = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue
$listener | Select-Object LocalAddress, LocalPort, OwningProcess
Get-Process -Id $listener.OwningProcess
```

Only if that process is the old Chef's Assistant server, stop it and start again:

```powershell
Stop-Process -Id $listener.OwningProcess
.\run_gradio.ps1
```

Do not stop the process if it belongs to a different application. In that case,
start Chef's Assistant on another port:

```powershell
.\run_gradio.ps1 --port 7861
```

Then open <http://127.0.0.1:7861>.

## 7. Optional temporary public link

```powershell
.\run_gradio.ps1 --share
```

Gradio will print a temporary public URL. Anyone with that URL may be able to use
the model while the server is running. Use it only when public access is intended,
and stop the server with `Ctrl+C` when finished.

## Quick reference

| Action | Command |
| --- | --- |
| Start locally | `.\run_gradio.ps1` |
| Stop | Press `Ctrl+C` in the server terminal |
| Restart | Press `Ctrl+C`, then run `.\run_gradio.ps1` |
| Start on another port | `.\run_gradio.ps1 --port 7861` |
| Create temporary public URL | `.\run_gradio.ps1 --share` |
| Local address | <http://127.0.0.1:7860> |
