import * as vscode from 'vscode';
import * as path from 'path';
import { pickModel, showError, showInfo, withProgress } from './ui';
import { createVirtualEnv, installRequirements, findPython, getVenvPython, detectDevice } from './envManager';
import { getAvailableModels, getModelById, downloadModel, isModelDownloaded, ModelInfo } from './modelManager';
import * as fs from 'fs';
import { PythonServerManager } from './serverManager';
import { predictAtCursor } from './predict';

const STATE_KEYS = {
  ENV_PATH: 'smallCoder.envPath',
  MODEL_ID: 'smallCoder.modelId',
  DEVICE: 'smallCoder.device',
};

let serverManager: PythonServerManager | undefined;
let outputChannel: vscode.OutputChannel;
let statusBar: vscode.StatusBarItem;

function updateStatusBar(modelLabel: string, device: string, running: boolean): void {
  const runIcon = running ? '$(circle-filled)' : '$(circle-outline)';
  const isCuda = device.toLowerCase() === 'cuda';
  // Highlight GPU runs with a "zap" codicon in green; CPU keeps the neutral board icon.
  const deviceIcon = isCuda ? '$(zap)' : '$(circuit-board)';
  statusBar.text = `${deviceIcon} [${modelLabel}] ${runIcon} ${device.toUpperCase()}`;
  statusBar.color = isCuda || running ? new vscode.ThemeColor('charts.green') : undefined;
}

export async function activate(context: vscode.ExtensionContext) {
  outputChannel = vscode.window.createOutputChannel('SmallCoder');
  outputChannel.appendLine('SmallCoder extension activating...');

  serverManager = new PythonServerManager(outputChannel, context.extensionPath);

  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.command = 'small-coder-extension.openLogs';
  statusBar.tooltip = 'SmallCoder — click to open logs';
  const savedModel = context.globalState.get<string>(STATE_KEYS.MODEL_ID) ?? 'none';
  const savedDevice = context.globalState.get<string>(STATE_KEYS.DEVICE) ?? 'cpu';
  updateStatusBar(getModelById(savedModel)?.label ?? savedModel, savedDevice, false);
  statusBar.show();
  context.subscriptions.push(statusBar);

  serverManager.onStart = (model, device) => {
    // The server may be started with a local filesystem path, not the model id.
    // Resolve the display label from the saved model id so the status bar keeps
    // showing a short label (and the device) instead of a long setup path.
    const savedId = context.globalState.get<string>(STATE_KEYS.MODEL_ID) ?? model;
    updateStatusBar(getModelById(savedId)?.label ?? savedId, device, true);
  };
  serverManager.onStop = () => {
    const m = context.globalState.get<string>(STATE_KEYS.MODEL_ID) ?? 'none';
    const d = context.globalState.get<string>(STATE_KEYS.DEVICE) ?? 'cpu';
    updateStatusBar(getModelById(m)?.label ?? m, d, false);
  };

  context.subscriptions.push(
    vscode.commands.registerCommand('small-coder-extension.setupRuntime', async () => {
      await setupRuntime(context);
    }),
    vscode.commands.registerCommand('small-coder-extension.predict', async () => {
      await predictCommand(context);
    }),
    vscode.commands.registerCommand('small-coder-extension.forceCpu', async () => {
      await forceCpu(context);
    }),
    vscode.commands.registerCommand('small-coder-extension.forceCuda', async () => {
      await forceCuda(context);
    }),
    vscode.commands.registerCommand('small-coder-extension.downloadModel', async () => {
      await downloadModelCommand(context);
    }),
    vscode.commands.registerCommand('small-coder-extension.openLogs', async () => {
      outputChannel.show(true);
    }),
  );
}

async function setupRuntime(context: vscode.ExtensionContext) {
  try {
    const models = getAvailableModels();
    const model = await pickModel(models, context.globalState.get<string>(STATE_KEYS.MODEL_ID) ?? 'smallcoder-tiny');
    if (!model) {
      return;
    }

    const pythonPath = await findPython();
    if (!pythonPath) {
      showError('Python not found. Install Python 3.10+ and retry.');
      return;
    }

    const envPath = await promptForEnvPath(context, pythonPath);
    if (!envPath) {
      return;
    }

    await withProgress('Creating SmallCoder runtime', async (progress) => {
      progress.report({ message: 'Creating virtual environment' });
      await createVirtualEnv(envPath, pythonPath, outputChannel);

      progress.report({ message: 'Installing runtime requirements' });
      await installRequirements(envPath, `${context.extensionPath}/server/requirements.txt`, outputChannel);
    });

    context.globalState.update(STATE_KEYS.ENV_PATH, envPath);
    context.globalState.update(STATE_KEYS.MODEL_ID, model.id);

    const detectedDevice = detectDevice(envPath);
    context.globalState.update(STATE_KEYS.DEVICE, detectedDevice);
    updateStatusBar(getModelById(model.id)?.label ?? model.id, detectedDevice, false);
    showInfo(`SmallCoder: detected device → ${detectedDevice.toUpperCase()}`);

    await downloadModel(
      model,
      context.storageUri?.fsPath ?? `${context.globalStoragePath}/models`,
      outputChannel,
      getVenvPython(envPath),
    );

    showInfo('SmallCoder runtime created. Use Predict from cursor to generate completions.');
  } catch (error) {
    showError(`Setup runtime failed: ${String(error)}`);
  }
}

async function promptForEnvPath(context: vscode.ExtensionContext, pythonPath: string) {
  const storageFsPath = context.globalStorageUri?.fsPath ?? context.globalStoragePath;
  const defaultPath = path.join(storageFsPath, 'small-coder-env');
  const envPath = await vscode.window.showInputBox({
    title: 'SmallCoder virtual environment path',
    value: context.globalState.get<string>(STATE_KEYS.ENV_PATH) ?? defaultPath,
    prompt: `Python runtime: ${pythonPath}`,
  });

  return envPath?.trim();
}

async function predictCommand(context: vscode.ExtensionContext) {
  try {
    const envPath = context.globalState.get<string>(STATE_KEYS.ENV_PATH);
    if (!envPath) {
      showError('SmallCoder runtime not configured. Run Setup runtime first.');
      return;
    }

    const currentModelId = context.globalState.get<string>(STATE_KEYS.MODEL_ID) ?? 'smallcoder-tiny';
    const model = getModelById(currentModelId);

    const modelsStorage = context.storageUri?.fsPath ?? `${context.globalStoragePath}/models`;

    // Determine what to pass to server: ModelInfo or model path/string.
    let modelOrArg: ModelInfo | string = model ?? currentModelId;

    // If we have a ModelInfo, prefer local downloaded folder if exists.
    if (model) {
      const localCandidate = path.join(modelsStorage, model.id);
      if (fs.existsSync(localCandidate)) {
        const downloaded = await isModelDownloaded(localCandidate);
        if (!downloaded) {
          showError(
            `Model "${model.label}" was not fully downloaded. Run "SmallCoder: Download Model" to download it.`,
          );
          return;
        }
        modelOrArg = localCandidate;
      } else if (model.id.startsWith('local-')) {
        const alt = model.id.replace(/^local-/, '');
        const altPath = path.join(modelsStorage, alt);
        if (fs.existsSync(altPath)) {
          modelOrArg = altPath;
        }
      }
    } else {
      // No ModelInfo defined. currentModelId may be raw model repo id or local- prefixed name.
      // If it's a path that exists, use it. If 'local-<name>', map to modelsStorage/<name>.
      if (fs.existsSync(currentModelId)) {
        modelOrArg = currentModelId;
      } else if (currentModelId.startsWith('local-')) {
        const alt = currentModelId.replace(/^local-/, '');
        const altPath = path.join(modelsStorage, alt);
        if (fs.existsSync(altPath)) {
          modelOrArg = altPath;
        }
      } else {
        // fallback: try modelsStorage/currentModelId
        const candidate = path.join(modelsStorage, currentModelId);
        if (fs.existsSync(candidate)) {
          modelOrArg = candidate;
        }
      }
    }

    let port: number | undefined;
    await withProgress(
      'SmallCoder: Loading model (first run downloads ~3 GB — check SmallCoder output for progress)',
      async () => {
        port = await serverManager?.startServer(
          envPath,
          modelOrArg,
          context.globalState.get<string>(STATE_KEYS.DEVICE) ?? 'cpu',
        );
      },
    );
    if (!port) {
      showError('Unable to start SmallCoder server.');
      return;
    }

    await predictAtCursor(port, outputChannel);
  } catch (error) {
    showError(`Prediction failed: ${String(error)}`);
  }
}

async function forceCpu(context: vscode.ExtensionContext) {
  context.globalState.update(STATE_KEYS.DEVICE, 'cpu');
  await serverManager?.stopServer();
  const m = context.globalState.get<string>(STATE_KEYS.MODEL_ID) ?? 'none';
  updateStatusBar(getModelById(m)?.label ?? m, 'cpu', false);
  showInfo('SmallCoder: device set to CPU. Server will use CPU on next prediction.');
}

async function forceCuda(context: vscode.ExtensionContext) {
  const detectedDevice = context.globalState.get<string>(STATE_KEYS.DEVICE);
  if (detectedDevice !== 'cuda') {
    const confirmed = await vscode.window.showWarningMessage(
      'CUDA was not detected during Setup Runtime. Force CUDA anyway?',
      'Yes, force CUDA',
      'Cancel',
    );
    if (confirmed !== 'Yes, force CUDA') {
      return;
    }
  }
  context.globalState.update(STATE_KEYS.DEVICE, 'cuda');
  await serverManager?.stopServer();
  const m = context.globalState.get<string>(STATE_KEYS.MODEL_ID) ?? 'none';
  updateStatusBar(getModelById(m)?.label ?? m, 'cuda', false);
  showInfo('SmallCoder: device set to CUDA. Server will use CUDA on next prediction.');
}

async function downloadModelCommand(context: vscode.ExtensionContext) {
  const currentModelId = context.globalState.get<string>(STATE_KEYS.MODEL_ID) ?? 'smallcoder-tiny';
  const model = getModelById(currentModelId);
  if (!model) {
    showError('Selected model not found. Run Setup runtime again.');
    return;
  }

  const envPath = context.globalState.get<string>(STATE_KEYS.ENV_PATH);
  const modelPath = await downloadModel(
    model,
    context.storageUri?.fsPath ?? `${context.globalStoragePath}/models`,
    outputChannel,
    envPath ? getVenvPython(envPath) : undefined,
  );
  showInfo('Model download completed.');

  // Load the model right after downloading by starting the inference server.
  if (!envPath) {
    showError('Runtime not configured — run "Setup runtime" first to load the model.');
    return;
  }

  const device = context.globalState.get<string>(STATE_KEYS.DEVICE) ?? 'cpu';
  let port: number | undefined;
  await withProgress(
    'SmallCoder: Loading model (first run downloads ~3 GB — check SmallCoder output for progress)',
    async () => {
      port = await serverManager?.startServer(envPath, modelPath, device);
    },
  );
  if (port) {
    showInfo('Model downloaded and loaded.');
  } else {
    showError('Model downloaded, but the SmallCoder server failed to start.');
  }
}

export function deactivate() {
  serverManager?.stopServer();
}
