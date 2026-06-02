import * as vscode from 'vscode';
import * as path from 'path';
import { pickModel, showError, showInfo, withProgress } from './ui';
import { createVirtualEnv, installRequirements, findPython } from './envManager';
import { getAvailableModels, getModelById, downloadModel } from './modelManager';
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

export async function activate(context: vscode.ExtensionContext) {
  outputChannel = vscode.window.createOutputChannel('SmallCoder');
  outputChannel.appendLine('SmallCoder extension activating...');

  serverManager = new PythonServerManager(outputChannel, context.extensionPath);

  context.subscriptions.push(
    vscode.commands.registerCommand('small-coder-extension.setupRuntime', async () => {
      await setupRuntime(context);
    }),
    vscode.commands.registerCommand('small-coder-extension.predict', async () => {
      await predictCommand(context);
    }),
    vscode.commands.registerCommand('small-coder-extension.toggleDevice', async () => {
      await toggleDevice(context);
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
    context.globalState.update(
      STATE_KEYS.DEVICE,
      vscode.workspace.getConfiguration().get<string>('smallCoder.device', 'cpu'),
    );

    await downloadModel(model, context.storageUri?.fsPath ?? `${context.globalStoragePath}/models`, outputChannel);

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
    let modelOrArg: string | typeof model = model ?? currentModelId;

    // If we have a ModelInfo, prefer local downloaded folder if exists.
    if (model) {
      const localCandidate = path.join(modelsStorage, model.id);
      if (fs.existsSync(localCandidate)) {
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

    const port = await serverManager?.startServer(
      envPath,
      modelOrArg,
      context.globalState.get<string>(STATE_KEYS.DEVICE) ?? 'cpu',
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

async function toggleDevice(context: vscode.ExtensionContext) {
  const config = vscode.workspace.getConfiguration();
  const current = config.get<string>('smallCoder.device', 'cpu');
  const next = current === 'cpu' ? 'cuda' : 'cpu';
  await config.update('smallCoder.device', next, vscode.ConfigurationTarget.Global);
  context.globalState.update(STATE_KEYS.DEVICE, next);
  showInfo(`Device set to ${next}. Restart SmallCoder server before predicting.`);
  await serverManager?.stopServer();
}

async function downloadModelCommand(context: vscode.ExtensionContext) {
  const currentModelId = context.globalState.get<string>(STATE_KEYS.MODEL_ID) ?? 'smallcoder-tiny';
  const model = getModelById(currentModelId);
  if (!model) {
    showError('Selected model not found. Run Setup runtime again.');
    return;
  }

  await downloadModel(model, context.storageUri?.fsPath ?? `${context.globalStoragePath}/models`, outputChannel);
  showInfo('Model download completed.');
}

export function deactivate() {
  serverManager?.stopServer();
}
