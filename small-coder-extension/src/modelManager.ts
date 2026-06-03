import * as fs from 'fs/promises';
import * as path from 'path';
import { spawn } from 'child_process';
import { OutputChannel } from 'vscode';

export interface ModelInfo {
  id: string;
  label: string;
  description: string;
  size: string;
  url?: string;
  checksum?: string;
}

const MODELS: ModelInfo[] = [
  {
    id: 'Juanxxo/qwen2.5-1.5B-code-adapter',
    label: 'Qwen Code Adapted',
    description: 'A model trained with code and documentation from Claudios',
    size: '53MB',
    url: 'Juanxxo/qwen2.5-1.5B-code-adapter',
    checksum: '',
  },
];

export function getAvailableModels(): ModelInfo[] {
  return [...MODELS];
}

export function getModelById(id: string): ModelInfo | undefined {
  return MODELS.find((model) => model.id === id);
}

function spawnPython(pythonPath: string, script: string, args: string[], output: OutputChannel): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(pythonPath, [script, ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';

    proc.stdout?.on('data', (chunk) => {
      const data = chunk.toString();
      stdout += data;
      output.appendLine(data);
    });

    proc.stderr?.on('data', (chunk) => {
      const data = chunk.toString();
      stderr += data;
      output.appendLine(`[ERROR] ${data}`);
    });

    proc.on('close', (code) => {
      if (code === 0) {
        resolve(stdout.trim());
      } else {
        reject(new Error(`Process exited with code ${code}: ${stderr}`));
      }
    });

    proc.on('error', reject);
  });
}

async function findPythonPath(): Promise<string> {
  // Silently try python3, then fall back to python
  try {
    const silentOutput: OutputChannel = {
      name: 'silent',
      append: () => {},
      appendLine: () => {},
      clear: () => {},
      show: () => {},
      hide: () => {},
      replace: () => {},
      dispose: () => {},
    };
    await spawnPython('python3', '-c', ['import sys'], silentOutput);
    return 'python3';
  } catch {
    return 'python';
  }
}

export async function downloadModel(
  model: ModelInfo,
  storagePath: string,
  output: OutputChannel,
  venvPythonPath?: string,
): Promise<string> {
  await fs.mkdir(storagePath, { recursive: true });
  const modelPath = path.join(storagePath, model.id);
  await fs.mkdir(modelPath, { recursive: true });

  const metadata = {
    id: model.id,
    label: model.label,
    description: model.description,
    downloadedAt: new Date().toISOString(),
  };

  await fs.writeFile(path.join(modelPath, 'model-metadata.json'), JSON.stringify(metadata, null, 2), 'utf8');
  output.appendLine(`Model ${model.id} prepared at ${modelPath}`);

  if (model.url) {
    const extensionPath = path.dirname(path.dirname(__filename));
    const downloadScript = path.join(extensionPath, 'server', 'download_model.py');
    // Prefer venv Python (has huggingface_hub installed) over system Python
    const pythonPath = venvPythonPath ?? (await findPythonPath());
    output.appendLine(`Downloading model ${model.id} from HuggingFace Hub using ${pythonPath}...`);
    const downloadedPath = await spawnPython(
      pythonPath,
      downloadScript,
      ['--model-id', model.url, '--cache-dir', modelPath],
      output,
    );
    output.appendLine(`Model ${model.id} downloaded successfully from ${downloadedPath}`);
  }

  return modelPath;
}

export async function isModelDownloaded(modelPath: string): Promise<boolean> {
  try {
    const entries = await fs.readdir(modelPath);
    // A properly downloaded HF model/adapter has more than just our metadata file
    return entries.some((e) => e !== 'model-metadata.json');
  } catch {
    return false;
  }
}
